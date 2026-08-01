"""Optimize the exported MossFormerGAN SE 16K ONNX model."""

from collections import deque
from pathlib import Path
import sys

import onnx


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "MossFormer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "MossFormer_Optimized")

ENABLE_FP16 = False    # Mixed FP16: CUDA-safe FP32 guards are selected below.
UPGRADE_OPSET = 0


def _value_producers(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }


def _input_normalization_nodes(model: onnx.ModelProto) -> list[str]:
    """Select the raw-PCM RMS chain that must remain in FP32.

    Blanket FP16 converts the INT16 input Cast to FLOAT16. Squaring raw PCM then
    overflows for amplitudes above sqrt(65504), making the RMS infinite and the
    normalized waveform zero. Selection is topology-based and validated so an
    exporter change fails loudly instead of silently producing unsafe FP16.
    """

    producers = _value_producers(model)
    graph_inputs = {value.name for value in model.graph.input}
    boundary_values: set[str] = set()
    for node in model.graph.node:
        if not node.name.startswith("/stft_model/"):
            continue
        for value in node.input:
            producer = producers.get(value)
            if value in graph_inputs or (
                producer is not None
                and not producer.name.startswith("/stft_model/")
            ):
                boundary_values.add(value)

    selected: set[str] = set()
    visited: set[str] = set()
    queue = deque(boundary_values)
    reaches_audio = False
    while queue:
        value = queue.popleft()
        if value in visited:
            continue
        visited.add(value)
        if value == "noisy_audio":
            reaches_audio = True
            continue
        producer = producers.get(value)
        if producer is None or producer.name.startswith("/stft_model/"):
            continue
        if producer.op_type != "Constant":
            selected.add(producer.name)
        queue.extend(producer.input)

    nodes = [node.name for node in model.graph.node if node.name in selected]
    ops = [node.op_type for node in model.graph.node if node.name in selected]
    expected = {
        ("Cast", "Mul", "ReduceMean", "Add", "Sqrt", "Div"),
        ("Cast", "Reshape", "Mul", "ReduceMean", "Add", "Sqrt", "Div"),
        ("Cast", "Mul", "Mul", "ReduceMean", "Add", "Sqrt", "Div"),
        ("Cast", "Mul", "Reshape", "Mul", "ReduceMean", "Add", "Sqrt", "Div"),
    }
    if not reaches_audio or tuple(ops) not in expected:
        raise RuntimeError(
            "Unexpected input-normalization topology; refusing unsafe FP16: "
            f"reaches_audio={reaches_audio}, ops={ops}, nodes={nodes}"
        )
    return nodes


def _fp32_frontend_nodes(model: onnx.ModelProto, normalization: set[str]) -> list[str]:
    """Select one contiguous FP32 frontend through spectral compression.

    Normalized speech can create coherent STFT bins above 256. The STFT output
    itself fits FP16, but squaring it does not. Keep every data and shape node from
    input normalization through complex rescaling in FP32, then cast once at the
    first SyncANet convolution instead of creating FP16 islands between guarded ops.
    """

    producers = _value_producers(model)
    first_network_conv = next(
        node
        for node in model.graph.node
        if node.op_type == "Conv" and not node.name.startswith("/stft_model/")
    )
    ancestors: set[str] = set()
    visited: set[str] = set()
    queue = deque(first_network_conv.input[:1])
    while queue:
        value = queue.popleft()
        if value in visited:
            continue
        visited.add(value)
        producer = producers.get(value)
        if producer is None:
            continue
        ancestors.add(producer.name)
        queue.extend(producer.input)

    sensitive_nodes = [
        node.name
        for node in model.graph.node
        if node.name in ancestors
        and node.name not in normalization
        and not node.name.startswith("/stft_model/")
        and node.op_type in {"Mul", "ReduceSum", "Pow", "Clip"}
    ]
    selected = set(sensitive_nodes)
    ops = [node.op_type for node in model.graph.node if node.name in selected]
    expected = ["Mul", "ReduceSum", "Pow", "Clip", "Pow", "Mul"]
    if ops != expected or not normalization.issubset(ancestors):
        raise RuntimeError(
            "Unexpected STFT power-compression topology; refusing unsafe FP16: "
            f"ops={ops}, nodes={sensitive_nodes}, "
            f"contains_normalization={normalization.issubset(ancestors)}"
        )
    return [
        node.name
        for node in model.graph.node
        if node.name in ancestors and node.name and node.op_type != "Constant"
    ]


def _output_rescaling_nodes(model: onnx.ModelProto) -> list[str]:
    """Keep final RMS restoration in FP32 before scaling to the output range.

    Float audio inputs are lifted to int16 amplitude before RMS normalization.
    Restoring that RMS can exceed the FP16 finite range even though the next
    multiplication by 1/32768 brings the public output safely back in range.
    """

    producers = _value_producers(model)
    graph_output = next(
        (value for value in model.graph.output if value.name == "denoised_audio"),
        None,
    )
    output_scale = producers.get(graph_output.name) if graph_output is not None else None
    if output_scale is None or output_scale.op_type != "Mul":
        raise RuntimeError(
            "Unexpected output-scaling topology; refusing unsafe FP16: "
            f"output_producer={output_scale}"
        )

    dynamic_inputs = [
        value
        for value in output_scale.input
        if value in producers and producers[value].op_type != "Constant"
    ]
    if len(dynamic_inputs) != 1:
        raise RuntimeError(
            "Unexpected output-scaling inputs; refusing unsafe FP16: "
            f"node={output_scale.name}, inputs={list(output_scale.input)}"
        )
    rms_restore = producers[dynamic_inputs[0]]
    restore_input_ops = {
        producers[value].op_type
        for value in rms_restore.input
        if value in producers
    }
    if rms_restore.op_type != "Mul" or restore_input_ops != {"Div", "Sqrt"}:
        raise RuntimeError(
            "Unexpected RMS-restoration topology; refusing unsafe FP16: "
            f"node={rms_restore.name}, op={rms_restore.op_type}, "
            f"input_ops={sorted(restore_input_ops)}"
        )
    return [rms_restore.name, output_scale.name]


def fp16_node_block_list(model_path: str) -> list[str]:
    model = onnx.load(model_path, load_external_data=False)
    normalization = _input_normalization_nodes(model)
    frontend = _fp32_frontend_nodes(model, set(normalization))
    output_rescaling = _output_rescaling_nodes(model)
    blocked = frontend + output_rescaling
    print(f"  Keeping {len(blocked)} overflow-sensitive nodes in FP32: {blocked}")
    return blocked


MODEL_PLANS = {
    "MossFormerGAN_SE_16K": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=4,
        hidden_size=128,
        opt_level=1,
        only_onnxruntime=True,
        f16_node_block_list=fp16_node_block_list if ENABLE_FP16 else None,
    ),
}


CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)


