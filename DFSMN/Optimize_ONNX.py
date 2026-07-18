"""Optimize the exported DFSMN ONNX model."""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "DFSMN_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "DFSMN_Optimized")

ENABLE_FP16 = False   # Mixed FP16/FP32 CUDA graph; see FP16_CUDA_Test/README.md.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Select complete FP32 islands for the wide fbank and PCM output paths.

    The export normalizes PCM before the fused analysis convolution, allowing that
    expensive convolution and the DFSMN mask network to run in FP16. Squaring the
    fbank bins and restoring int16-domain power intentionally produces values far
    beyond FP16, so the whole square-to-log chain remains FP32. The terminal waveform
    scale and clip also remain FP32 to prevent 32767 rounding to 32768 before INT16.
    """
    model = onnx.load(src_path, load_external_data=False)
    graph = model.graph
    nodes = list(graph.node)
    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in graph.initializer
    }

    expected_scalars = {
        "input_scale": 1.0 / 32768.0,
        "input_power_scale": 32768.0 * 32768.0,
        "output_scale": 32768.0,
    }
    for name, expected in expected_scalars.items():
        value = initializers.get(name)
        if value is None or value.size != 1 or float(value.reshape(-1)[0]) != expected:
            raise RuntimeError(
                f"Unsafe DFSMN FP16 source: expected {name}={expected}, got {value}"
            )

    producers = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for value in node.input:
            consumers.setdefault(value, []).append(node)

    analysis_conv = next(
        (
            node for node in nodes
            if node.op_type == "Conv" and "analysis_conv_weight" in node.input
        ),
        None,
    )
    if analysis_conv is None:
        raise RuntimeError("Could not identify the DFSMN fused analysis convolution")
    normalize = producers.get(analysis_conv.input[0])
    analysis_users = consumers.get(analysis_conv.output[0], [])
    analysis_split = next((node for node in analysis_users if node.op_type == "Split"), None)
    if (
        normalize is None
        or normalize.op_type != "Mul"
        or "input_scale" not in normalize.input
        or analysis_split is None
        or len(analysis_split.output) != 2
    ):
        raise RuntimeError(
            "Unsafe DFSMN FP16 source: expected input-scale Mul -> analysis Conv -> 2-way Split"
        )

    fbank_split = next(
        (
            node for node in consumers.get(analysis_split.output[0], [])
            if node.op_type == "Split" and len(node.output) == 2
        ),
        None,
    )
    if fbank_split is None:
        raise RuntimeError("Could not identify the packed real/imaginary fbank Split")

    # Cast the packed fbank tensor once, then keep its real/imaginary Split and
    # every descendant through Log in one FP32 island. This avoids both a second
    # boundary cast and any FP32 -> FP16 -> FP32 sandwich, while the packed
    # mask-STFT output remains FP16 end to end.
    reachable = set(fbank_split.output)
    fbank_nodes: list[onnx.NodeProto] = [fbank_split]
    reached_log = False
    for node in nodes[nodes.index(fbank_split) + 1:]:
        if not any(value in reachable for value in node.input):
            continue
        fbank_nodes.append(node)
        reachable.update(value for value in node.output if value)
        if node.op_type == "Log":
            reached_log = True
            break
    if not reached_log:
        raise RuntimeError("Could not identify the complete DFSMN square-to-log fbank chain")
    if not any("input_power_scale" in node.input for node in fbank_nodes):
        raise RuntimeError("The DFSMN fbank chain does not restore int16-domain power")
    unexpected_ops = {
        node.op_type for node in fbank_nodes
        if node.op_type not in {"Split", "Mul", "Add", "MatMul", "Clip", "Log"}
    }
    if unexpected_ops:
        raise RuntimeError(f"Unexpected operators in the DFSMN fbank FP32 island: {unexpected_ops}")

    output_cast = producers.get(graph.output[0].name)
    cast_to = None if output_cast is None else next(
        (
            helper.get_attribute_value(attribute)
            for attribute in output_cast.attribute
            if attribute.name == "to"
        ),
        None,
    )
    output_clip = producers.get(output_cast.input[0]) if output_cast is not None else None
    output_scale = (
        producers.get(output_clip.input[0])
        if output_clip is not None and output_clip.input
        else None
    )
    if (
        output_cast is None
        or output_cast.op_type != "Cast"
        or cast_to != TensorProto.INT16
        or output_clip is None
        or output_clip.op_type != "Clip"
        or output_scale is None
        or output_scale.op_type != "Mul"
        or "output_scale" not in output_scale.input
    ):
        raise RuntimeError(
            "Unexpected DFSMN output topology; expected Mul(output_scale) -> Clip -> Cast(INT16)"
        )

    protected = [
        node.name
        for node in [*fbank_nodes, output_scale, output_clip]
        if node.name
    ]
    if len(protected) != len(fbank_nodes) + 2:
        raise RuntimeError("DFSMN FP16 safety nodes must have stable non-empty names")
    print(
        "  Scale-safe frontend verified; keeping complete FP32 islands: "
        f"fbank={len(fbank_nodes)} nodes, output=2 nodes"
    )
    return protected

MODEL_PLANS = {
    "DFSMN": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=0,
        hidden_size=0,
        opt_level=1,
        only_onnxruntime=True,
        first_slim_no_shape_infer="auto",
        second_slim_no_shape_infer="auto",
        fp16_symbolic_shape_infer="auto",
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
        f16_op_block_list=["Range", "ReduceL2", "ReduceMean", "ReduceSum", "Sqrt"],
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    # Preserve true FP32 constants used by blocked nodes. Converting the 2**30
    # fbank power restoration or 32767 clip bound to FP16 is invalid.
    f16_force_initializers=False,
)

if __name__ == "__main__":
    run_optimizer(CONFIG)

