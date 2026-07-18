"""Optimize the exported Deep Echo AEC ONNX model."""

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

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "Deep_Echo_AEC_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Deep_Echo_AEC_Optimized")

ENABLE_FP16 = False   # CUDA-safe FP16 graph; the output PCM boundary remains FP32.
UPGRADE_OPSET = 0


def _cast_target(node: onnx.NodeProto) -> int | None:
    return next(
        (
            helper.get_attribute_value(attribute)
            for attribute in node.attribute
            if attribute.name == "to"
        ),
        None,
    )


def _constant_scalars(graph: onnx.GraphProto) -> dict[str, float]:
    values: dict[str, float] = {}
    for tensor in graph.initializer:
        array = np.asarray(numpy_helper.to_array(tensor))
        if array.size == 1:
            values[tensor.name] = float(array.reshape(()))
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attribute in node.attribute:
            value = helper.get_attribute_value(attribute)
            if isinstance(value, onnx.TensorProto):
                array = np.asarray(numpy_helper.to_array(value))
                if array.size == 1:
                    values[node.output[0]] = float(array.reshape(()))
                    break
            if isinstance(value, (float, int)):
                values[node.output[0]] = float(value)
                break
    return values


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Validate the source contract and retain only the PCM output chain in FP32.

    PCM is normalized before centering, so its reduction is safe in FP16. The two
    cepstral norms quarter-scale both centered values and epsilon, keeping their
    formerly overflowing variance sums finite without F32 -> F16 -> F32 cast
    sandwiches. The final PCM scale and clip stay FP32 because 32767 rounds to
    32768 in FP16 before the INT16 cast.
    """
    model = onnx.load(src_path, load_external_data=False)
    graph = model.graph
    nodes = list(graph.node)
    producers = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in graph.initializer
    }
    scalar_values = _constant_scalars(graph)

    stft_kernel = initializers.get("custom_stft.stft_kernel")
    if stft_kernel is None:
        raise RuntimeError("Could not find the Deep Echo STFT kernel")
    stft_max = float(np.max(np.abs(stft_kernel)))
    if not np.isfinite(stft_max) or not 0.9 <= stft_max <= 1.1:
        raise RuntimeError(
            "Unsafe Deep Echo FP16 source: expected an unscaled STFT table near "
            f"[-1, 1], found max abs {stft_max:.8g}. Re-export the model."
        )

    input_mean = next((node for node in nodes if node.name == "/ReduceMean"), None)
    input_center = next((node for node in nodes if node.name == "/Sub"), None)
    input_scale = (
        producers.get(input_mean.input[0])
        if input_mean is not None and input_mean.input
        else None
    )
    input_cast = (
        producers.get(input_scale.input[0])
        if input_scale is not None and input_scale.input
        else None
    )
    input_scale_value = next(
        (scalar_values[value] for value in input_scale.input if value in scalar_values),
        None,
    ) if input_scale is not None else None
    if (
        input_cast is None
        or input_cast.op_type != "Cast"
        or _cast_target(input_cast) != TensorProto.FLOAT
        or input_scale is None
        or input_scale.op_type != "Mul"
        or input_scale_value is None
        or not np.isclose(input_scale_value, 1.0 / 32768.0)
        or input_center is None
        or input_center.op_type != "Sub"
        or input_scale.output[0] not in input_center.input
    ):
        raise RuntimeError(
            "Unexpected Deep Echo input topology; expected Cast -> 1/32768 scale "
            "-> ReduceMean/Sub before STFT"
        )

    ceps_prefixes = (
        "/iccrn/cfb_e1/ceps_unit/LN/",
        "/iccrn/cfb_d1/ceps_unit/LN/",
    )
    for prefix in ceps_prefixes:
        prefixed = [node for node in nodes if node.name.startswith(prefix)]
        reductions = [node for node in prefixed if node.op_type == "ReduceSum"]
        scale_nodes = [
            node
            for node in prefixed
            if node.op_type == "Mul"
            and any(
                value in scalar_values and np.isclose(scalar_values[value], 0.25)
                for value in node.input
            )
        ]
        if len(reductions) != 1 or len(scale_nodes) != 1:
            raise RuntimeError(
                f"Could not verify quarter-scaled cepstral variance chain: {prefix}"
            )

    output_cast = producers.get(graph.output[0].name)
    clip = producers.get(output_cast.input[0]) if output_cast is not None else None
    scale = producers.get(clip.input[0]) if clip is not None else None
    if (
        output_cast is None
        or output_cast.op_type != "Cast"
        or _cast_target(output_cast) != TensorProto.INT16
        or clip is None
        or clip.op_type != "Clip"
        or scale is None
        or scale.op_type != "Mul"
    ):
        raise RuntimeError(
            "Unexpected Deep Echo output topology; expected Mul -> Clip -> Cast(INT16)"
        )

    blocked = [scale.name, clip.name]
    if not all(blocked):
        raise RuntimeError("Deep Echo FP16 output safety nodes require stable names")
    print(
        "  Verified normalized input, quarter-scaled cepstral reductions, and "
        f"STFT max={stft_max:.6g}; keeping output nodes in FP32: {blocked}"
    )
    return blocked


MODEL_PLANS = {
    "Deep_Echo_AEC": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=0,
        hidden_size=0,
        opt_level=1,
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    # Constants owned by the FP32 PCM scale/clip island must remain true FP32.
    f16_force_initializers=False,
)

if __name__ == "__main__":
    run_optimizer(CONFIG)
