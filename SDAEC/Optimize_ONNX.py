"""Optimize the exported SDAEC ONNX model."""

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

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "SDAEC_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "SDAEC_Optimized")

ENABLE_FP16 = False   # Mixed FP16/FP32 CUDA graph; see FP16_CUDA_Test/README.md.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Select and validate the FP32 waveform-output safety island.

    The exporter intentionally keeps PCM scales outside the DFT/COLA tables. This
    makes both tables accurately representable in FP16, but the final multiplication
    by 32767 and its clipping limits must remain FP32: 32767 rounds to 32768 in FP16,
    which can wrap to -32768 when cast directly to INT16.
    """
    model = onnx.load(src_path, load_external_data=False)
    graph = model.graph
    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in graph.initializer
    }

    stft_kernel = initializers.get("custom_stft.stft_kernel")
    inv_win_sum = initializers.get("iccrn.custom_istft.inv_win_sum")
    if stft_kernel is None or inv_win_sum is None:
        raise RuntimeError("Could not find the SDAEC STFT/ISTFT table initializers")
    stft_max = float(np.max(np.abs(stft_kernel)))
    inv_win_sum_max = float(np.max(np.abs(inv_win_sum)))
    if not np.isfinite(stft_max) or not 0.5 <= stft_max <= 1.1:
        raise RuntimeError(
            "Unsafe SDAEC FP16 source: the PCM input scale appears folded into the "
            f"STFT kernel (max abs {stft_max:.8g}); re-export with input_scale=1.0"
        )
    if not np.isfinite(inv_win_sum_max) or inv_win_sum_max > 256.0:
        raise RuntimeError(
            "Unsafe SDAEC FP16 source: the PCM output scale appears folded into COLA "
            f"normalization (max abs {inv_win_sum_max:.8g}); re-export with output_scale=1.0"
        )

    producers = {
        output: node
        for node in graph.node
        for output in node.output
        if output
    }
    output_cast = producers.get(graph.output[0].name)
    if output_cast is None or output_cast.op_type != "Cast":
        raise RuntimeError("Could not identify the SDAEC output INT16 Cast")
    cast_to = next(
        (
            helper.get_attribute_value(attribute)
            for attribute in output_cast.attribute
            if attribute.name == "to"
        ),
        None,
    )
    clip = producers.get(output_cast.input[0])
    scale = producers.get(clip.input[0]) if clip is not None and clip.input else None
    if (
        cast_to != TensorProto.INT16
        or clip is None
        or clip.op_type != "Clip"
        or scale is None
        or scale.op_type != "Mul"
    ):
        raise RuntimeError(
            "Unexpected SDAEC output topology; expected FP32 Mul -> Clip -> Cast(INT16)"
        )
    blocked = [scale.name, clip.name]
    if not all(blocked):
        raise RuntimeError("SDAEC FP16 safety nodes must have stable non-empty names")
    print(
        "  Scale-safe tables verified "
        f"(STFT max={stft_max:.6g}, COLA reciprocal max={inv_win_sum_max:.6g}); "
        f"keeping output nodes in FP32: {blocked}"
    )
    return blocked


MODEL_PLANS = {
    "SDAEC": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=0,
        hidden_size=0,
        opt_level=1,
        only_onnxruntime=True,
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
        f16_op_block_list=["Range", "ReduceL2", "ReduceMean", "ReduceSum", "Sqrt"],
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    # Preserve real FP32 constants for blocked nodes. Otherwise the converter can
    # quantize a safety-island initializer first and cast the rounded value back.
    f16_force_initializers=False,
)

if __name__ == "__main__":
    run_optimizer(CONFIG)
