"""Optimize the exported MelBandRoformer Mono ONNX model."""

from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "MelBandRoformer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "MelBandRoformer_Optimized")

# User config
QUANT_METHOD = "Q8"                    # "Q4" | "Q8" | "DYNAMIC_Q8" | "F16" | "F32"
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"  # "AFFINE_REFINE_V2" | "DEFAULT" | "RTN" | "HQQ" | "k_quant"
BLOCK_SIZE = 32                         # Power of two in [16, 256].
ACCURACY_LEVEL = 4                      # MatMulNBits: 1=FP32, 2=FP16, 3=BF16, 4=INT8, 0=default.
QUANT_SYMMETRIC = False                 # Asymmetric generally preserves more quality.
QUANT_FORMAT = "QOperator"             # "QOperator"; "QDQ" is supported for DEFAULT Q4 only.

DYNAMIC_WEIGHT_TYPE = "QInt8"          # "QInt8" | "QUInt8"
DYNAMIC_PER_CHANNEL = False
DYNAMIC_REDUCE_RANGE = False

ENABLE_FP16 = False                     # Optional FP16 conversion after the selected method.
UPGRADE_OPSET = 0


MODEL_PLANS = {
    "MelBandRoformer": Plan(
        method=QUANT_METHOD,
        algo=WEIGHT_ONLY_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        symmetric=QUANT_SYMMETRIC,
        quant_format=QUANT_FORMAT,
        fp16=ENABLE_FP16,
        num_heads=8,
        hidden_size=1536,
        opt_level=2,
        use_gpu=False,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
        dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
        per_channel=DYNAMIC_PER_CHANNEL,
        reduce_range=DYNAMIC_REDUCE_RANGE,
    ),
}


CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    fuse_consecutive_reshapes=True,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
