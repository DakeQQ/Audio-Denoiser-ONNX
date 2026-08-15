"""Optimize the exported GAP-URGENet ONNX pipeline."""

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

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "GAP_URGENet_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "GAP_URGENet_Optimized")

# User config
QUANT_METHOD = "Q8"                     # "Q4" | "Q8" | "F16" | "F32"
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 64                         # Power of two in [16, 256].
ACCURACY_LEVEL = 4                      # MatMulNBits: 1=FP32, 2=FP16, 3=BF16, 4=INT8, 0=default.
QUANT_SYMMETRIC = False                 # Asymmetric generally preserves more quality.
QUANT_FORMAT = "QOperator"              # AFFINE_REFINE_V2 supports QOperator only.

ENABLE_FP16 = False                     # Optional FP16 conversion after the selected method.
UPGRADE_OPSET = 0

MODEL_PLANS = {
    "GAP_URGENet": Plan(
        method=QUANT_METHOD,
        algo=WEIGHT_ONLY_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        symmetric=QUANT_SYMMETRIC,
        quant_format=QUANT_FORMAT,
        fp16=ENABLE_FP16,
        num_heads=0,
        hidden_size=0,
        opt_level=1,
        transformer=False,
        external=True,
        first_slim_no_shape_infer="auto",
        second_slim_no_shape_infer="auto",
        fp16_symbolic_shape_infer="auto",
    ),
    "GAP_URGENet_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
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