"""Optimize the exported H-GTCRN ONNX model."""

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

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "H_GTCRN_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "H_GTCRN_Optimized")

ENABLE_FP16 = False  # F16 got silent outputs, overflow.
UPGRADE_OPSET = 0

MODEL_PLANS = {
    "H_GTCRN": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=0,
        hidden_size=0,
        opt_level=1,
        first_slim_no_shape_infer="auto",
        second_slim_no_shape_infer="auto",
        fp16_symbolic_shape_infer="auto",
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

