"""Optimize the exported MossFormer2 Super Resolution ONNX model."""

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


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "MossFormer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "MossFormer_Optimized")

ENABLE_FP16 = False   # F16 silent output, overflow.
UPGRADE_OPSET = 0  # Keep the exported opset; the old opset-18 upgrade was commented out.


MODEL_PLANS = {
    "MossFormer2_SR": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=4,
        hidden_size=128,
        opt_level=1 if ENABLE_FP16 else 2,
        only_onnxruntime=True,
        use_gpu=False,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
        fp16_symbolic_shape_infer=False,
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
