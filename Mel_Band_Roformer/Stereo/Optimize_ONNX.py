"""Optimize the exported MelBandRoformer Stereo ONNX model."""

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

ENABLE_DYNAMIC_Q8 = True     # For CPU, Works well
ENABLE_FP16 = False          # Works, but affect quality.
UPGRADE_OPSET = 0

MODEL_METHOD = "DYNAMIC_Q8" if ENABLE_DYNAMIC_Q8 else ("F16" if ENABLE_FP16 else "F32")


MODEL_PLANS = {
    "MelBandRoformer": Plan(
        method=MODEL_METHOD,
        fp16=ENABLE_FP16,
        num_heads=8,
        hidden_size=1536,
        opt_level=2,
        use_gpu=False,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
        dynamic_weight_type="QInt8",
        per_channel=False,
        reduce_range=False,
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
