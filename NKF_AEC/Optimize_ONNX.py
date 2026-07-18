"""Optimize the exported NKF AEC ONNX model."""

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

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "NKF_AEC_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "NKF_AEC_Optimized")

ENABLE_FP16 = False     # F16 works and perform well.
UPGRADE_OPSET = 0

# Keep raw-PCM centering in FP32. CUDA's FP16 ReduceMean accumulates the 32,000
# int16-scale samples before dividing, so even a small DC offset can overflow
# to Inf and turn the final int16 waveform into silence. The centered waveform
# is cast to FP16 immediately before the STFT; all later model compute remains
# FP16.
FP16_WAVEFORM_NODE_BLOCK_LIST = ["/ReduceMean", "/Sub"]

MODEL_PLANS = {
    "NKF_AEC": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=0,
        hidden_size=0,
        opt_level=1,
        only_onnxruntime=True,
        f16_node_block_list=FP16_WAVEFORM_NODE_BLOCK_LIST,
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
