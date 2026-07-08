import argparse
from pathlib import Path

import onnxruntime

from audio_onnx_metadata import REQUIRED_AUDIO_METADATA_KEYS, metadata_path_for_model


def main():
    parser = argparse.ArgumentParser(description="Print and validate Audio-Denoiser ONNX metadata keys.")
    parser.add_argument("model", type=Path, help="Main ONNX model path or metadata carrier path.")
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    metadata_path = model_path if model_path.stem.endswith("_Metadata") else metadata_path_for_model(model_path)
    if not metadata_path.exists():
        metadata_path = model_path

    session = onnxruntime.InferenceSession(str(metadata_path), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map or {}
    missing = [key for key in REQUIRED_AUDIO_METADATA_KEYS if key not in metadata]
    print("\n".join(sorted(metadata)))
    if missing:
        raise SystemExit(f"Missing required metadata keys: {', '.join(missing)}")


if __name__ == "__main__":
    main()