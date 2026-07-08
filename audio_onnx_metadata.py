import gc
import shutil
from pathlib import Path

import onnx


REQUIRED_AUDIO_METADATA_KEYS = (
    "audio_metadata_version",
    "producer",
    "model_name",
    "task",
    "model_family",
    "dynamic_axes",
    "opset",
    "input_audio_dtype",
    "output_audio_dtype",
    "in_sample_rate",
    "out_sample_rate",
    "model_sample_rate",
    "input_audio_length",
    "input_to_output_scale",
    "max_dynamic_audio_seconds",
    "normalize_audio_default",
    "normalize_target_rms",
)


class METADATA_CARRIER:
    def __call__(self, marker):
        return marker

    def forward(self, marker):
        return marker


def metadata_path_for_model(model_path):
    path = Path(model_path)
    return path.with_name(path.stem + "_Metadata.onnx")


def build_model_metadata(*sections):
    metadata = {}
    for section in sections:
        if not section:
            continue
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "1" if value else "0"
            elif isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            else:
                value = str(value)
            metadata[str(key)] = value
    return metadata


def read_onnx_metadata(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    del model
    gc.collect()
    return metadata


def write_onnx_metadata(model_path, metadata):
    if not metadata:
        return
    model = onnx.load(str(model_path), load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in build_model_metadata(metadata).items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, str(model_path))
    del model
    gc.collect()


def export_metadata_carrier(metadata_path, metadata, opset):
    import torch

    class _MetadataCarrier(torch.nn.Module):
        def forward(self, marker):
            return marker

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    marker = torch.zeros((1,), dtype=torch.int64)
    torch.onnx.export(
        _MetadataCarrier(),
        marker,
        str(metadata_path),
        input_names=["marker"],
        output_names=["marker_out"],
        dynamic_axes=None,
        do_constant_folding=True,
        opset_version=int(opset),
        dynamo=False,
    )
    write_onnx_metadata(metadata_path, metadata)


def stamp_export_metadata(main_model_path, metadata, opset):
    metadata = build_model_metadata(metadata)
    metadata_model_path = metadata_path_for_model(main_model_path)
    export_metadata_carrier(metadata_model_path, metadata, opset)
    write_onnx_metadata(main_model_path, metadata)
    return metadata_model_path


def build_audio_metadata_from_globals(
    namespace,
    *,
    producer,
    model_name,
    task,
    model_family,
    max_dynamic_audio_seconds,
    normalize_audio_default,
    normalize_target_rms=4096.0,
    batch_fold_inference_default=None,
    input_channels=1,
    output_channels=1,
    num_audio_inputs=1,
    feature_kind=None,
    center_pad=None,
    pad_mode=None,
    extra=None,
):
    def pick(*names, default=None):
        for name in names:
            if name in namespace:
                return namespace[name]
        return default

    in_sample_rate = pick("IN_SAMPLE_RATE", "ORIGINAL_SAMPLE_RATE")
    out_sample_rate = pick("OUT_SAMPLE_RATE", "SUPER_SAMPLE_RATE", default=in_sample_rate)
    model_sample_rate = pick("MODEL_SAMPLE_RATE", "SUPER_SAMPLE_RATE", default=out_sample_rate)
    input_audio_length = pick("INPUT_AUDIO_LENGTH")
    export_audio_length = pick("EXPORT_AUDIO_LENGTH", default=input_audio_length)
    dynamic_axes = bool(pick("DYNAMIC_AXES", default=False))
    model_audio_length = pick("MODEL_AUDIO_LENGTH", default=None)
    if model_audio_length is None and None not in (input_audio_length, in_sample_rate, model_sample_rate):
        model_audio_length = input_audio_length if dynamic_axes else int(round(input_audio_length * model_sample_rate / in_sample_rate))
    output_audio_length = pick("OUTPUT_AUDIO_LENGTH", default=None)
    if output_audio_length is None and None not in (input_audio_length, in_sample_rate, out_sample_rate):
        output_audio_length = input_audio_length if dynamic_axes else int(round(input_audio_length * out_sample_rate / in_sample_rate))

    fold_window_length = pick("FOLD_WINDOW_LENGTH", default=None)
    fold_input_length = None
    if fold_window_length and None not in (in_sample_rate, model_sample_rate):
        fold_input_length = max(1, int(round(fold_window_length * in_sample_rate / model_sample_rate)))
    use_batch_fold = pick("USE_BATCH_FOLD", default=None)
    if batch_fold_inference_default is None and use_batch_fold is not None:
        batch_fold_inference_default = bool(use_batch_fold)

    metadata = build_model_metadata(
        {
            "audio_metadata_version": 1,
            "producer": producer,
            "model_name": model_name,
            "task": task,
            "model_family": model_family,
            "dynamic_axes": dynamic_axes,
            "opset": pick("OPSET"),
            "input_audio_dtype": pick("IN_AUDIO_DTYPE", "INPUT_AUDIO_DTYPE"),
            "output_audio_dtype": pick("OUT_AUDIO_DTYPE", "OUTPUT_AUDIO_DTYPE"),
            "in_sample_rate": in_sample_rate,
            "out_sample_rate": out_sample_rate,
            "model_sample_rate": model_sample_rate,
            "input_audio_length": input_audio_length,
            "export_audio_length": export_audio_length,
            "model_audio_length": model_audio_length,
            "output_audio_length": output_audio_length,
            "input_to_output_scale": pick(
                "INPUT_TO_OUTPUT_SCALE",
                default=(None if None in (in_sample_rate, out_sample_rate) else float(out_sample_rate / in_sample_rate)),
            ),
            "batch_window_seconds": pick("BATCH_WINDOW_SECONDS", default=None),
            "use_batch_fold": use_batch_fold,
            "batch_fold_inference_default": batch_fold_inference_default,
            "fold_window_length": fold_window_length,
            "fold_input_length": fold_input_length,
            "max_dynamic_audio_seconds": max_dynamic_audio_seconds,
            "normalize_audio_default": normalize_audio_default,
            "normalize_target_rms": normalize_target_rms,
            "window_type": pick("WINDOW_TYPE", default=None),
            "nfft": pick("NFFT", "NFFT_STFT", "NFFT_A2", default=None),
            "window_length": pick("WINDOW_LENGTH", "WINDOW_LENGTH_A", default=None),
            "hop_length": pick("HOP_LENGTH", "HOP_LENGTH_A", default=None),
            "max_signal_length": pick("MAX_SIGNAL_LENGTH", default=None),
            "center_pad": center_pad,
            "pad_mode": pick("PAD_MODE", "STFT_PAD_MODE", default=pad_mode),
            "feature_kind": feature_kind,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "num_audio_inputs": num_audio_inputs,
        },
        extra,
    )
    return metadata


def read_source_metadata(main_model_path):
    main_model_path = Path(main_model_path)
    metadata = {}
    if main_model_path.exists():
        metadata = read_onnx_metadata(main_model_path)
    metadata_model_path = metadata_path_for_model(main_model_path)
    if not metadata and metadata_model_path.exists():
        metadata = read_onnx_metadata(metadata_model_path)
    return metadata


def copy_metadata_model(src_main_model_path, dst_main_model_path):
    src_metadata_path = metadata_path_for_model(src_main_model_path)
    dst_metadata_path = metadata_path_for_model(dst_main_model_path)
    if not src_metadata_path.exists():
        return False
    dst_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_metadata_path, dst_metadata_path)
    return True


def preserve_optimized_metadata(src_main_model_path, dst_main_model_path, metadata):
    if metadata:
        write_onnx_metadata(dst_main_model_path, metadata)
        copy_metadata_model(src_main_model_path, dst_main_model_path)
    else:
        print(
            f"WARNING: No ONNX metadata found in {src_main_model_path} or "
            f"{metadata_path_for_model(src_main_model_path)}. Re-export the model to get metadata."
        )


def _missing_key_message(key):
    return (
        f"Required metadata key {key} is missing. "
        "Re-export with the matching Export_*.py and rerun Optimize_ONNX.py."
    )


class MetadataReader:
    def __init__(self, metadata):
        self.metadata = dict(metadata or {})

    def string(self, key, default=None, required=False):
        value = self.metadata.get(key)
        if value is None or value == "":
            if required:
                raise KeyError(_missing_key_message(key))
            return default
        return value

    def required_int(self, key):
        return int(self.string(key, required=True))

    def optional_int(self, key, default=None):
        value = self.string(key, default=None)
        return default if value is None else int(value)

    def required_float(self, key):
        return float(self.string(key, required=True))

    def optional_float(self, key, default=None):
        value = self.string(key, default=None)
        return default if value is None else float(value)

    def required_bool(self, key):
        return _parse_bool(self.string(key, required=True), key)

    def optional_bool(self, key, default=None):
        value = self.string(key, default=None)
        return default if value is None else _parse_bool(value, key)


def _parse_bool(value, key):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Metadata key {key} must be a boolean encoded as 1/0, got {value!r}.")


def load_runtime_metadata(model_path, make_session, required_keys=REQUIRED_AUDIO_METADATA_KEYS):
    model_path = Path(model_path)
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Required metadata model is missing: {metadata_path}. "
            "Re-export with the matching Export_*.py and rerun Optimize_ONNX.py."
        )
    metadata_session = make_session(str(metadata_path))
    metadata = metadata_session.get_modelmeta().custom_metadata_map or {}
    reader = MetadataReader(metadata)
    for key in required_keys:
        reader.string(key, required=True)
    return reader


def _static_int_dim(value):
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_audio_metadata(reader, session):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs:
        return
    input_shape = inputs[0].shape
    static_input_length = _static_int_dim(input_shape[-1]) if input_shape else None
    metadata_input_length = reader.optional_int("export_audio_length", None)
    if metadata_input_length is None:
        metadata_input_length = reader.optional_int("input_audio_length", None)
    if static_input_length is not None and metadata_input_length is not None and static_input_length != metadata_input_length:
        raise ValueError(
            f"ONNX input length {static_input_length} does not match metadata input length {metadata_input_length}. "
            "Re-export with the matching Export_*.py and rerun Optimize_ONNX.py."
        )

    input_channels = reader.optional_int("input_channels", None)
    if input_channels is not None and len(input_shape) >= 3:
        static_input_channels = _static_int_dim(input_shape[-2])
        if static_input_channels is not None and static_input_channels != input_channels:
            raise ValueError(
                f"ONNX input channels {static_input_channels} do not match metadata input_channels={input_channels}."
            )

    output_channels = reader.optional_int("output_channels", None)
    if output_channels is not None and outputs:
        output_shape = outputs[0].shape
        if len(output_shape) >= 3:
            static_output_channels = _static_int_dim(output_shape[-2])
            if static_output_channels is not None and static_output_channels != output_channels:
                raise ValueError(
                    f"ONNX output channels {static_output_channels} do not match metadata output_channels={output_channels}."
                )

    num_audio_inputs = reader.optional_int("num_audio_inputs", None)
    if num_audio_inputs is not None and len(inputs) < num_audio_inputs:
        raise ValueError(f"ONNX model has {len(inputs)} inputs, metadata num_audio_inputs={num_audio_inputs}.")


def runtime_config_from_metadata(reader):
    in_sample_rate = reader.required_int("in_sample_rate")
    out_sample_rate = reader.required_int("out_sample_rate")
    model_sample_rate = reader.required_int("model_sample_rate")
    fold_window_length = reader.optional_int("fold_window_length", 0)
    config = {
        "IN_SAMPLE_RATE": in_sample_rate,
        "OUT_SAMPLE_RATE": out_sample_rate,
        "MODEL_SAMPLE_RATE": model_sample_rate,
        "INPUT_TO_OUTPUT_SCALE": reader.required_float("input_to_output_scale"),
        "BATCH_WINDOW_SECONDS": reader.optional_float("batch_window_seconds", 0.0),
        "HOP_LENGTH": reader.optional_int("hop_length", 0),
        "FOLD_WINDOW_LENGTH": fold_window_length,
        "FOLD_INPUT_LENGTH": reader.optional_int(
            "fold_input_length",
            max(1, int(round(fold_window_length * in_sample_rate / model_sample_rate))) if fold_window_length else 0,
        ),
        "BATCH_FOLD_INFERENCE": reader.optional_bool("batch_fold_inference_default", False),
        "MAX_DYNAMIC_AUDIO_SECONDS": reader.required_int("max_dynamic_audio_seconds"),
        "NORMALIZE_AUDIO": reader.required_bool("normalize_audio_default"),
        "NORMALIZE_TARGET_RMS": reader.required_float("normalize_target_rms"),
        "INPUT_CHANNELS": reader.optional_int("input_channels", 1),
        "OUTPUT_CHANNELS": reader.optional_int("output_channels", 1),
        "N_CHANNELS": reader.optional_int("input_channels", 1),
        "NUM_AUDIO_INPUTS": reader.optional_int("num_audio_inputs", 1),
        "PAD_HEAD": reader.optional_int("pad_head", 0),
        "ENC_STRIDE": reader.optional_int("enc_stride", 0),
        "OUTPUT_SOURCES": reader.optional_int("output_sources", 1),
        "ORIGINAL_SAMPLE_RATE": reader.optional_int("original_sample_rate", in_sample_rate),
        "SUPER_SAMPLE_RATE": reader.optional_int("super_sample_rate", out_sample_rate),
        "SCALE_FACTOR": reader.optional_float("scale_factor", float(out_sample_rate / in_sample_rate)),
    }
    return config