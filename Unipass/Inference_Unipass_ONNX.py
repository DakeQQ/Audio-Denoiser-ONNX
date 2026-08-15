#!/usr/bin/env python3
"""Run the exported multi-stage UniPASE ONNX speech-enhancement pipeline."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "Unipass_Optimized"
DEFAULT_AUDIO = TARGET_ROOT / "Test_Examples" / "denoise" / "speech_with_noise1.wav"
ENCODER_NAME = "UniPASE_Encoder.onnx"
ENHANCER_NAME = "UniPASE_Enhancer.onnx"
POSTNET_NAME = "UniPASE_PostNet_48k.onnx"
METADATA_NAME = "UniPASE_Metadata.onnx"

# User settings.
model_dir = DEFAULT_MODEL_DIR
test_noisy_audio = DEFAULT_AUDIO
save_denoised_audio  = Path(__file__).resolve().parent / "denoised.wav"
USE_POSTNET = True
ORT_Accelerate_Providers = []   # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
ORT_LOG = False                 # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16 = False                # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
CPU_DISABLE_MATMUL_ADD_FUSION = True
CPU_DISABLE_NCHWC = True
CPU_EXTRA_DISABLED_OPTIMIZERS = [
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS = 0                 # Number of ONNX Runtime/OpenVINO worker threads. Set 0 for auto.
DEVICE_ID = 0                   # The GPU id, default to 0.
NORMALIZE_AUDIO = False         # Set True to RMS-normalize input audio before inference.
NORMALIZE_TARGET_RMS = 4096.0   # Target RMS when NORMALIZE_AUDIO is True.
DYNAMIC_WINDOW_SAMPLES = 32000  # Set 0 to process the full dynamic input in one call.
INV_INT16 = float(1.0 / 32768.0)

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import numpy_dtype_from_onnx_meta, resolve_onnx_shape


def _resolve_model_dir(default_model_dir: Path) -> Path:
    if len(sys.argv) <= 1:
        return default_model_dir
    candidate = Path(sys.argv[1]).expanduser()
    return candidate if candidate.is_dir() else candidate.parent


def _dynamic_metadata_int(metadata: dict[str, str], key: str) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Dynamic UniPASE metadata requires a positive integer {key!r}.") from error
    if value <= 0:
        raise ValueError(f"Dynamic UniPASE metadata requires a positive integer {key!r}, got {value}.")
    return value


# ONNX Runtime settings
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            "device_type": "CPU",  # CPU, NPU, GPU, GPU.0, GPU.1
            "precision": "ACCURACY",  # FP32, FP16, ACCURACY
            "num_of_threads": MAX_THREADS if MAX_THREADS != 0 else 8,
            "num_streams": 1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }
    ]
    device_type = "cpu"
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            "device_id": DEVICE_ID,
            "gpu_mem_limit": 24 * 1024 * 1024 * 1024,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "sdpa_kernel": "2",
            "use_tf32": "1",
            "fuse_conv_bias": "0",
            "cudnn_conv_use_max_workspace": "1",
            "cudnn_conv1d_pad_to_nc1d": "0",
            "tunable_op_enable": "0",
            "tunable_op_tuning_enable": "0",
            "tunable_op_max_tuning_duration_ms": 10,
            "do_copy_in_default_stream": "0",
            "enable_cuda_graph": "0",
            "prefer_nhwc": "0",
            "enable_skip_layer_norm_strict_mode": "1",
            "use_ep_level_unified_stream": "0",
        }
    ]
    device_type = "cuda"
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            "device_id": DEVICE_ID,
            "performance_preference": "high_performance",
            "device_filter": "gpu",
            "disable_metacommands": "false",
            "enable_graph_capture": "false",
            "enable_graph_serialization": "false",
        }
    ]
    device_type = "dml"
else:
    device_type = "cpu"
    provider_options = None


def normalise_audio(audio: np.ndarray, input_dtype: np.dtype, target_rms: float | None = None) -> np.ndarray:
    if target_rms is None:
        target_rms = NORMALIZE_TARGET_RMS
    if NORMALIZE_AUDIO:
        audio = audio.astype(np.float32)
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0.0:
            audio *= target_rms / (rms + 1e-7)
        if np.issubdtype(input_dtype, np.integer):
            limits = np.iinfo(input_dtype)
            np.clip(audio, limits.min, limits.max, out=audio)
        return audio.astype(input_dtype, copy=False)

    if input_dtype != np.int16:
        audio = audio * INV_INT16
    return audio.astype(input_dtype, copy=False)


def _pad_final_static_window(audio: np.ndarray, padded_length: int) -> np.ndarray:
    """Extend a static model's final window without creating false PLC packets."""

    if padded_length < audio.shape[-1]:
        raise ValueError("Static padded length cannot be shorter than the input audio.")
    padding = padded_length - audio.shape[-1]
    if padding == 0:
        return audio
    mode = "reflect" if audio.shape[-1] > 1 else "edge"
    return np.pad(audio, ((0, 0), (0, 0), (0, padding)), mode=mode)


def _build_run_options(silent: bool) -> onnxruntime.RunOptions:
    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level = 0 if not silent else 4
    run_options.log_verbosity_level = 4
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return run_options


def _build_session_opts_ort() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    configs = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_gelu_approximation": "1",
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if ORT_FP16 else ""
        ),
    }
    for key, value in configs.items():
        options.add_session_config_entry(key, value)
    return options


def _ortvalue_from_meta(meta, runtime_shape):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(
            resolve_onnx_shape(meta, runtime_shape),
            dtype=numpy_dtype_from_onnx_meta(meta),
        ),
        device_type,
        DEVICE_ID,
    )


def _update_ortvalue(ort_value, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    if hasattr(ort_value, "update_inplace"):
        ort_value.update_inplace(array)
    else:
        np.copyto(ort_value.numpy(), array)


def _run_iobinding(session: onnxruntime.InferenceSession, binding: onnxruntime.IOBinding) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


def _make_session(path: str) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(path, **_packed)


session_opts_ort = _build_session_opts_ort()
run_options = _build_run_options(silent=not ORT_LOG)
_CPU_EP_ONLY = not ORT_Accelerate_Providers or set(ORT_Accelerate_Providers) == {"CPUExecutionProvider"}
disabled_opts = []
if ORT_FP16:
    disabled_opts.extend(["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"])
if _CPU_EP_ONLY and CPU_DISABLE_MATMUL_ADD_FUSION:
    disabled_opts.append("MatMulAddFusion")
if _CPU_EP_ONLY and CPU_DISABLE_NCHWC:
    disabled_opts.append("NchwcTransformer")
if _CPU_EP_ONLY:
    disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
disabled_opts = disabled_opts or None
_packed = {
    "sess_options": session_opts_ort,
    "providers": ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    "provider_options": provider_options,
    "disabled_optimizers": disabled_opts,
}


def _load_pipeline_metadata(metadata_path: Path) -> dict[str, str]:
    session = _make_session(str(metadata_path))
    return dict(session.get_modelmeta().custom_metadata_map)


def main() -> None:
    selected_model_dir = _resolve_model_dir(model_dir)
    encoder_path = selected_model_dir / ENCODER_NAME
    enhancer_path = selected_model_dir / ENHANCER_NAME
    postnet_path = selected_model_dir / POSTNET_NAME
    metadata = _load_pipeline_metadata(selected_model_dir / METADATA_NAME)
    encoder_session = _make_session(str(encoder_path))
    enhancer_session = _make_session(str(enhancer_path))
    postnet_session = (
        _make_session(str(postnet_path))
        if USE_POSTNET and postnet_path.is_file() and "postnet_out_sample_rate" in metadata
        else None
    )

    input_rate = int(metadata["in_sample_rate"])
    output_rate = int(metadata["out_sample_rate"])
    encoder_inputs = encoder_session.get_inputs()
    encoder_outputs = encoder_session.get_outputs()
    enhancer_inputs = enhancer_session.get_inputs()
    enhancer_outputs = enhancer_session.get_outputs()
    encoder_input = encoder_inputs[0]
    enhancer_output = enhancer_outputs[0]
    input_dtype = numpy_dtype_from_onnx_meta(encoder_input)
    output_dtype = numpy_dtype_from_onnx_meta(enhancer_output)

    print(f"\nUsable Providers: {encoder_session.get_providers()}")
    print(f"\nTest Input Audio: {test_noisy_audio}")
    audio_segment = AudioSegment.from_file(str(test_noisy_audio)).set_frame_rate(input_rate)
    input_shape = encoder_input.shape
    input_channels = input_shape[1] if isinstance(input_shape[1], int) else audio_segment.channels
    audio = np.asarray(audio_segment.set_channels(input_channels).get_array_of_samples(), dtype=np.int16)
    audio = normalise_audio(audio, input_dtype).reshape(-1, input_channels).T[np.newaxis, ...]
    audio_length = audio.shape[-1]
    if audio_length <= 0:
        raise ValueError(f"Input audio {test_noisy_audio} contains no samples.")
    dynamic_time_axis = not isinstance(input_shape[-1], int)
    if dynamic_time_axis:
        if metadata.get("dynamic_axes") != "1":
            raise ValueError("The encoder has a dynamic time axis but metadata does not declare dynamic_axes=1.")
        frame_stride = _dynamic_metadata_int(metadata, "frame_stride")
        feature_channels = _dynamic_metadata_int(metadata, "feature_channels")
        if any(isinstance(meta.shape[1], int) for meta in enhancer_inputs) or isinstance(enhancer_output.shape[-1], int):
            raise ValueError("Dynamic encoder and enhancer time axes must be exported together.")
        if DYNAMIC_WINDOW_SAMPLES < 0 or (
            DYNAMIC_WINDOW_SAMPLES and DYNAMIC_WINDOW_SAMPLES % frame_stride
        ):
            raise ValueError(
                f"DYNAMIC_WINDOW_SAMPLES must be 0 or a positive multiple of {frame_stride}, "
                f"got {DYNAMIC_WINDOW_SAMPLES}."
            )
    else:
        input_length = int(input_shape[-1])
        num_windows = max(1, math.ceil(audio_length / input_length))
        padded_length = num_windows * input_length
        if padded_length != audio_length:
            audio = _pad_final_static_window(audio, padded_length)

        input_buffer = _ortvalue_from_meta(encoder_input, audio[..., :input_length].shape)
        encoder_output_buffers = [_ortvalue_from_meta(meta, meta.shape) for meta in encoder_outputs]
        enhancer_output_buffer = _ortvalue_from_meta(enhancer_output, enhancer_output.shape)
        encoder_binding = encoder_session.io_binding()
        encoder_binding.bind_ortvalue_input(encoder_input.name, input_buffer)
        for meta, buffer in zip(encoder_outputs, encoder_output_buffers):
            encoder_binding.bind_ortvalue_output(meta.name, buffer)
        enhancer_binding = enhancer_session.io_binding()
        for meta, buffer in zip(enhancer_inputs, encoder_output_buffers):
            enhancer_binding.bind_ortvalue_input(meta.name, buffer)
        enhancer_binding.bind_ortvalue_output(enhancer_output.name, enhancer_output_buffer)

        def process_segment(slice_start: int, slice_end: int) -> tuple[float, np.ndarray]:
            _update_ortvalue(input_buffer, audio[..., slice_start:slice_end])
            _run_iobinding(encoder_session, encoder_binding)
            _run_iobinding(enhancer_session, enhancer_binding)
            output = enhancer_binding.get_outputs()[0]
            return slice_start * 100.0 / padded_length, np.array(output.numpy(), copy=True)

    print("\nRunning UniPASE by ONNX Runtime.")
    started = time.time()
    base_length = int(round(audio_length * output_rate / input_rate))
    if dynamic_time_axis:
        window_samples = DYNAMIC_WINDOW_SAMPLES or audio_length
        results = []
        slice_start = 0
        while slice_start < audio_length:
            slice_end = min(slice_start + window_samples, audio_length)
            segment = audio[..., slice_start:slice_end]
            segment_length = segment.shape[-1]
            padded_segment_length = math.ceil(segment_length / frame_stride) * frame_stride
            if padded_segment_length != segment_length:
                segment = np.pad(segment, ((0, 0), (0, 0), (0, padded_segment_length - segment_length)))
            encoder_values = encoder_session.run(
                None,
                {encoder_input.name: segment},
                run_options=run_options,
            )
            expected_feature_shape = (1, padded_segment_length // frame_stride, feature_channels)
            if any(value.shape != expected_feature_shape for value in encoder_values):
                raise RuntimeError(
                    f"Dynamic encoder output shapes {[value.shape for value in encoder_values]} do not match "
                    f"the expected {expected_feature_shape}."
                )
            denoised_segment = enhancer_session.run(
                [enhancer_output.name],
                {meta.name: value for meta, value in zip(enhancer_inputs, encoder_values, strict=True)},
                run_options=run_options,
            )[0]
            expected_audio_shape = (1, 1, padded_segment_length)
            if denoised_segment.shape != expected_audio_shape:
                raise RuntimeError(
                    f"Dynamic enhancer output shape {denoised_segment.shape} does not match "
                    f"the expected {expected_audio_shape}."
                )
            results.append(denoised_segment[..., :segment_length])
            if slice_end < audio_length:
                print(f"Complete: {slice_end * 100.0 / audio_length:.3f}%")
            slice_start = slice_end
        denoised_audio = np.concatenate(results, axis=-1).reshape(-1)[:base_length]
    else:
        results = []
        slice_start = 0
        slice_end = input_length
        while slice_end <= padded_length:
            results.append(process_segment(slice_start, slice_end))
            print(f"Complete: {results[-1][0]:.3f}%")
            slice_start += input_length
            slice_end = slice_start + input_length
        denoised_audio = np.concatenate([result[1] for result in results], axis=-1).reshape(-1)[:base_length]
    print("Complete: 100.00%")
    final_rate = output_rate
    final_dtype = output_dtype

    if postnet_session is not None:
        postnet_input = postnet_session.get_inputs()[0]
        postnet_output = postnet_session.get_outputs()[0]
        postnet_input_length = int(postnet_input.shape[-1])
        postnet_windows = max(1, math.ceil(denoised_audio.size / postnet_input_length))
        postnet_audio = np.pad(
            denoised_audio.astype(numpy_dtype_from_onnx_meta(postnet_input), copy=False),
            (0, postnet_windows * postnet_input_length - denoised_audio.size),
        ).reshape(1, 1, -1)
        postnet_input_buffer = _ortvalue_from_meta(postnet_input, postnet_audio[..., :postnet_input_length].shape)
        postnet_output_buffer = _ortvalue_from_meta(postnet_output, postnet_output.shape)
        postnet_binding = postnet_session.io_binding()
        postnet_binding.bind_ortvalue_input(postnet_input.name, postnet_input_buffer)
        postnet_binding.bind_ortvalue_output(postnet_output.name, postnet_output_buffer)

        def process_postnet_segment(slice_start: int, slice_end: int) -> np.ndarray:
            _update_ortvalue(postnet_input_buffer, postnet_audio[..., slice_start:slice_end])
            _run_iobinding(postnet_session, postnet_binding)
            return np.array(postnet_binding.get_outputs()[0].numpy(), copy=True)

        results = []
        slice_start = 0
        slice_end = postnet_input_length
        while slice_end <= postnet_audio.shape[-1]:
            results.append(process_postnet_segment(slice_start, slice_end))
            slice_start += postnet_input_length
            slice_end = slice_start + postnet_input_length
        final_rate = int(metadata["postnet_out_sample_rate"])
        final_length = int(round(base_length * final_rate / output_rate))
        denoised_audio = np.concatenate(results, axis=-1).reshape(-1)[:final_length]
        final_dtype = numpy_dtype_from_onnx_meta(postnet_output)

    elapsed = time.time() - started
    if final_dtype == np.float16:
        denoised_audio = denoised_audio.astype(np.float32)
    output_path = save_denoised_audio.expanduser()
    if not output_path.is_absolute():
        output_path = selected_model_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        denoised_audio,
        final_rate,
        subtype="PCM_16" if final_dtype == np.int16 else "FLOAT",
    )
    audio_duration = denoised_audio.size / final_rate if final_rate > 0 else 0.0
    rtf = elapsed / audio_duration if audio_duration > 0.0 else float("inf")
    print(f"\nDenoise Process Complete.\n\nSaving to: {output_path}.\n\nReal-Time Factor (RTF): {rtf:.4f}")


if __name__ == "__main__":
    main()