"""Run a static end-to-end DPDFNet audio ONNX model with reusable ORT buffers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "Example_Audio.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Example_Audio.py and audio_onnx_metadata.py")
from Example_Audio import example_audio_path
from audio_onnx_metadata import (
    load_runtime_metadata,
    numpy_dtype_from_onnx_meta,
    runtime_config_from_metadata,
    validate_audio_metadata,
)


parent_path = Path(__file__).resolve().parent
onnx_model_A = parent_path / "DPDFNet_Optimized" / "DPDFNet.onnx"
test_noisy_audio = Path(example_audio_path("denoise/speech_with_noise1.wav"))
save_denoised_audio = parent_path / "denoised.wav"


ORT_Accelerate_Providers = []  # Choose one: CUDAExecutionProvider, OpenVINOExecutionProvider, DmlExecutionProvider.
ORT_LOG = False
ORT_FP16 = False
CPU_DISABLE_MATMUL_ADD_FUSION = True
CPU_DISABLE_NCHWC = True
CPU_EXTRA_DISABLED_OPTIMIZERS = ["ConvAddActivationFusion", "MatmulTransposeFusion"]
MAX_THREADS = 0  # Zero lets ONNX Runtime select the thread count.
DEVICE_ID = 0
NORMALIZE_AUDIO = False
NORMALIZE_TARGET_RMS = 4096.0
INV_INT16 = float(1.0 / 32768.0)


def _resolve_onnx_model_path(default_model_path: Path) -> Path:
    if len(sys.argv) <= 1:
        return default_model_path
    candidate = Path(sys.argv[1]).expanduser()
    if candidate.is_dir():
        candidate = candidate / default_model_path.name
    return candidate


onnx_model_A = _resolve_onnx_model_path(onnx_model_A)


def resolve_model_path(candidate: Path) -> Path:
    candidate = candidate.expanduser()
    if candidate.is_dir():
        candidate = candidate / onnx_model_A.name
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"DPDFNet ONNX model was not found: {candidate}")
    return candidate


def _selected_provider() -> tuple[list[str], list[dict], str]:
    requested = set(ORT_Accelerate_Providers)
    if "OpenVINOExecutionProvider" in requested:
        options = {
            "device_type": "CPU",  # CPU, NPU, GPU, GPU.0, GPU.1
            "precision": "ACCURACY",  # FP32, FP16, ACCURACY
            "num_of_threads": MAX_THREADS if MAX_THREADS != 0 else 8,
            "num_streams": 1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"], [options, {}], "cpu"
    if "CUDAExecutionProvider" in requested:
        options = {
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
            "do_copy_in_default_stream": "1",
            "enable_cuda_graph": "0",
            "prefer_nhwc": "0",
            "enable_skip_layer_norm_strict_mode": "0",
            "use_ep_level_unified_stream": "0",
        }
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [options, {}], "cuda"
    if "DmlExecutionProvider" in requested:
        options = {
            "device_id": DEVICE_ID,
            "performance_preference": "high_performance",
            "device_filter": "gpu",
            "disable_metacommands": "false",
            "enable_graph_capture": "false",
            "enable_graph_serialization": "false",
        }
        return ["DmlExecutionProvider", "CPUExecutionProvider"], [options, {}], "dml"
    return ["CPUExecutionProvider"], [{}], "cpu"


def build_session_options() -> onnxruntime.SessionOptions:
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
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else ""
        ),
    }
    for key, value in configs.items():
        options.add_session_config_entry(key, value)
    return options


def build_run_options() -> onnxruntime.RunOptions:
    options = onnxruntime.RunOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options


def create_session(model_path: Path) -> tuple[onnxruntime.InferenceSession, str, onnxruntime.RunOptions]:
    providers, provider_options, device_type = _selected_provider()
    cpu_only = providers == ["CPUExecutionProvider"]
    disabled_optimizers = []
    if ORT_FP16:
        disabled_optimizers.extend(["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"])
    if cpu_only and CPU_DISABLE_MATMUL_ADD_FUSION:
        disabled_optimizers.append("MatMulAddFusion")
    if cpu_only and CPU_DISABLE_NCHWC:
        disabled_optimizers.append("NchwcTransformer")
    if cpu_only:
        disabled_optimizers.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=build_session_options(),
        providers=providers,
        provider_options=provider_options,
        disabled_optimizers=disabled_optimizers or None,
    )
    return session, device_type, build_run_options()


def _static_shape(meta) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in meta.shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"DPDFNet expects static ONNX shapes, but {meta.name!r} has shape {meta.shape}."
        ) from error


def _update_ortvalue(ort_value: onnxruntime.OrtValue, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values)
    if hasattr(ort_value, "update_inplace"):
        ort_value.update_inplace(values)
    else:
        np.copyto(ort_value.numpy(), values)


def normalise_audio(audio: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Convert decoded int16 PCM to the dtype expected by the exported audio graph."""

    values = audio.astype(np.float32, copy=False)
    if NORMALIZE_AUDIO:
        rms = np.sqrt(np.mean(values * values, dtype=np.float32), dtype=np.float32)
        if rms > 0.0:
            values = values * (NORMALIZE_TARGET_RMS / (rms + 1e-7))
    if np.issubdtype(target_dtype, np.integer):
        limits = np.iinfo(target_dtype)
        return np.clip(values, limits.min, limits.max).astype(target_dtype, copy=False)
    return (values * INV_INT16).astype(target_dtype, copy=False)


def main() -> None:
    model_path = resolve_model_path(onnx_model_A)
    input_path = test_noisy_audio.expanduser().resolve()
    output_path = save_denoised_audio.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio was not found: {input_path}")

    session, device_type, run_options = create_session(model_path)
    metadata = load_runtime_metadata(
        model_path,
        lambda metadata_model_path: create_session(Path(metadata_model_path))[0],
    )
    validate_audio_metadata(metadata, session)
    runtime_config = runtime_config_from_metadata(metadata)
    in_sample_rate = runtime_config["IN_SAMPLE_RATE"]
    out_sample_rate = runtime_config["OUT_SAMPLE_RATE"]
    input_to_output_scale = runtime_config["INPUT_TO_OUTPUT_SCALE"]
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("The end-to-end DPDFNet graph must expose exactly one audio input and output.")
    input_meta, output_meta = inputs[0], outputs[0]
    input_shape = _static_shape(input_meta)
    output_shape = _static_shape(output_meta)
    if len(input_shape) != 3 or len(output_shape) != 3 or input_shape[:2] != (1, 1) or output_shape[:2] != (1, 1):
        raise ValueError(
            f"Expected static mono audio I/O, got input={input_shape}, output={output_shape}."
        )
    input_dtype = numpy_dtype_from_onnx_meta(input_meta)
    output_dtype = numpy_dtype_from_onnx_meta(output_meta)
    supported_dtypes = {np.dtype(np.int16), np.dtype(np.float16), np.dtype(np.float32)}
    if np.dtype(input_dtype) not in supported_dtypes or np.dtype(output_dtype) not in supported_dtypes:
        raise TypeError(f"Unsupported DPDFNet audio dtypes: input={input_dtype}, output={output_dtype}.")

    print(f"\nUsable Providers: {session.get_providers()}")
    print(f"Model: {model_path}")
    print(
        f"DPDFNet profile: {metadata.string('model_name', required=True)}; "
        f"input={in_sample_rate} Hz/{np.dtype(input_dtype).name}, "
        f"output={out_sample_rate} Hz/{np.dtype(output_dtype).name}"
    )
    print(f"Static audio capacity: {input_shape[-1]} input samples -> {output_shape[-1]} output samples")

    audio_segment = (
        AudioSegment.from_file(str(input_path))
        .set_frame_rate(in_sample_rate)
        .set_channels(1)
        .set_sample_width(2)
    )
    source_pcm = np.asarray(audio_segment.get_array_of_samples(), dtype=np.int16)
    if source_pcm.size == 0:
        raise ValueError("Cannot process an empty audio file.")
    audio = normalise_audio(source_pcm, np.dtype(input_dtype)).reshape(1, 1, -1)
    source_length = audio.shape[-1]
    input_length = input_shape[-1]
    if input_length < 1:
        raise ValueError("DPDFNet ONNX input audio length must be positive.")
    num_windows = max(1, int(np.ceil(source_length / input_length)))
    padded_length = num_windows * input_length
    if padded_length != source_length:
        audio = np.pad(audio, ((0, 0), (0, 0), (0, padded_length - source_length)))

    input_buffer = onnxruntime.OrtValue.ortvalue_from_numpy(
        np.empty(input_shape, dtype=input_dtype), device_type, DEVICE_ID
    )
    binding = session.io_binding()
    binding.bind_ortvalue_input(input_meta.name, input_buffer)
    binding.bind_output(output_meta.name, device_type, DEVICE_ID)

    start_time = time.perf_counter()
    output_chunks = []
    for window_index in range(num_windows):
        start = window_index * input_length
        end = start + input_length
        _update_ortvalue(input_buffer, audio[..., start:end])
        session.run_with_iobinding(binding, run_options=run_options)
        output_chunks.append(np.array(binding.get_outputs()[0].numpy(), copy=True))
        print(f"Complete: {(window_index + 1) * 100.0 / num_windows:.2f}%")
    elapsed = time.perf_counter() - start_time
    target_output_length = int(round(source_length * input_to_output_scale))
    denoised_audio = np.concatenate(output_chunks, axis=-1).reshape(-1)[:target_output_length]
    if output_dtype == np.float16:
        denoised_audio = denoised_audio.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        denoised_audio,
        out_sample_rate,
        subtype="PCM_16" if output_dtype == np.int16 else "FLOAT",
    )
    duration_seconds = target_output_length / out_sample_rate
    rtf = elapsed / duration_seconds if duration_seconds > 0 else float("inf")
    print(
        f"\nDenoise process complete.\n\nSaving to: {output_path}\n\n"
        f"Windows: {num_windows}; end-to-end RTF: {rtf:.4f}"
    )


if __name__ == "__main__":
    main()