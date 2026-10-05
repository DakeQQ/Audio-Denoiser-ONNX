import time
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import (
    load_runtime_metadata,
    numpy_dtype_from_onnx_meta,
    resolve_onnx_shape,
    runtime_config_from_metadata,
    validate_audio_metadata,
)
from Example_Audio import model_audio_path


parent_path              = Path(__file__).resolve().parent                            # The folder that contains this script.
onnx_model_A             = str(parent_path / "RTAEC_Optimized" / "RTAEC.onnx")            # The onnx model path.
test_near_end_audio      = model_audio_path("rtaec", "near_end")                    # The near end audio path.
test_far_end_audio       = model_audio_path("rtaec", "far_end")                     # The far end audio path.
save_aec_output          = str(parent_path / "aec.wav")                              # The output Acoustic Echo Cancellation audio path.


def _resolve_onnx_model_path(default_model_path: str) -> str:
    if len(sys.argv) <= 1:
        return default_model_path
    candidate = Path(sys.argv[1]).expanduser()
    if candidate.is_dir():
        candidate = candidate / Path(default_model_path).name
    return str(candidate)


onnx_model_A = _resolve_onnx_model_path(onnx_model_A)


ORT_Accelerate_Providers   = []       # The mixed FP16 RTAEC graph is validated on CUDA. Use [] for CPU FP32 fallback.
ORT_LOG                    = False    # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                   = False    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on CPU.
CPU_DISABLE_MATMUL_TRANSPOSE_FUSION = False
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on CPU / ORT build.
    "ConvAddActivationFusion",
]
MAX_THREADS                = 0        # Number of ONNX Runtime/OpenVINO worker threads. Set 0 for auto.
DEVICE_ID                  = 0        # The GPU id, default to 0.
NORMALIZE_AUDIO            = False    # Set True to RMS-normalize input audio before inference.
NORMALIZE_TARGET_RMS       = 4096.0   # Target RMS when NORMALIZE_AUDIO is True.
CHUNK_OVERLAP_RATIO        = 0.25     # Measured quality/compute Pareto point for the stateless block model.
INV_INT16                  = float(1.0 / 32768.0)


def align_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    device_type      = 'cpu'
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    device_type      = 'cuda'
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '0',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '0',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    device_type      = 'dml'
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'gpu',                        # [any, npu, gpu]
            'disable_metacommands': 'false',
            'enable_graph_capture': 'false',
            'enable_graph_serialization': 'false',
        }
    ]
else:
    device_type      = 'cpu'
    provider_options = None


def normalise_audio(audio: np.ndarray, input_dtype_np, target_rms=None) -> np.ndarray:
    if target_rms is None:
        target_rms = NORMALIZE_TARGET_RMS
    target_dtype = np.dtype(input_dtype_np)
    if NORMALIZE_AUDIO:
        _audio = audio.astype(np.float32)
        rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
        if rms > 0.0:
            _audio *= (target_rms / (rms + 1e-7))
        if np.issubdtype(target_dtype, np.integer):
            limits = np.iinfo(target_dtype)
            np.clip(_audio, limits.min, limits.max, out=_audio)
    else:
        _audio = audio
    if np.issubdtype(target_dtype, np.floating):
        _audio = _audio.astype(np.float32, copy=False) * INV_INT16
    return _audio.astype(target_dtype, copy=False)


def _build_run_options(silent: bool) -> onnxruntime.RunOptions:
    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level  = 0 if not silent else 4
    run_options.log_verbosity_level = 4
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return run_options


def _build_session_opts_ort() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level       = 0 if ORT_LOG else 4
    opts.log_verbosity_level      = 4
    opts.inter_op_num_threads     = MAX_THREADS
    opts.intra_op_num_threads     = MAX_THREADS
    opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    cfgs = {
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
    for key, value in cfgs.items():
        opts.add_session_config_entry(key, value)
    return opts


def _ortvalue_from_meta(meta, runtime_shape):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(
            resolve_onnx_shape(meta, runtime_shape),
            dtype=numpy_dtype_from_onnx_meta(meta),
        ),
        device_type,
        DEVICE_ID,
    )


def _update_ortvalue(ort_value, array):
    array = np.ascontiguousarray(array)
    if hasattr(ort_value, "update_inplace"):
        ort_value.update_inplace(array)
    else:
        np.copyto(ort_value.numpy(), array)


def _run_iobinding(session, binding):
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
if _CPU_EP_ONLY and CPU_DISABLE_MATMUL_TRANSPOSE_FUSION:
    disabled_opts.append("MatmulTransposeFusion")
if _CPU_EP_ONLY:
    disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
disabled_opts = disabled_opts or None
_packed = {
    'sess_options': session_opts_ort,
    'providers': ORT_Accelerate_Providers or ['CPUExecutionProvider'],
    'provider_options': provider_options,
    'disabled_optimizers': disabled_opts,
}

# Auto-fallback to exported ONNX model if optimized model is not generated yet
if not Path(onnx_model_A).is_file():
    _fallback_model = str(parent_path / "RTAEC_ONNX" / "RTAEC.onnx")
    if Path(_fallback_model).is_file():
        print(f"Optimized model not found at {onnx_model_A}; falling back to exported model: {_fallback_model}")
        onnx_model_A = _fallback_model

ort_session_A = _make_session(onnx_model_A)
_model_metadata = load_runtime_metadata(onnx_model_A, _make_session)
validate_audio_metadata(_model_metadata, ort_session_A)
_runtime_config = runtime_config_from_metadata(_model_metadata)
IN_SAMPLE_RATE = _runtime_config["IN_SAMPLE_RATE"]
OUT_SAMPLE_RATE = _runtime_config["OUT_SAMPLE_RATE"]
MODEL_SAMPLE_RATE = _runtime_config["MODEL_SAMPLE_RATE"]
INPUT_TO_OUTPUT_SCALE = _runtime_config["INPUT_TO_OUTPUT_SCALE"]
BATCH_WINDOW_SECONDS = _runtime_config["BATCH_WINDOW_SECONDS"]
HOP_LENGTH = _runtime_config["HOP_LENGTH"]
FOLD_WINDOW_LENGTH = _runtime_config["FOLD_WINDOW_LENGTH"]
FOLD_INPUT_LENGTH = _runtime_config["FOLD_INPUT_LENGTH"]
BATCH_FOLD_INFERENCE = _runtime_config["BATCH_FOLD_INFERENCE"]
MAX_DYNAMIC_AUDIO_SECONDS = _runtime_config["MAX_DYNAMIC_AUDIO_SECONDS"]
INPUT_CHANNELS = _runtime_config["INPUT_CHANNELS"]
OUTPUT_CHANNELS = _runtime_config["OUTPUT_CHANNELS"]
N_CHANNELS = _runtime_config["N_CHANNELS"]
NUM_AUDIO_INPUTS = _runtime_config["NUM_AUDIO_INPUTS"]
PAD_HEAD = _runtime_config["PAD_HEAD"]
ENC_STRIDE = _runtime_config["ENC_STRIDE"]
OUTPUT_SOURCES = _runtime_config["OUTPUT_SOURCES"]
ORIGINAL_SAMPLE_RATE = _runtime_config["ORIGINAL_SAMPLE_RATE"]
SUPER_SAMPLE_RATE = _runtime_config["SUPER_SAMPLE_RATE"]
SCALE_FACTOR = _runtime_config["SCALE_FACTOR"]
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name
output_dtype_np = numpy_dtype_from_onnx_meta(out_name_A[0])


def load_model_audio(path, input_meta):
    audio_segment = AudioSegment.from_file(path).set_frame_rate(IN_SAMPLE_RATE)
    input_shape = input_meta.shape
    if len(input_shape) != 3:
        raise ValueError(
            f"Expected rank-3 ONNX audio input {input_meta.name!r}, got shape {input_shape}."
        )
    input_channels = input_shape[1] if isinstance(input_shape[1], int) else audio_segment.channels
    audio = np.array(
        audio_segment.set_channels(input_channels).get_array_of_samples(),
        dtype=np.int16,
    )
    audio = normalise_audio(audio, numpy_dtype_from_onnx_meta(input_meta))
    return audio.reshape(-1, input_channels).T[np.newaxis, ...]


# Load the input audio
print(f"\nTest Input Near_End Audio: {test_near_end_audio}\nTest Input Far_End Audio: {test_far_end_audio}")
near_end_audio = load_model_audio(test_near_end_audio, in_name_A[0])
far_end_audio = load_model_audio(test_far_end_audio, in_name_A[1])
near_end_audio_len = near_end_audio.shape[-1]
far_nd_audio_len = far_end_audio.shape[-1]
min_len = min(near_end_audio_len, far_nd_audio_len)
near_end_audio = near_end_audio[..., :min_len]
far_end_audio = far_end_audio[..., :min_len]
shape_value_in = in_name_A[0].shape[-1]
shape_value_out = out_name_A[0].shape[-1]
if not isinstance(shape_value_in, int):
    INPUT_AUDIO_LENGTH = max(MAX_DYNAMIC_AUDIO_SECONDS * IN_SAMPLE_RATE, min_len)  # Default to slice in 30 seconds.
    if BATCH_FOLD_INFERENCE and INPUT_AUDIO_LENGTH / IN_SAMPLE_RATE > BATCH_WINDOW_SECONDS:
        INPUT_AUDIO_LENGTH = align_to_multiple(INPUT_AUDIO_LENGTH, FOLD_INPUT_LENGTH)
else:
    INPUT_AUDIO_LENGTH = shape_value_in
fold_active = BATCH_FOLD_INFERENCE and isinstance(INPUT_AUDIO_LENGTH, int) and INPUT_AUDIO_LENGTH >= FOLD_INPUT_LENGTH
if fold_active:
    print(f"Batch Fold Window: {FOLD_WINDOW_LENGTH} model samples; ONNX Input Slice: {INPUT_AUDIO_LENGTH} input samples")


def chunk_geometry(audio_len):
    if audio_len > INPUT_AUDIO_LENGTH:
        input_hop_length = max(
            1,
            int(round(HOP_LENGTH * IN_SAMPLE_RATE / MODEL_SAMPLE_RATE)),
        )
        input_stride = align_to_multiple(
            int(round(INPUT_AUDIO_LENGTH * (1.0 - CHUNK_OVERLAP_RATIO))),
            input_hop_length,
        )
        input_stride = min(
            max(input_hop_length, input_stride),
            INPUT_AUDIO_LENGTH - input_hop_length,
        )
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / input_stride)) + 1
    else:
        input_stride = INPUT_AUDIO_LENGTH
        num_windows = 1

    aligned_input_length = (num_windows - 1) * input_stride + INPUT_AUDIO_LENGTH
    output_window_length = (
        shape_value_out
        if isinstance(shape_value_out, int)
        else max(1, int(round(INPUT_AUDIO_LENGTH * INPUT_TO_OUTPUT_SCALE)))
    )
    output_stride = max(
        1,
        int(round(input_stride * INPUT_TO_OUTPUT_SCALE)),
    )
    output_overlap = max(0, output_window_length - output_stride)
    aligned_output_length = (num_windows - 1) * output_stride + output_window_length
    return (
        num_windows,
        aligned_input_length,
        input_stride,
        output_window_length,
        output_stride,
        output_overlap,
        aligned_output_length,
    )


def pad_audio(audio, aligned_length):
    pad_amount = aligned_length - audio.shape[-1]
    if pad_amount <= 0:
        return audio
    pad_block = np.zeros((*audio.shape[:-1], pad_amount), dtype=audio.dtype)
    return np.concatenate((audio, pad_block), axis=-1)


original_input_length = min_len
(
    num_windows,
    aligned_input_length,
    input_stride,
    output_window_length,
    output_stride,
    output_overlap,
    aligned_output_length,
) = chunk_geometry(original_input_length)
near_end_audio = pad_audio(near_end_audio, aligned_input_length)
far_end_audio = pad_audio(far_end_audio, aligned_input_length)

min_len = int(round(original_input_length * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))

input_buffer_0 = _ortvalue_from_meta(
    in_name_A[0],
    near_end_audio[..., :INPUT_AUDIO_LENGTH].shape,
)
input_buffer_1 = _ortvalue_from_meta(
    in_name_A[1],
    far_end_audio[..., :INPUT_AUDIO_LENGTH].shape,
)
binding_A = ort_session_A.io_binding()
binding_A.bind_ortvalue_input(in_name_A0, input_buffer_0)
binding_A.bind_ortvalue_input(in_name_A1, input_buffer_1)
binding_A.bind_output(out_name_A0, device_type, DEVICE_ID)


def process_segment(_slice_start, _slice_end, _near_end_audio, _far_end_audio):
    _update_ortvalue(input_buffer_0, _near_end_audio[:, :, _slice_start: _slice_end])
    _update_ortvalue(input_buffer_1, _far_end_audio[:, :, _slice_start: _slice_end])
    _run_iobinding(ort_session_A, binding_A)
    output = binding_A.get_outputs()[0]
    return np.array(output.numpy(), copy=True)


# Start to run RTAEC
print("\nRunning the RTAEC by ONNX Runtime.")
use_ola = output_overlap > 0
if use_ola:
    fade_phase = (np.arange(output_overlap, dtype=np.float32) + 0.5) / output_overlap
    fade_in = np.sin(0.5 * np.pi * fade_phase) ** 2
    fade_out = fade_in[::-1].copy()
    out_buffer = np.empty(aligned_output_length, dtype=np.float32)

saved_chunks = []
start_time = time.time()
for chunk_index in range(num_windows):
    slice_start = chunk_index * input_stride
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    out_chunk = process_segment(slice_start, slice_end, near_end_audio, far_end_audio)
    pct = 100.0 * min(slice_start, original_input_length) / original_input_length
    print(f"Complete: {pct:.3f}%")
    if use_ola:
        chunk_audio = out_chunk.reshape(-1).astype(np.float32, copy=False)
        if chunk_audio.size != output_window_length:
            raise RuntimeError(
                f"Unexpected RTAEC output length: {chunk_audio.size}, expected {output_window_length}"
            )
        output_start = chunk_index * output_stride
        output_end = output_start + output_window_length
        if chunk_index == 0:
            out_buffer[output_start:output_end] = chunk_audio
        else:
            overlap_end = output_start + output_overlap
            out_buffer[output_start:overlap_end] = (
                out_buffer[output_start:overlap_end] * fade_out
                + chunk_audio[:output_overlap] * fade_in
            )
            out_buffer[overlap_end:output_end] = chunk_audio[output_overlap:]
    else:
        saved_chunks.append(out_chunk)

if use_ola:
    denoised_wav = out_buffer[:min_len]
else:
    denoised_wav = np.concatenate(saved_chunks, axis=-1).reshape(-1)[:min_len]
end_time = time.time()
print(f"Complete: 100.00%")

# Save the denoised wav.
elapsed = end_time - start_time
audio_duration = min_len / OUT_SAMPLE_RATE
rtf = elapsed / audio_duration
if output_dtype_np == np.float16:
    denoised_wav = denoised_wav.astype(np.float32)
if output_dtype_np == np.int16:
    if denoised_wav.dtype != np.int16:
        denoised_wav = np.rint(denoised_wav).clip(-32768.0, 32767.0).astype(np.int16)
    save_subtype = 'PCM_16'
else:
    denoised_wav = np.clip(denoised_wav, -1.0, 1.0)
    save_subtype = 'PCM_16'
sf.write(save_aec_output, denoised_wav, OUT_SAMPLE_RATE, subtype=save_subtype)
print(f"\nAEC Process Complete.\n\nSaving to: {save_aec_output}.\n\nReal-Time Factor (RTF): {rtf:.4f}")
