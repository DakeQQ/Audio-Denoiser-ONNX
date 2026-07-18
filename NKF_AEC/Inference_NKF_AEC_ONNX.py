import time
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import load_runtime_metadata, runtime_config_from_metadata, validate_audio_metadata
from Example_Audio import model_audio_path


parent_path         = Path(__file__).resolve().parent                         # The folder that contains this script.
onnx_model_A        = str(parent_path / "NKF_AEC_Optimized" / "NKF_AEC.onnx") # The optimized onnx model path.
test_near_end_audio = model_audio_path("nkf_aec", "near_end")                # The near end audio path.
test_far_end_audio  = model_audio_path("nkf_aec", "far_end")                 # The far end audio path.
save_aec_output     = str(parent_path / "aec.wav")                            # The output Acoustic Echo Cancellation audio path.


def _resolve_onnx_model_path(default_model_path: str) -> str:
    if len(sys.argv) <= 1:
        return default_model_path
    candidate = Path(sys.argv[1]).expanduser()
    if candidate.is_dir():
        candidate = candidate / Path(default_model_path).name
    return str(candidate)


onnx_model_A = _resolve_onnx_model_path(onnx_model_A)


ORT_Accelerate_Providers  = []       # If you have accelerated devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider']
                                      # else keep empty.
ORT_LOG                   = False    # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                  = False    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS               = 0        # Number of ONNX Runtime/OpenVINO worker threads. Set 0 for auto.
DEVICE_ID                 = 0        # The GPU id, default to 0.
NORMALIZE_AUDIO           = False    # Set True to RMS-normalize input audio before inference.
NORMALIZE_TARGET_RMS      = 4096.0   # Target RMS when NORMALIZE_AUDIO is True.


def align_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
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
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
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
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
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
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()
else:
    # Please config by yourself for others providers.
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
    provider_options = None

_ort_device_obj = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)


def normalise_audio(audio: np.ndarray, input_dtype_np, target_rms=None) -> np.ndarray:
    if target_rms is None:
        target_rms = NORMALIZE_TARGET_RMS
    # Fuse the pydub int16 samples, the optional RMS normalisation and the single cast to the
    # model input dtype. pydub returns int16 PCM; for a float model input those int16 values are
    # cast straight to float, the ZipEnhancer require [-32768, 32767] float values.
    if NORMALIZE_AUDIO:
        _audio = audio.astype(np.float32)
        rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
        if rms > 0.0:
            _audio *= (target_rms / (rms + 1e-7))
        if input_dtype_np == np.int16:
            np.clip(_audio, -32768.0, 32767.0, out=_audio)
            return _audio.astype(np.int16)
        # _audio is already float32, so only a float16 model input needs a further cast.
        if input_dtype_np == np.float16:
            return _audio.astype(np.float16)
        return _audio
    if input_dtype_np == np.int16:
        return audio
    return audio.astype(input_dtype_np, copy=False)


def _build_run_options(silent: bool) -> onnxruntime.RunOptions:
    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level = 0 if not silent else 4
    run_options.log_verbosity_level = 4
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return run_options


def _build_session_opts_ort() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 0 if ORT_LOG else 4
    opts.log_verbosity_level = 4
    opts.inter_op_num_threads = MAX_THREADS
    opts.intra_op_num_threads = MAX_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
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


def _numpy_dtype_from_meta(meta):
    meta_type = meta.type
    if "int16" in meta_type:
        return np.int16
    if "int32" in meta_type:
        return np.int32
    if "int64" in meta_type:
        return np.int64
    if "float16" in meta_type:
        return np.float16
    return np.float32


def _ort_zeros(shape, dtype):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device_type, DEVICE_ID)


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
if _CPU_EP_ONLY:
    disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
disabled_opts = disabled_opts or None
_packed = {
    'sess_options': session_opts_ort,
    'providers': ORT_Accelerate_Providers or ['CPUExecutionProvider'],
    'provider_options': provider_options,
    'disabled_optimizers': disabled_opts,
}

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
input_dtype_np = _numpy_dtype_from_meta(in_name_A[0])
output_dtype_np = _numpy_dtype_from_meta(out_name_A[0])


# Load the input audio
print(f"\nTest Input Near_End Audio: {test_near_end_audio}\nTest Input Far_End Audio: {test_far_end_audio}")
near_end_audio = np.array(AudioSegment.from_file(test_near_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
far_end_audio = np.array(AudioSegment.from_file(test_far_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
near_end_audio_len = len(near_end_audio)
far_end_audio_len = len(far_end_audio)
min_len = min(near_end_audio_len, far_end_audio_len)
near_end_audio = normalise_audio(near_end_audio, input_dtype_np)
far_end_audio = normalise_audio(far_end_audio, input_dtype_np)
near_end_audio = near_end_audio[:min_len]
far_end_audio = far_end_audio[:min_len]
near_end_audio = near_end_audio.reshape(1, 1, -1)
far_end_audio = far_end_audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = max(MAX_DYNAMIC_AUDIO_SECONDS * IN_SAMPLE_RATE, min_len)  # Default to slice in 4 seconds. You can adjust it.
    if BATCH_FOLD_INFERENCE and INPUT_AUDIO_LENGTH / IN_SAMPLE_RATE > BATCH_WINDOW_SECONDS:
        INPUT_AUDIO_LENGTH = align_to_multiple(INPUT_AUDIO_LENGTH, FOLD_INPUT_LENGTH)
else:
    INPUT_AUDIO_LENGTH = shape_value_in
fold_active = BATCH_FOLD_INFERENCE and isinstance(INPUT_AUDIO_LENGTH, int) and INPUT_AUDIO_LENGTH >= FOLD_INPUT_LENGTH
if fold_active:
    print(f"Batch Fold Window: {FOLD_WINDOW_LENGTH} model samples; ONNX Input Slice: {INPUT_AUDIO_LENGTH} input samples")


def align_audio(audio, audio_len):
    stride_step = INPUT_AUDIO_LENGTH
    if audio_len > INPUT_AUDIO_LENGTH:
        if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int) & (OUT_SAMPLE_RATE == IN_SAMPLE_RATE):
            stride_step = shape_value_out
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        if fold_active:
            pad_block = np.zeros((1, 1, pad_amount), dtype=audio.dtype)
        else:
            final_slice = audio[:, :, -pad_amount:].astype(np.float32)
            pad_block = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, pad_block), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        if fold_active:
            pad_block = np.zeros((1, 1, INPUT_AUDIO_LENGTH - audio_len), dtype=audio.dtype)
        else:
            audio_float = audio.astype(np.float32)
            pad_block = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, pad_block), axis=-1)
    aligned_len = audio.shape[-1]
    return audio, aligned_len, stride_step


near_end_audio, _, _ = align_audio(near_end_audio, min_len)
far_end_audio, aligned_len, stride_step = align_audio(far_end_audio, min_len)

min_len = int(min_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
inv_audio_len = float(100.0 / min_len)

output_audio_length = shape_value_out if isinstance(shape_value_out, int) else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
input_buffer_0 = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), input_dtype_np)
input_buffer_1 = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), input_dtype_np)
output_buffer = _ort_zeros((1, 1, output_audio_length), output_dtype_np)
binding_A = ort_session_A.io_binding()
binding_A.bind_ortvalue_input(in_name_A0, input_buffer_0)
binding_A.bind_ortvalue_input(in_name_A1, input_buffer_1)
binding_A.bind_ortvalue_output(out_name_A0, output_buffer)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _near_end_audio, _far_end_audio):
    _update_ortvalue(input_buffer_0, _far_end_audio[:, :, _slice_start: _slice_end])
    _update_ortvalue(input_buffer_1, _near_end_audio[:, :, _slice_start: _slice_end])
    _run_iobinding(ort_session_A, binding_A)
    return _slice_start * _inv_audio_len, np.array(output_buffer.numpy(), copy=True)


# Start to run NKF-AEC
print("\nRunning the NKF-AEC by ONNX Runtime.")
results = []
start_time = time.time()
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
while slice_end <= aligned_len:
    results.append(process_segment(inv_audio_len, slice_start, slice_end, near_end_audio, far_end_audio))
    print(f"Complete: {results[-1][0]:.3f}%")
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
saved = [result[1] for result in results]
enhanced_wav = np.concatenate(saved, axis=-1).reshape(-1)[:min_len]
end_time = time.time()
print(f"Complete: 100.00%")

# Save the enhanced wav.
elapsed = end_time - start_time
audio_duration = min_len / OUT_SAMPLE_RATE
rtf = elapsed / audio_duration
if output_dtype_np == np.float16:
    enhanced_wav = enhanced_wav.astype(np.float32)
sf.write(save_aec_output, enhanced_wav, OUT_SAMPLE_RATE, subtype='PCM_16' if output_dtype_np == np.int16 else 'FLOAT')
print(f"\nAEC Process Complete.\n\nSaving to: {save_aec_output}.\n\nReal-Time Factor (RTF): {rtf:.4f}")
