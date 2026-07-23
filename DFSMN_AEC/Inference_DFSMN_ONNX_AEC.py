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


parent_path          = Path(__file__).resolve().parent                                # The folder that contains this script.
onnx_model_A         = str(parent_path / "DFSMN_AEC_Optimized" / "DFSMN_AEC.onnx")   # The optimized onnx model path.
test_near_end_audio  = model_audio_path("dfsmn_aec", "near_end")                    # The near end audio path.
test_far_end_audio   = model_audio_path("dfsmn_aec", "far_end")                     # The far end audio path.
save_aec_output      = str(parent_path / "aec.wav")                                  # The output Acoustic Echo Cancellation audio path.
save_timestamps_second = parent_path / "timestamps_second.txt"                       # VAD speech timestamps in hh:mm:ss.mmm format.
save_timestamps_indices = parent_path / "timestamps_indices.txt"                     # VAD speech timestamps in input-sample indices.


def _resolve_onnx_model_path(default_model_path: str) -> str:
    if len(sys.argv) <= 1:
        return default_model_path
    candidate = Path(sys.argv[1]).expanduser()
    if candidate.is_dir():
        candidate = candidate / Path(default_model_path).name
    return str(candidate)


onnx_model_A = _resolve_onnx_model_path(onnx_model_A)


ORT_Accelerate_Providers  = []  # The mixed FP16/FP32 graph is validated on CUDA. Use [] for CPU fallback.
                                        # else keep empty.
ORT_LOG                    = False      # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                   = False      # Preserve the exported mixed-precision policy and disable runtime FP16 recasting passes.
CPU_DISABLE_MATMUL_ADD_FUSION = True  # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True              # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [     # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]
MAX_THREADS                = 0          # Number of ONNX Runtime/OpenVINO worker threads. Set 0 for auto.
DEVICE_ID                  = 0          # The GPU id, default to 0.
NORMALIZE_AUDIO            = False      # Set True to RMS-normalize input audio before inference.
NORMALIZE_TARGET_RMS       = 4096.0     # Target RMS when NORMALIZE_AUDIO is True.


def align_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type':                  'CPU',       # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision':                    'ACCURACY',  # [FP32, FP16, ACCURACY]
            'num_of_threads':               MAX_THREADS if MAX_THREADS != 0 else 8,
            'num_streams':                  1,
            'enable_opencl_throttling':     False,
            'enable_qdq_optimizer':         False,       # Enable it carefully
            'disable_dynamic_shapes':       False
        }
    ]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id':                            DEVICE_ID,
            'gpu_mem_limit':                        24 * 1024 * 1024 * 1024,  # 24 GB
            'arena_extend_strategy':                'kNextPowerOfTwo',        # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search':               'EXHAUSTIVE',            # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel':                          '2',                     # ["0", "1", "2"]
            'use_tf32':                             '0',                     # Match the mixed-precision CUDA validation profile.
            'fuse_conv_bias':                       '0',                     # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace':         '1',
            'cudnn_conv1d_pad_to_nc1d':             '0',
            'tunable_op_enable':                    '0',
            'tunable_op_tuning_enable':             '0',
            'tunable_op_max_tuning_duration_ms':    10,
            'do_copy_in_default_stream':            '0',
            'enable_cuda_graph':                    '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc':                          '0',
            'enable_skip_layer_norm_strict_mode':   '0',
            'use_ep_level_unified_stream':          '0',
        }
    ]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id':                    DEVICE_ID,
            'performance_preference':       'high_performance',  # [high_performance, default, minimum_power]
            'device_filter':                'gpu',               # [any, npu, gpu]
            'disable_metacommands':         'false',
            'enable_graph_capture':         'false',
            'enable_graph_serialization':   'false',
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
    run_options                     = onnxruntime.RunOptions()
    run_options.log_severity_level  = 0 if not silent else 4
    run_options.log_verbosity_level = 4
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return run_options


def _build_session_opts_ort() -> onnxruntime.SessionOptions:
    opts                          = onnxruntime.SessionOptions()
    opts.log_severity_level       = 0 if ORT_LOG else 4
    opts.log_verbosity_level      = 4
    opts.inter_op_num_threads     = MAX_THREADS
    opts.intra_op_num_threads     = MAX_THREADS
    opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    cfgs = {
        "session.set_denormal_as_zero":                  "1",
        "session.intra_op.allow_spinning":               "1",
        "session.inter_op.allow_spinning":               "1",
        "session.enable_quant_qdq_cleanup":              "1",
        "session.qdq_matmulnbits_accuracy_level":        "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level":        "2",
        "optimization.enable_gelu_approximation":        "1",
        "optimization.minimal_build_optimizations":      "",
        "optimization.enable_cast_chain_elimination":    "1",
        "optimization.disable_specified_optimizers":     (
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
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype),
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
if _CPU_EP_ONLY:
    disabled_opts.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
disabled_opts = disabled_opts or None
_packed = {
    'sess_options': session_opts_ort,
    'providers': ORT_Accelerate_Providers or ["CPUExecutionProvider"],
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
OUTPUT_VAD_RESULT = _model_metadata.optional_bool("output_vad_result", False)
if OUTPUT_VAD_RESULT:
    OUTPUT_FRAME_SHIFT_SECONDS = _model_metadata.required_float("output_frame_shift_seconds")
    OUTPUT_FRAME_LENGTH = _model_metadata.required_int("output_frame_shift_samples")
    FBANK_WINDOW_LENGTH = _model_metadata.required_int("fbank_window_length_samples")
    SPEAKING_SCORE = _model_metadata.required_float("speaking_score")
    SILENCE_SCORE = _model_metadata.required_float("silence_score")
    LOOK_AHEAD = _model_metadata.required_float("look_ahead_seconds")
    FUSION_THRESHOLD = _model_metadata.required_float("fusion_threshold_seconds")
    MIN_SPEECH_DURATION = _model_metadata.required_float("min_speech_duration_seconds")
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
expected_outputs = _model_metadata.optional_int("num_outputs", 2 if OUTPUT_VAD_RESULT else 1)
if len(out_name_A) != expected_outputs:
    raise ValueError(
        f"ONNX model has {len(out_name_A)} outputs, metadata num_outputs={expected_outputs}.")
if OUTPUT_VAD_RESULT and (len(out_name_A) != 2 or out_name_A[1].name != "vad_results"):
    raise ValueError("OUTPUT_VAD_RESULT requires a second ONNX output named 'vad_results'.")
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name
input_dtype_np = _numpy_dtype_from_meta(in_name_A[0])
output_dtype_np = _numpy_dtype_from_meta(out_name_A[0])
if OUTPUT_VAD_RESULT:
    out_name_A1 = out_name_A[1].name
    vad_output_dtype_np = _numpy_dtype_from_meta(out_name_A[1])


# Load the input audio
print(f"\nTest Input Near_End Audio: {test_near_end_audio}\nTest Input Far_End Audio: {test_far_end_audio}")
near_end_audio = np.array(AudioSegment.from_file(test_near_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
far_end_audio = np.array(AudioSegment.from_file(test_far_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
near_end_audio_len = len(near_end_audio)
far_nd_audio_len = len(far_end_audio)
min_len = min(near_end_audio_len, far_nd_audio_len)
input_audio_len = min_len
near_end_audio = near_end_audio[:min_len]
far_end_audio = far_end_audio[:min_len]
near_end_audio = normalise_audio(near_end_audio, input_dtype_np)
far_end_audio = normalise_audio(far_end_audio, input_dtype_np)
near_end_audio = near_end_audio.reshape(1, 1, -1)
far_end_audio = far_end_audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(MAX_DYNAMIC_AUDIO_SECONDS * IN_SAMPLE_RATE, min_len)  # Default to slice in 30 seconds. You can adjust it.
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
        if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int) & (IN_SAMPLE_RATE == OUT_SAMPLE_RATE):
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


def valid_vad_frame_count(input_samples):
    model_samples = int(round(input_samples * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
    if fold_active:
        frames = 0
        while model_samples > 0:
            window_samples = min(model_samples, FOLD_WINDOW_LENGTH)
            if window_samples >= FBANK_WINDOW_LENGTH:
                frames += 1 + (window_samples - FBANK_WINDOW_LENGTH) // OUTPUT_FRAME_LENGTH
            model_samples -= FOLD_WINDOW_LENGTH
        return frames
    if model_samples < FBANK_WINDOW_LENGTH:
        return 0
    return 1 + (model_samples - FBANK_WINDOW_LENGTH) // OUTPUT_FRAME_LENGTH


def vad_frame_times(segment_start, input_samples):
    segment_start_seconds = segment_start / IN_SAMPLE_RATE
    model_samples = int(round(input_samples * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
    if not fold_active:
        frame_count = valid_vad_frame_count(input_samples)
        return segment_start_seconds + np.arange(frame_count) * OUTPUT_FRAME_SHIFT_SECONDS

    frame_times = []
    window_start = 0
    while window_start < model_samples:
        window_samples = min(FOLD_WINDOW_LENGTH, model_samples - window_start)
        if window_samples >= FBANK_WINDOW_LENGTH:
            frame_count = 1 + (
                window_samples - FBANK_WINDOW_LENGTH) // OUTPUT_FRAME_LENGTH
            frame_times.extend(
                segment_start_seconds
                + window_start / MODEL_SAMPLE_RATE
                + np.arange(frame_count) * OUTPUT_FRAME_SHIFT_SECONDS
            )
        window_start += FOLD_WINDOW_LENGTH
    return np.asarray(frame_times, dtype=np.float64)


def probabilities_to_silence(probabilities, speaking_score, silence_score,
                             look_ahead_frames):
    silence = True
    states = []
    full_look_ahead_end = max(0, len(probabilities) - look_ahead_frames)
    for index in range(full_look_ahead_end):
        probability = probabilities[index]
        future = probabilities[index:index + look_ahead_frames]
        if silence:
            silence = not (
                probability >= speaking_score
                and np.mean(future >= speaking_score) >= speaking_score
            )
        elif probability <= silence_score:
            silence = np.mean(future <= silence_score) > silence_score
        else:
            silence = False
        states.append(silence)

    for probability in probabilities[full_look_ahead_end:]:
        if silence:
            silence = probability < speaking_score
        else:
            silence = probability <= silence_score
        states.append(silence)
    return states


def vad_to_timestamps(silence_states, frame_duration, frame_times=None):
    if frame_times is None:
        frame_times = np.arange(len(silence_states), dtype=np.float64) * frame_duration
    if len(frame_times) != len(silence_states):
        raise ValueError(
            f"Expected one frame time per VAD state, got "
            f"{len(frame_times)} times and {len(silence_states)} states.")
    timestamps = []
    start = None
    for index, silence in enumerate(silence_states):
        if silence and start is not None:
            timestamps.append((start, frame_times[index] + frame_duration))
            start = None
        elif not silence and start is None:
            start = frame_times[index]
    if start is not None:
        timestamps.append((start, frame_times[-1] + frame_duration))
    return timestamps


def process_timestamps(timestamps, fusion_threshold, min_duration):
    filtered = [
        (start, end)
        for start, end in timestamps
        if end - start >= min_duration
    ]
    fused = []
    for start, end in filtered:
        if fused and start - fused[-1][1] <= fusion_threshold:
            fused[-1] = (fused[-1][0], end)
        else:
            fused.append((start, end))
    return fused


def format_time(seconds):
    total_milliseconds = round(float(seconds) * 1000)
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def save_vad_timestamps(timestamps):
    print("\nVAD Timestamps in Seconds:")
    with save_timestamps_second.open("w", encoding="utf-8") as stream:
        for start, end in timestamps:
            line = f"{format_time(start)} --> {format_time(end)}"
            stream.write(line + "\n")
            print(line)

    print("\nVAD Timestamps in Indices:")
    with save_timestamps_indices.open("w", encoding="utf-8") as stream:
        for start, end in timestamps:
            line = (
                f"{round(start * IN_SAMPLE_RATE)} --> "
                f"{round(end * IN_SAMPLE_RATE)}")
            stream.write(line + "\n")
            print(line)


near_end_audio, _, _ = align_audio(near_end_audio, min_len)
far_end_audio, aligned_len, stride_step = align_audio(far_end_audio, min_len)


min_len = int(min_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
inv_audio_len = float(100.0 / min_len)


output_audio_length = shape_value_out if isinstance(shape_value_out, int) else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
input_buffer_0 = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), input_dtype_np)
input_buffer_1 = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), input_dtype_np)
output_buffer = _ort_zeros((1, 1, output_audio_length), output_dtype_np)
if OUTPUT_VAD_RESULT:
    vad_shape = out_name_A[1].shape
    if len(vad_shape) != 1:
        raise ValueError(f"Expected rank-1 vad_results, got shape {vad_shape}.")
    vad_frame_count = (
        vad_shape[0]
        if isinstance(vad_shape[0], int)
        else valid_vad_frame_count(INPUT_AUDIO_LENGTH)
    )
    vad_output_buffer = _ort_zeros((vad_frame_count,), vad_output_dtype_np)
binding_A = ort_session_A.io_binding()
binding_A.bind_ortvalue_input(in_name_A0, input_buffer_0)
binding_A.bind_ortvalue_input(in_name_A1, input_buffer_1)
binding_A.bind_ortvalue_output(out_name_A0, output_buffer)
if OUTPUT_VAD_RESULT:
    binding_A.bind_ortvalue_output(out_name_A1, vad_output_buffer)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _near_end_audio, _far_end_audio):
    _update_ortvalue(input_buffer_0, _near_end_audio[:, :, _slice_start: _slice_end])
    _update_ortvalue(input_buffer_1, _far_end_audio[:, :, _slice_start: _slice_end])
    _run_iobinding(ort_session_A, binding_A)
    vad_results = (
        np.array(vad_output_buffer.numpy(), copy=True)
        if OUTPUT_VAD_RESULT else None)
    return _slice_start * _inv_audio_len, np.array(output_buffer.numpy(), copy=True), vad_results


# Start to run DFSMN_AEC
print("\nRunning the DFSMN_AEC by ONNX Runtime.")
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
denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:min_len]
if OUTPUT_VAD_RESULT:
    vad_probabilities = []
    vad_times = []
    for segment_index, result in enumerate(results):
        segment_start = segment_index * stride_step
        valid_input_samples = max(
            0, min(INPUT_AUDIO_LENGTH, input_audio_len - segment_start))
        valid_frames = valid_vad_frame_count(valid_input_samples)
        vad_probabilities.extend(result[2][:valid_frames])
        segment_frame_times = vad_frame_times(segment_start, valid_input_samples)
        if len(segment_frame_times) != valid_frames:
            raise RuntimeError(
                f"VAD frame timing mismatch: {len(segment_frame_times)} times "
                f"for {valid_frames} probabilities.")
        vad_times.extend(segment_frame_times)
    vad_probabilities = np.asarray(vad_probabilities, dtype=np.float32)
    vad_times = np.asarray(vad_times, dtype=np.float64)
    look_ahead_frames = max(1, int(LOOK_AHEAD / OUTPUT_FRAME_SHIFT_SECONDS))
    silence_states = probabilities_to_silence(
        vad_probabilities, SPEAKING_SCORE, SILENCE_SCORE, look_ahead_frames)
    timestamps = process_timestamps(
        vad_to_timestamps(
            silence_states, OUTPUT_FRAME_SHIFT_SECONDS, vad_times),
        FUSION_THRESHOLD,
        MIN_SPEECH_DURATION,
    )
end_time = time.time()
print(f"Complete: 100.00%")

# Save the denoised wav.
if output_dtype_np == np.float16:
    denoised_wav = denoised_wav.astype(np.float32)
sf.write(save_aec_output, denoised_wav, OUT_SAMPLE_RATE, subtype='PCM_16' if output_dtype_np == np.int16 else 'FLOAT')
if OUTPUT_VAD_RESULT:
    save_vad_timestamps(timestamps)
elapsed = end_time - start_time
audio_duration = min_len / OUT_SAMPLE_RATE if OUT_SAMPLE_RATE > 0 else 0.0
rtf = elapsed / audio_duration if audio_duration > 0.0 else float('inf')
vad_summary = (
    f"\n\nVAD Frames: {len(vad_probabilities)}."
    if OUTPUT_VAD_RESULT else "")
print(f"\nAEC Process Complete.\n\nSaving to: {save_aec_output}.{vad_summary}\n\nReal-Time Factor (RTF): {rtf:.4f}")
