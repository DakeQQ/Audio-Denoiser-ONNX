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
model_path          = str(parent_path / "UL_UNAS_Optimized" / "UL_UNAS.onnx") # The optimized onnx model path.
test_noisy_audio    = model_audio_path("ul_unas")                            # The noisy audio path.
save_denoised_audio = str(parent_path / "denoised.wav")                      # The output denoised audio path.


def _resolve_onnx_model_path(default_model_path: str) -> str:
    if len(sys.argv) <= 1:
        return default_model_path
    candidate = Path(sys.argv[1]).expanduser()
    if candidate.is_dir():
        candidate = candidate / Path(default_model_path).name
    return str(candidate)


model_path = _resolve_onnx_model_path(model_path)


ORT_Accelerate_Providers = []               # If you have accelerated devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider']
                                            # else keep empty.
ORT_LOG                     = False         # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                    = False         # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS                 = 0             # Number of ONNX Runtime/OpenVINO worker threads. Set 0 for auto.
DEVICE_ID                   = 0             # The GPU id, default to 0.
NORMALIZE_AUDIO             = False         # Set True to RMS-normalize input audio before inference.
NORMALIZE_TARGET_RMS        = 4096.0        # Target RMS when NORMALIZE_AUDIO is True.


# ONNX Runtime settings
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                              # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                           # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,                     # Enable it carefully
            'disable_dynamic_shapes': False,
        }
    ]
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,          # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',        # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',            # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                                # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '0',                             # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '0',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                          # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',       # [high_performance, default, minimum_power]
            'device_filter': 'gpu',                             # [any, npu, gpu]
            'disable_metacommands': 'false',
            'enable_graph_capture': 'false',
            'enable_graph_serialization': 'false',
        }
    ]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
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


def pad_audio_tail_with_context(audio: np.ndarray, target_length: int) -> np.ndarray:
    current_length = audio.shape[-1]
    if current_length >= target_length:
        return audio

    pad_amount = target_length - current_length
    if current_length == 0:
        padding = np.zeros((*audio.shape[:-1], pad_amount), dtype=audio.dtype)
    elif current_length == 1:
        padding = np.repeat(audio[..., -1:], pad_amount, axis=-1)
    else:
        padding = np.pad(audio, ((0, 0), (0, 0), (0, pad_amount)), mode='reflect')[..., current_length:]

    return np.concatenate((audio, padding.astype(audio.dtype, copy=False)), axis=-1)


def align_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def pad_audio_tail(audio: np.ndarray, target_length: int, fold_active: bool) -> np.ndarray:
    # Batch-fold path: pad with zeros (silence passes through STFT->net->ISTFT and is trimmed off).
    # Non-fold path: keep the original reflect-context padding.
    if fold_active:
        pad_amount = target_length - audio.shape[-1]
        if pad_amount <= 0:
            return audio
        padding = np.zeros((*audio.shape[:-1], pad_amount), dtype=audio.dtype)
        return np.concatenate((audio, padding), axis=-1)
    return pad_audio_tail_with_context(audio, target_length)


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
disabled_opts = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16 else None
)
_packed = {
    'sess_options': session_opts_ort,
    'providers': ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    'provider_options': provider_options,
    'disabled_optimizers': disabled_opts,
}


def main():
    global IN_SAMPLE_RATE, OUT_SAMPLE_RATE, MODEL_SAMPLE_RATE, INPUT_TO_OUTPUT_SCALE
    global BATCH_WINDOW_SECONDS, HOP_LENGTH, FOLD_WINDOW_LENGTH, FOLD_INPUT_LENGTH
    global BATCH_FOLD_INFERENCE, MAX_DYNAMIC_AUDIO_SECONDS, NORMALIZE_AUDIO, NORMALIZE_TARGET_RMS
    global INPUT_CHANNELS, OUTPUT_CHANNELS, N_CHANNELS, NUM_AUDIO_INPUTS
    global PAD_HEAD, ENC_STRIDE, OUTPUT_SOURCES, ORIGINAL_SAMPLE_RATE, SUPER_SAMPLE_RATE, SCALE_FACTOR

    ort_session_A = _make_session(model_path)
    _model_metadata = load_runtime_metadata(model_path, _make_session)
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

    print(f"\nModel Path: {model_path}")
    print(f"\nUsable Providers: {ort_session_A.get_providers()}")
    in_name_A = ort_session_A.get_inputs()
    out_name_A = ort_session_A.get_outputs()
    in_name_A0 = in_name_A[0].name
    out_name_A0 = out_name_A[0].name
    input_dtype_np = _numpy_dtype_from_meta(in_name_A[0])
    output_dtype_np = _numpy_dtype_from_meta(out_name_A[0])

    # Load the input audio
    print(f"\nTest Input Audio: {test_noisy_audio}")
    audio = np.array(
        AudioSegment.from_file(test_noisy_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(),
        dtype=np.int16,
    )
    audio = normalise_audio(audio, input_dtype_np)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)

    shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
    shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
    if isinstance(shape_value_in, str):
        input_audio_length = min(MAX_DYNAMIC_AUDIO_SECONDS * IN_SAMPLE_RATE, audio_len)  # Default dynamic slice length before fold alignment.
        if BATCH_FOLD_INFERENCE and input_audio_length / IN_SAMPLE_RATE > BATCH_WINDOW_SECONDS:
            input_audio_length = align_to_multiple(input_audio_length, FOLD_INPUT_LENGTH)
    else:
        input_audio_length = shape_value_in
    fold_active = BATCH_FOLD_INFERENCE and isinstance(input_audio_length, int) and input_audio_length >= FOLD_INPUT_LENGTH
    if fold_active:
        print(f"Batch Fold Window: {FOLD_WINDOW_LENGTH} model samples; ONNX Input Slice: {input_audio_length} input samples")
    stride_step = input_audio_length

    if audio_len > input_audio_length:
        if (shape_value_in != shape_value_out) and isinstance(shape_value_in, int) and isinstance(shape_value_out, int) and (OUT_SAMPLE_RATE == IN_SAMPLE_RATE):
            stride_step = shape_value_out
        num_windows = int(np.ceil((audio_len - input_audio_length) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + input_audio_length
        audio = pad_audio_tail(audio, total_length_needed, fold_active)
    elif audio_len < input_audio_length:
        audio = pad_audio_tail(audio, input_audio_length, fold_active)

    aligned_len = audio.shape[-1]
    inv_audio_len = float(100.0 / aligned_len)
    output_audio_len = int(audio_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)

    output_audio_length = shape_value_out if isinstance(shape_value_out, int) else int(round(input_audio_length * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
    input_buffer = _ort_zeros((1, 1, input_audio_length), input_dtype_np)
    output_buffer = _ort_zeros((1, 1, output_audio_length), output_dtype_np)
    binding_A = ort_session_A.io_binding()
    binding_A.bind_ortvalue_input(in_name_A0, input_buffer)
    binding_A.bind_ortvalue_output(out_name_A0, output_buffer)

    def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
        _update_ortvalue(input_buffer, _audio[:, :, _slice_start:_slice_end])
        _run_iobinding(ort_session_A, binding_A)
        return _slice_start * _inv_audio_len, np.array(output_buffer.numpy(), copy=True)

    # Start to run UL-UNAS
    print("\nRunning the UL-UNAS by ONNX Runtime.")
    results = []
    start_time = time.time()
    slice_start = 0
    slice_end = input_audio_length
    while slice_end <= aligned_len:
        results.append(process_segment(inv_audio_len, slice_start, slice_end, audio))
        print(f"Complete: {results[-1][0]:.3f}%")
        slice_start += stride_step
        slice_end = slice_start + input_audio_length
    saved = [result[1] for result in results]
    denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:output_audio_len]
    end_time = time.time()
    elapsed = end_time - start_time
    audio_duration = output_audio_len / OUT_SAMPLE_RATE if OUT_SAMPLE_RATE > 0 else 0.0
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
    print("Complete: 100.00%")

    # Save the denoised wav.
    if output_dtype_np == np.float16:
        denoised_wav = denoised_wav.astype(np.float32)
    sf.write(save_denoised_audio, denoised_wav, OUT_SAMPLE_RATE, subtype='PCM_16' if output_dtype_np == np.int16 else 'FLOAT')
    print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nReal-Time Factor (RTF): {rtf:.4f}")


if __name__ == "__main__":
    main()
