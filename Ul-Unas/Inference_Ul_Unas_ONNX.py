import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment


script_dir = os.path.dirname(os.path.abspath(__file__))

model_path          = r"/home/DakeQQ/Downloads/Ul_Unas_Optimized/Ul_Unas.onnx"  # The optimized onnx model path.
test_noisy_audio    = r"./example/0174.wav"                                     # The noisy audio path.
save_denoised_audio = r"./denoised.wav"                                         # The output denoised audio path.


ORT_Accelerate_Providers = []           # If you have accelerated devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider']
                                        # else keep empty.
MAX_THREADS = 4                         # Number of parallel threads for audio denoising.
DEVICE_ID = 0                           # The GPU id, default to 0.
IN_SAMPLE_RATE  = 16000                 # UL-UNAS is designed for 16kHz only.
OUT_SAMPLE_RATE = 16000                 # UL-UNAS is designed for 16kHz only.
NORMALIZE_AUDIO = False                 # Normalize the input audio to a target RMS level (e.g., 8192) before processing. It can help improve the performance of the model, especially for low-volume audio. Set it to True if you want to enable it.


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': True,
        }
    ]
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'DEFAULT',          # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '0',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
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
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu',                        # [any, npu, gpu]
        }
    ]
else:
    # Please config by yourself for others providers.
    provider_options = None


def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-12))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    return audio


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


def main():
    # ONNX Runtime settings
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
    session_opts.inter_op_num_threads = 1       # Run different nodes with num_threads. Set 0 for auto.
    session_opts.intra_op_num_threads = 1       # Under the node, execute the operators with num_threads. Set 0 for auto.
    session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

    if ORT_Accelerate_Providers:
        ort_session_A = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_opts,
            providers=ORT_Accelerate_Providers,
            provider_options=provider_options,
        )
    else:
        ort_session_A = onnxruntime.InferenceSession(model_path, sess_options=session_opts)

    print(f"\nModel Path: {model_path}")
    print(f"\nUsable Providers: {ort_session_A.get_providers()}")
    in_name_A = ort_session_A.get_inputs()
    out_name_A = ort_session_A.get_outputs()
    in_name_A0 = in_name_A[0].name
    out_name_A0 = out_name_A[0].name

    # Load the input audio
    print(f"\nTest Input Audio: {test_noisy_audio}")
    audio = np.array(
        AudioSegment.from_file(test_noisy_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(),
        dtype=np.int16,
    )
    if NORMALIZE_AUDIO:
        audio = normalise_audio(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)

    shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
    shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
    if isinstance(shape_value_in, str):
        input_audio_length = min(30 * IN_SAMPLE_RATE, audio_len)  # Default to slice in 30 seconds. You can adjust it.
    else:
        input_audio_length = shape_value_in
    stride_step = input_audio_length

    if audio_len > input_audio_length:
        if (shape_value_in != shape_value_out) and isinstance(shape_value_in, int) and isinstance(shape_value_out, int) and (OUT_SAMPLE_RATE == IN_SAMPLE_RATE):
            stride_step = shape_value_out
        num_windows = int(np.ceil((audio_len - input_audio_length) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + input_audio_length
        audio = pad_audio_tail_with_context(audio, total_length_needed)
    elif audio_len < input_audio_length:
        audio = pad_audio_tail_with_context(audio, input_audio_length)

    aligned_len = audio.shape[-1]
    inv_audio_len = float(100.0 / aligned_len)
    output_audio_len = int(audio_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)

    def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
        return _slice_start * _inv_audio_len, ort_session_A.run(
            [out_name_A0],
            {
                in_name_A0: _audio[:, :, _slice_start:_slice_end],
            },
        )[0]

    # Start to run Ul-Unas
    print("\nRunning the Ul-Unas by ONNX Runtime.")
    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel denoised the audio.
        futures = []
        slice_start = 0
        slice_end = input_audio_length
        while slice_end <= aligned_len:
            futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, audio))
            slice_start += stride_step
            slice_end = slice_start + input_audio_length
        for future in futures:
            results.append(future.result())
            print(f"Complete: {results[-1][0]:.3f}%")
    results.sort(key=lambda x: x[0])
    saved = [result[1] for result in results]
    denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:output_audio_len]
    end_time = time.time()
    elapsed = end_time - start_time
    audio_duration = output_audio_len / OUT_SAMPLE_RATE if OUT_SAMPLE_RATE > 0 else 0.0
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
    print("Complete: 100.00%")

    # Save the denoised wav.
    sf.write(save_denoised_audio, denoised_wav, OUT_SAMPLE_RATE, format='WAVEX')
    print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {elapsed:.3f} Seconds\nAudio Duration: {audio_duration:.3f} Seconds\nRTF: {rtf:.4f}")


if __name__ == "__main__":
    main()
