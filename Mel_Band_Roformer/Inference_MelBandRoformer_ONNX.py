import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment

onnx_model_A = "/home/DakeQQ/Downloads/MelBandRoformer_Optimized/MelBandRoformer.ort"   # The exported onnx model path.
test_noisy_audio = "./test.wav"                                                         # The noisy audio path.
save_denoised_audio = "./test_denoised.wav"                                             # The output denoised audio path.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
MAX_THREADS = 4                         # Number of parallel threads for audio denoising.
SAMPLE_RATE = 44100                     # The Mel-Band-Roformer parameter, do not edit the value.
DEVICE_ID = 0

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',
            'precision': 'ACCURACY',
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False
        }
    ]
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'enable_cuda_graph': '0',  # Set to '0' to avoid potential errors when enabled.
            'use_tf32': '1'            
        }
    ]
else:
    provider_options = None

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# Load the input audio
print(f"\nTest Input Audio: {test_noisy_audio}")
audio = AudioSegment.from_file(test_noisy_audio).set_frame_rate(SAMPLE_RATE)
audio_channels = audio.channels
audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
if audio_channels == 1:
    audio = audio.reshape(1, 1, -1)
    audio = np.concatenate((audio, audio), axis=1)
else:
    audio = audio.reshape(1, -1, 2).transpose(0, 2, 1)
audio_len = audio.shape[-1]
inv_audio_len = float(100.0 / audio_len)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_in_channel = ort_session_A._inputs_meta[0].shape[1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(SAMPLE_RATE, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int):
        stride_step = shape_value_out
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, shape_value_in_channel, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, shape_value_in_channel, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Start to run Mel-Band-Roformer
print("\nRunning the MelBandRoformer by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel denoised the audio.
    futures = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, audio))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.3f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
denoised_wav = np.concatenate(saved, axis=-1)[0, :, :audio_len].transpose()
end_time = time.time()
print(f"Complete: 100.00%")

# Save the denoised wav.
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
