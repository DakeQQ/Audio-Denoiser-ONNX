import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment

onnx_model_A = "/home/DakeQQ/Downloads/ZipEnhancer_Optimized/ZipEnhancer.ort"                                                         # The exported onnx model path.
test_noisy_audio = "/home/DakeQQ/Downloads/speech_zipenhancer_ans_multiloss_16k_base/examples/speech_with_noise1.wav"                 # The noisy audio path.
save_denoised_audio = "/home/DakeQQ/Downloads/speech_zipenhancer_ans_multiloss_16k_base/examples/speech_with_noise1_denoised.wav"     # The output denoised audio path.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
MAX_THREADS = 4                         # Number of parallel threads for audio denoising.
SAMPLE_RATE = 16000                     # The ZipEnhancer parameter, do not edit the value.


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
model_type = ort_session_A._inputs_meta[0].type
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


# Load the input audio
print(f"\nTest Input Audio: {test_noisy_audio}")
audio = np.array(AudioSegment.from_file(test_noisy_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
audio_len = len(audio)
inv_audio_len = float(100.0 / audio_len)
if "int16" not in model_type:
    audio = audio.astype(np.float32) / 32768.0
    if "float16" in model_type:
        audio = audio.astype(np.float16)
audio = audio.reshape(1, 1, -1)
shape_value = ort_session_A._inputs_meta[0].shape[2]
if isinstance(shape_value, str):
    INPUT_AUDIO_LENGTH = min(64000, audio_len)  # 36000 for (8 threads + 32GB RAM), 64000 for (4 threads + 32GB RAM), Max <= 99999 for model limit.
else:
    INPUT_AUDIO_LENGTH = shape_value
if audio_len > INPUT_AUDIO_LENGTH:
    final_slice = audio[:, :, audio_len // INPUT_AUDIO_LENGTH * INPUT_AUDIO_LENGTH:]
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - final_slice.shape[-1]))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    white_noise = (np.sqrt(np.mean(audio * audio)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


def process_segment(_inv_audio_len, _slice_start, step, _audio, _ort_session_A, _in_name_A0, _out_name_A0):
    return _slice_start * _inv_audio_len, _ort_session_A.run([_out_name_A0], {_in_name_A0: _audio[:, :, _slice_start: _slice_start + step]})[0]


# Start to run ZipEnhancer
print("\nRunning the ZipEnhancer by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel denoised the audio.
    futures = []
    for slice_start in range(0, aligned_len, INPUT_AUDIO_LENGTH):
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, INPUT_AUDIO_LENGTH, audio, ort_session_A, in_name_A0, out_name_A0))
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.2f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
denoised_wav = (np.concatenate(saved, axis=-1)[0, 0, :audio_len]).astype(np.float32)
end_time = time.time()
print(f"Complete: 100.00%")


# Save the denoised wav.
if "int16" in model_type:
    denoised_wav /= 32768.0
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='FLAC')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
