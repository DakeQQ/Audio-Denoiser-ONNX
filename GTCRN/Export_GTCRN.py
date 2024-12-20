import gc
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
import torch
from pydub import AudioSegment

from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.
from modeling_modified.gtcrn import GTCRN


model_path = "/home/DakeQQ/Downloads/gtcrn-main"                          # The GTCRN download path.
onnx_model_A = "/home/DakeQQ/Downloads/GTCRN_ONNX/GTCRN.onnx"             # The exported onnx model path.
test_noisy_audio =  model_path + "/test_wavs/mix.wav"                     # The noisy audio path.
save_denoised_audio = model_path + "/test_wavs/denoised.wav"              # The output denoised audio path.

ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
USE_PCM_INT16 = False                   # Enable it, if the audio input is PCM wav data with dtype int16 (short).
MAX_SIGNAL_LENGTH = 1024 if DYNAMIC_AXES else 384  # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
INPUT_AUDIO_LENGTH = 15360               # Set for static axis export: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.
WINDOW_TYPE = 'kaiser'                  # Type of window function used in the STFT
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 512                              # Number of FFT components for the STFT process
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 16000                     # The GTCRN parameter, do not edit the value.
MAX_THREADS = 8                         # Number of parallel threads for test audio denoising.


class GTCRN_CUSTOM(torch.nn.Module):
    def __init__(self, gtcrn, stft_model, istft_model, use_pcm_int16):
        super(GTCRN_CUSTOM, self).__init__()
        self.gtcrn = gtcrn
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.use_pcm_int16 = use_pcm_int16
        self.inv_int16 = 1.0 / 32768.0

    def forward(self, audio):
        if self.use_pcm_int16:
            audio = self.inv_int16 * audio.float()
        real_part, imag_part = self.stft_model(audio, 'constant')
        magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part)
        magnitude, s_real, s_imag = self.gtcrn.forward(magnitude, real_part, imag_part)
        audio = self.istft_model(magnitude, s_real, s_imag)
        if self.use_pcm_int16:
            audio = (audio * 32768).to(torch.int16)
        return audio


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    gtcrn = GTCRN().eval()
    ckpt = torch.load(model_path + "/checkpoints/model_trained_on_dns3.tar", map_location='cpu')
    gtcrn.load_state_dict(ckpt['model'])
    gtcrn = GTCRN_CUSTOM(gtcrn, custom_stft, custom_istft, USE_PCM_INT16)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16 if USE_PCM_INT16 else torch.float32)
    torch.onnx.export(
        gtcrn,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del gtcrn
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
print('\nExport done!\n\nStart to run GTCRN by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 1       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 1       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


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
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(30 * SAMPLE_RATE, audio_len)  # Default to slice in 30 seconds. You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if shape_value_in != shape_value_out:
        stride_step = shape_value_out
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:]
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    white_noise = (np.sqrt(np.mean(audio * audio)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio, _ort_session_A, _in_name_A0, _out_name_A0):
    return _slice_start * _inv_audio_len, _ort_session_A.run([_out_name_A0], {_in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Start to run GTCRN
print("\nRunning the GTCRN by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel denoised the audio.
    futures = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, audio, ort_session_A, in_name_A0, out_name_A0))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.3f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
denoised_wav = (np.concatenate(saved, axis=-1)[0, 0, :audio_len]).astype(np.float32)
end_time = time.time()
print(f"Complete: 100.000%")


# Save the denoised wav.
if "int16" in model_type:
    denoised_wav /= 32768.0
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
