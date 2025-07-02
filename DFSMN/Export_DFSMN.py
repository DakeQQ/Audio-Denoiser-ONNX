import gc
import time
import numpy as np
import onnxruntime
import torch
import torchaudio
import soundfile as sf
from pydub import AudioSegment
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from concurrent.futures import ThreadPoolExecutor
from STFT_Process import STFT_Process                                                      # The custom STFT/ISTFT can be exported in ONNX format.


model_path = "/home/DakeQQ/Downloads/speech_dfsmn_ans_psm_48k_causal"                      # The DFSMN download path.
onnx_model_A = "/home/DakeQQ/Downloads/DFSMN_ONNX/DFSMN.onnx"                              # The exported onnx model path.
test_noisy_audio = model_path + "/examples/speech_with_noise_48k.wav"                      # The noisy audio path.
save_denoised_audio = "./speech_with_noise_48k_denoised.wav"          # The output denoised audio path.

ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
MAX_SIGNAL_LENGTH = 1024 if DYNAMIC_AXES else 128  # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
INPUT_AUDIO_LENGTH = 96000              # The maximum input audio length.
WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
N_MELS = 120                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT_STFT = 1920                        # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 1920                    # Length of windowing, edit it carefully.
HOP_LENGTH = 960                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 48000                     # The DFSMN parameter, do not edit the value.
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class DFSMN(torch.nn.Module):
    def __init__(self, dfsmn, stft_model, istft_model, nfft_stft, n_mels, sample_rate):
        super(DFSMN, self).__init__()
        self.dfsmn = dfsmn
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft

    def forward(self, audio):
        audio = audio.float()       # Don't divide by 32768.0
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2).clamp(min=1e-5).log()
        mask = self.dfsmn(mel_features).transpose(1, 2)
        real_part *= mask
        imag_part *= mask
        magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part)
        audio = self.istft_model(magnitude, real_part, imag_part)
        return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH,hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    dfsmn = pipeline(
        Tasks.acoustic_noise_suppression,
        model=model_path,
        device='cpu'
    ).model
    dfsmn = DFSMN(dfsmn, custom_stft, custom_istft, NFFT_STFT, N_MELS, SAMPLE_RATE)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        dfsmn,
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
    del dfsmn
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
print('\nExport done!\n\nStart to run DFSMN by ONNX Runtime.\n\nNow, loading the model...')


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
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


# Load the input audio
print(f"\nTest Input Audio: {test_noisy_audio}")
audio = np.array(AudioSegment.from_file(test_noisy_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
inv_audio_len = float(100.0 / audio_len)
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(96000, audio_len)  # You can adjust it.
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
    white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Start to run DFSMN
print("\nRunning the DFSMN by ONNX Runtime.")
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
denoised_wav = np.concatenate(saved, axis=-1)[0, 0, :audio_len]
end_time = time.time()
print(f"Complete: 100.00%")


# Save the denoised wav.
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
