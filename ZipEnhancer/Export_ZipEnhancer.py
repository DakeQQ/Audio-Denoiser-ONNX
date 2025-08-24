import gc
import shutil
import time
import site
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
import torch
from modelscope.models.base import Model
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

model_path = "/home/DakeQQ/Downloads/speech_zipenhancer_ans_multiloss_16k_base"         # The ZipEnhancer download path.
onnx_model_A = "/home/DakeQQ/Downloads/ZipEnhancer_ONNX/ZipEnhancer.onnx"               # The exported onnx model path.
test_noisy_audio = model_path + "/examples/speech_with_noise1.wav"                      # The noisy audio path.
save_denoised_audio = "./speech_with_noise1_denoised.wav"                               # The output denoised audio path.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
KEEP_ORIGINAL_SAMPLE_RATE = False       # If False, the model outputs audio at 16kHz; otherwise, it uses the original sample rate.
SAMPLE_RATE = 16000                     # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH = 16000              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4800 and less than 48000. Higher values yield better quality but time consume. It is better to set an integer multiple of the NFFT value.
MAX_SIGNAL_LENGTH = 1024 if DYNAMIC_AXES else (INPUT_AUDIO_LENGTH // 50 + 1)  # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hamming'                 # Type of window function used in the STFT
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 512                              # Number of FFT components for the STFT process
WINDOW_LENGTH = 400                     # Length of windowing, edit it carefully.
HOP_LENGTH = 100                        # Number of samples between successive frames in the STFT
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.


SAMPLE_RATE_SCALE = float(16000.0 / SAMPLE_RATE)


site_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/zipenhancer.py", site_package_path + "/modelscope/models/audio/ans/zipenhancer.py")
shutil.copyfile("./modeling_modified/generator.py", site_package_path + "/modelscope/models/audio/ans/zipenhancer_layers/generator.py")
shutil.copyfile("./modeling_modified/scaling.py", site_package_path + "/modelscope/models/audio/ans/zipenhancer_layers/scaling.py")
shutil.copyfile("./modeling_modified/zipenhancer_layer.py", site_package_path + "/modelscope/models/audio/ans/zipenhancer_layers/zipenhancer_layer.py")
shutil.copyfile("./modeling_modified/zipformer.py", site_package_path + "/modelscope/models/audio/ans/zipenhancer_layers/zipformer.py")


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class ZipEnhancer(torch.nn.Module):
    def __init__(self, zip_enhancer, stft_model, istft_model, sample_rate):
        super(ZipEnhancer, self).__init__()
        self.zip_enhancer = zip_enhancer
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.compress_factor = float(0.3)
        self.compress_factor_inv = float(1.0 / self.compress_factor)
        self.compress_factor_sqrt = float(self.compress_factor * 0.5)
        self.sample_rate = sample_rate

    def forward(self, audio):
        audio = audio.float()
        if SAMPLE_RATE_SCALE < 1.0:
            norm_factor = torch.sqrt(torch.mean(audio * audio, dim=-1, keepdim=True) + 1e-6)
            audio /= norm_factor
            if self.sample_rate != 16000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
        else:
            if self.sample_rate != 16000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
            norm_factor = torch.sqrt(torch.mean(audio * audio, dim=-1, keepdim=True) + 1e-6)
            audio /= norm_factor
        real_part, imag_part = self.stft_model(audio, 'constant')
        magnitude = torch.pow(real_part * real_part + imag_part * imag_part, self.compress_factor_sqrt)
        phase = torch.atan2(imag_part, real_part)
        magnitude, phase = self.zip_enhancer(magnitude, phase)
        audio = self.istft_model(torch.pow(magnitude, self.compress_factor_inv), phase)
        if SAMPLE_RATE_SCALE < 1.0:
            audio *= norm_factor
            if KEEP_ORIGINAL_SAMPLE_RATE and self.sample_rate != 16000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=1.0 / SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
        else:
            if KEEP_ORIGINAL_SAMPLE_RATE and self.sample_rate != 16000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=1.0 / SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
            audio *= norm_factor
        return (audio.clamp(min=-32768.0, max=32767.0)).to(torch.int16)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_istft = STFT_Process(model_type='istft_A', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    zip_enhancer = Model.from_pretrained(model_name_or_path=model_path, device='cpu').model.eval()
    zip_enhancer = ZipEnhancer(zip_enhancer, custom_stft, custom_istft, SAMPLE_RATE)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        zip_enhancer,
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
    del zip_enhancer
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
print('\nExport done!\n\nStart to run ZipEnhancer by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
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
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = max(96000, audio_len)  # 36000 for (8 threads + 32GB RAM), 64000 for (4 threads + 32GB RAM), Max <= 99999 for model limit.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
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
inv_audio_len = float(100.0 / audio_len)


if SAMPLE_RATE != 16000 and not KEEP_ORIGINAL_SAMPLE_RATE:
    SAMPLE_RATE = 16000
    audio_len = int(audio_len * SAMPLE_RATE_SCALE)



def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Start to run ZipEnhancer
print("\nRunning the ZipEnhancer by ONNX Runtime.")
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
denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:audio_len]
end_time = time.time()
print(f"Complete: 100.00%")


# Save the denoised wav.
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
