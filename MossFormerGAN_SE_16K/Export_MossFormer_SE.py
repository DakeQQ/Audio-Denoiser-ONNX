import gc
import shutil
import time
import site
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
import torch
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


onnx_model_A = "/home/DakeQQ/Downloads/MossFormer_ONNX/MossFormer_SE.onnx"    # The exported onnx model path.
test_noisy_audio = "./examples/speech_with_noise1.wav"                        # The noisy audio path.
save_denoised_audio = "./examples/speech_with_noise1_denoised.wav"            # The output denoised audio path.


DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 16000              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 8000 and less than 96000. Higher values yield better quality but time consume. It is better to set an integer multiple of the NFFT value.
MAX_SIGNAL_LENGTH = 1024 if DYNAMIC_AXES else (INPUT_AUDIO_LENGTH // 100 + 1)  # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hamming'                 # Type of window function used in the STFT
N_MELS = 60                             # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 400                              # Number of FFT components for the STFT process
WINDOW_LENGTH = 400                     # Length of windowing, edit it carefully.
HOP_LENGTH = 100                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 16000                     # The MossFormer_SE parameter, do not edit the value.
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.

site_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/generator.py", site_package_path + "/clearvoice/models/mossformer_gan_se/generator.py")
shutil.copyfile("./modeling_modified/mossformer.py", site_package_path + "/clearvoice/models/mossformer_gan_se/mossformer.py")
shutil.copyfile("./modeling_modified/se_layer.py", site_package_path + "/clearvoice/models/mossformer_gan_se/se_layer.py")
shutil.copyfile("./modeling_modified/fsmn.py", site_package_path + "/clearvoice/models/mossformer_gan_se/fsmn.py")


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class MOSSFORMER_SE(torch.nn.Module):
    def __init__(self, mossformer_se, stft_model, istft_model):
        super(MOSSFORMER_SE, self).__init__()
        self.mossformer_se = mossformer_se
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.compress_factor = float(0.3)
        self.compress_factor_inv = float(0.5 / self.compress_factor) - 0.5
        self.compress_factor_sqrt = float(self.compress_factor * 0.5)
        self.inv_int16 = float(1.0 / 32768.0)

    def forward(self, audio):
        # Initial preprocessing
        audio = audio * self.inv_int16
        norm_factor = torch.sqrt(torch.mean(audio * audio, dim=-1, keepdim=True) + 1e-6)

        # STFT and magnitude/phase extraction
        real_part, imag_part = self.stft_model(audio / norm_factor, 'constant')
        power = real_part * real_part + imag_part * imag_part
        magnitude_compress = torch.pow(power, self.compress_factor_sqrt)
        inv_magnitude = torch.pow(power, -0.5)
        cos_phase = real_part * inv_magnitude
        sin_phase = imag_part * inv_magnitude

        # Prepare input for model
        x_in = torch.cat([magnitude_compress, magnitude_compress * cos_phase, magnitude_compress * sin_phase], dim=0).unsqueeze(0)

        # MossFormer processing
        x = self.mossformer_se.dense_encoder(x_in.transpose(-1, -2))
        for ii in range(self.mossformer_se.n_layers):
            x = self.mossformer_se.blocks[ii](x)

        # Generate outputs
        mask = self.mossformer_se.mask_decoder(x).transpose(1, 2)
        complex_out = self.mossformer_se.complex_decoder(x).transpose(-1, -2)

        # Apply mask and combine with complex output
        out_mag = mask * magnitude_compress
        mag_real = out_mag * cos_phase
        mag_imag = out_mag * sin_phase

        final_real = mag_real + complex_out[:, 0]
        final_imag = mag_imag + complex_out[:, 1]

        # Decompress and reconstruct
        factor = torch.pow(final_real * final_real + final_imag * final_imag, self.compress_factor_inv)

        # ISTFT and final processing
        audio = self.istft_model(final_real * factor, final_imag * factor) * norm_factor

        return (audio * 32768.0).clamp(min=-32768.0, max=32767.0).to(torch.int16)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    from clearvoice import ClearVoice
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])
    mossformer = myClearVoice.models[0].model
    mossformer = MOSSFORMER_SE(mossformer.eval().float(), custom_stft, custom_istft)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        mossformer,
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
    del mossformer
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
print('\nExport done!\n\nStart to run MossFormer by ONNX Runtime.\n\nNow, loading the model...')


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


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
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
    INPUT_AUDIO_LENGTH = min(32000, audio_len)
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


# Start to run MossFormer_SE
print("\nRunning the MossFormer_SE by ONNX Runtime.")
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
