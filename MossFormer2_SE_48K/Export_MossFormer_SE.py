import gc
import shutil
import time
import site
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

model_path = "/home/DakeQQ/Downloads/MossFormer2_SE_48K"                              # The MossFormer2_SE_48K download folder.
onnx_model_A = "/home/DakeQQ/Downloads/MossFormer_ONNX/MossFormer2_SE_48K.onnx"       # The exported onnx model path.
test_noisy_audio = "./examples/speech_with_noise1.wav"                                # The noisy audio path.
save_denoised_audio = "./examples/speech_with_noise1_denoised.wav"                    # The output denoised audio path.


DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
KEEP_ORIGINAL_SAMPLE_RATE = True        # If False, the model outputs audio at 48kHz; otherwise, it uses the original sample rate.
SAMPLE_RATE = 48000                     # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH = 48000              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 8000 and less than 96000. Higher values yield better quality but time consume. It is better to set an integer multiple of the NFFT value.
MAX_SIGNAL_LENGTH = 4096 if DYNAMIC_AXES else (INPUT_AUDIO_LENGTH // 100 + 1)  # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hamming'                 # Type of window function used in the STFT
N_MELS = 60                             # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 1920                             # Number of FFT components for the STFT process
WINDOW_LENGTH = 1920                    # Length of windowing, edit it carefully.
HOP_LENGTH = 384                        # Number of samples between successive frames in the STFT
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.

SAMPLE_RATE_SCALE = float(48000.0 / SAMPLE_RATE)

site_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/__init__.py", site_package_path + "/clearvoice/__init__.py")
shutil.copyfile("./modeling_modified/network_wrapper.py", site_package_path + "/clearvoice/network_wrapper.py")
shutil.copyfile("./modeling_modified/mossformer2_block.py", site_package_path + "/clearvoice/models/mossformer2_se/mossformer2_block.py")
shutil.copyfile("./modeling_modified/mossformer2.py", site_package_path + "/clearvoice/models/mossformer2_se/mossformer2.py")
shutil.copyfile("./modeling_modified/fsmn.py", site_package_path + "/clearvoice/models/mossformer2_se/fsmn.py")
from clearvoice import ClearVoice


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class MOSSFORMER_SE(torch.nn.Module):
    def __init__(self, mossformer_se, stft_model, istft_model, nfft_stft, n_mels, sample_rate, max_signal_len):
        super(MOSSFORMER_SE, self).__init__()
        self.mossformer_se = mossformer_se.mossformer
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, 24000, n_mels, 48000, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        t = torch.arange(max_signal_len * 3, dtype=torch.float32)  # Create time steps
        sinu = t.unsqueeze(-1) * self.mossformer_se.pos_enc.inv_freq  # Calculate sine and cosine embeddings
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)  # Concatenate sine and cosine embeddings
        self.emb_pos = (emb * self.mossformer_se.pos_enc.scale).transpose(0, -1)  # Scale the embeddings
        self.emb_pos = self.emb_pos.unsqueeze(0).half()

        self.win_length_for_delta = 5
        self.n = (self.win_length_for_delta - 1) // 2
        denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3
        self.inv_denom = float(1.0 / denom)
        self.kernel = torch.arange(-self.n, self.n + 1, 1, dtype=torch.float32)
    
    def compute_deltas(self, specgram: torch.Tensor, mode: str = "replicate", time_dim: int = 256) -> torch.Tensor:
        specgram = specgram.reshape(-1, 1, time_dim)
        padded_specgram = torch.nn.functional.pad(specgram, (self.n, self.n), mode=mode)
        slices = [padded_specgram[..., i : i + time_dim] for i in range(self.win_length_for_delta)]
        windows = torch.stack(slices, dim=-1)
        output = windows.matmul(self.kernel) * self.inv_denom
        output = output.reshape(1, self.n_mels, time_dim)
        return output

    def forward(self, audio):
        audio = audio.float()
        if SAMPLE_RATE_SCALE < 1.0:
            audio = audio - torch.mean(audio)
            if self.sample_rate != 48000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
        else:
            if self.sample_rate != 48000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
            audio = audio - torch.mean(audio)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).clamp(min=1e-6).log()
        mel_features_len = mel_features.shape[-1].unsqueeze(0)
        mel_features_delta = self.compute_deltas(mel_features, time_dim=mel_features_len)
        mel_features_delta_2 = self.compute_deltas(mel_features_delta, time_dim=mel_features_len)
        mel_features = torch.cat([mel_features, mel_features_delta, mel_features_delta_2], dim=1)
        x = self.mossformer_se.norm(mel_features)
        x = self.mossformer_se.conv1d_encoder(x)
        x = x + self.emb_pos[..., :mel_features_len].float()
        x = self.mossformer_se.mdl(x, mel_features_len)
        x = self.mossformer_se.prelu(x)
        x = self.mossformer_se.conv1d_out(x)
        x = x.view(self.mossformer_se.num_spks, -1, mel_features_len)
        x = self.mossformer_se.output(x) * self.mossformer_se.output_gate(x)  # Element-wise multiplication for gating
        x = self.mossformer_se.conv1_decoder(x)
        x = x.view(1, self.mossformer_se.num_spks, -1, mel_features_len)  # Final reshaping for output
        x = self.mossformer_se.activation(x)  # Apply final activation
        x = x[:, 0]
        real_part *= x
        imag_part *= x
        audio = self.istft_model(real_part, imag_part)
        if KEEP_ORIGINAL_SAMPLE_RATE and self.sample_rate != 48000:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=1.0 / SAMPLE_RATE_SCALE,
                mode='linear',
                align_corners=True
            )
        return (audio.clamp(min=-32768.0, max=32767.0)).to(torch.int16)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'], model_path=model_path)
    mossformer = myClearVoice.models[0].model.eval().float().to("cpu")
    mossformer = MOSSFORMER_SE(mossformer, custom_stft, custom_istft, NFFT, N_MELS, SAMPLE_RATE, MAX_SIGNAL_LENGTH)
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
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = max(SAMPLE_RATE * 6, audio_len)
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int) & (KEEP_ORIGINAL_SAMPLE_RATE):
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
inv_audio_len = float(100.0 / aligned_len)


if SAMPLE_RATE != 48000 and not KEEP_ORIGINAL_SAMPLE_RATE:
    SAMPLE_RATE = 48000
    audio_len = int(audio_len * SAMPLE_RATE_SCALE)


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
denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:audio_len]
end_time = time.time()
print(f"Complete: 100.00%")


# Save the denoised wav.
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
