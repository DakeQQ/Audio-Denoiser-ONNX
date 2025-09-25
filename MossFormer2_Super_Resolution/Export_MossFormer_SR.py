import gc
import time
import site
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


model_path = "/home/DakeQQ/Downloads/MossFormer2_SR_48K"
onnx_model_A = "/home/DakeQQ/Downloads/MossFormer_ONNX/MossFormer2_SR.onnx"     # The exported onnx model path.
test_audio = "./examples/speech_with_noise1.wav"                                # The original audio path.
save_generated_audio = "./examples/speech_with_noise1_super_resolution.wav"     # The output super resolution audio path.


DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 48000              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 8000. Higher values yield better quality but time consume. It is better to set an integer multiple of the NFFT value.
MAX_SIGNAL_LENGTH = 1280                # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hamming'                 # Type of window function used in the STFT
N_MELS = 80                             # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 1024                             # Number of FFT components for the STFT process
WINDOW_LENGTH = 1024                    # Length of windowing, edit it carefully.
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT
NFFT_POST = 256                         # The MossFormer_SR parameter, do not edit the value.
WINDOW_LENGTH_POST = 256                # The MossFormer_SR parameter, do not edit the value.
HOP_LENGTH_POST = 128                   # The MossFormer_SR parameter, do not edit the value.
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.
SPUER_SAMPLE_RATE = 48000               # The target audio sample rate, do not edit the value.
ORIGINAL_SAMPLE_RATE = 16000            # The input audio sample rate. This value cannot be changed after the ONNX model is exported.


site_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/__init__.py", site_package_path + "/clearvoice/__init__.py")
shutil.copyfile("./modeling_modified/network_wrapper.py", site_package_path + "/clearvoice/network_wrapper.py")
from clearvoice import ClearVoice


class MOSSFORMER_SR(torch.nn.Module):
    def __init__(self, mossformer_sr, pre_stft, post_stft, post_istft, pre_nfft, post_nfft, n_mels, original_sample_rate, super_sample_rate, energy_threshold: float = 0.99, transition_ms: int = 100):
        super(MOSSFORMER_SR, self).__init__()
        self.mossformer_sr = mossformer_sr
        self.pre_stft = pre_stft
        self.post_stft = post_stft
        self.post_istft = post_istft
        self.fbank = (torchaudio.functional.melscale_fbanks(pre_nfft // 2 + 1, 0, 8000, n_mels, super_sample_rate, 'slaney', 'slaney')).transpose(0, 1).unsqueeze(0)
        self.scale_factor = float(super_sample_rate / original_sample_rate)
        self.energy_thr = float(energy_threshold)
        step = float(super_sample_rate / post_nfft)
        self.transition_len = int(round(transition_ms * super_sample_rate * 0.001))
        fade = torch.linspace(0., 1., self.transition_len, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("fade_ramp", fade, persistent=False)
        self.register_buffer("fade_ramp_inv", 1.0 - fade, persistent=False)
        self.freq_vec  = (torch.arange(post_nfft // 2 + 1, dtype=torch.float32) * step).view(1, -1, 1)

    def forward(self, audio):
        audio = audio.float()
        audio = audio - torch.mean(audio)
        if self.scale_factor != 1.0:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.scale_factor,
                mode="linear",
                align_corners=True,
                recompute_scale_factor=False
            )
        real_part, imag_part = self.pre_stft(audio, 'reflect')
        real_a, imag_a = self.post_stft(audio, 'reflect')
        mel_features = torch.matmul(self.fbank, torch.sqrt(real_part * real_part + imag_part * imag_part)).clamp(min=1e-5).log()
        mossformer_output = self.mossformer_sr[0](mel_features)
        generated_wav = self.mossformer_sr[1](mossformer_output)
        generated_wav = generated_wav[..., :audio.shape[-1]]
        real_b, imag_b = self.post_stft(generated_wav, 'reflect')
        psd = (real_a * real_a + imag_a * imag_a) * self.post_istft.inv_win_sum_for_mossformer
        energy_per_f = psd.sum(dim=-1)     
        cum_energy = energy_per_f.cumsum(dim=1)
        total_e = cum_energy[:, -1: ]
        cum_ratio = cum_energy / (total_e + 1e-6)
        mask_hi = (cum_ratio >= self.energy_thr).to(torch.uint8)
        idx_hi = mask_hi.argmax(dim=1, keepdim=True)
        f_high = self.freq_vec.squeeze(-1).gather(1, idx_hi).unsqueeze(-1)
        lp_mask = (self.freq_vec <= f_high).to(real_a.dtype)
        lp_mask_minus = 1. - lp_mask
        real_mix = real_a * lp_mask + real_b * lp_mask_minus
        imag_mix = imag_a * lp_mask + imag_b * lp_mask_minus
        wav_sub = self.post_istft(real_mix, imag_mix)
        cross_a = audio[..., :self.transition_len] * self.fade_ramp_inv
        cross_b = wav_sub[..., :self.transition_len] * self.fade_ramp
        head = cross_a + cross_b
        tail = wav_sub[..., self.transition_len:]
        smoothed = torch.cat([head, tail], dim=-1)
        super_audio = smoothed.clamp(-32768.0, 32767.0).to(torch.int16)
        return super_audio


print('Export start ...')
with torch.inference_mode():
    pre_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    post_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_POST, hop_len=HOP_LENGTH_POST, win_length=WINDOW_LENGTH_POST, max_frames=0, window_type=WINDOW_TYPE).eval()
    post_istft = STFT_Process(model_type='istft_B', n_fft=NFFT_POST, hop_len=HOP_LENGTH_POST, win_length=WINDOW_LENGTH_POST, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
   
    myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'], model_path=model_path)
    mossformer = myClearVoice.models[0].model.eval().float().to("cpu")
    mossformer = MOSSFORMER_SR(mossformer, pre_stft, post_stft, post_istft, NFFT, NFFT_POST, N_MELS, ORIGINAL_SAMPLE_RATE, SPUER_SAMPLE_RATE, transition_ms=int(0.1 * INPUT_AUDIO_LENGTH * 1000 / ORIGINAL_SAMPLE_RATE))
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        mossformer,
        (audio,),
        onnx_model_A,
        input_names=['original_audio'],
        output_names=['super_resolution_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'original_audio': {2: 'audio_len'},
            'super_resolution_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del mossformer
    del audio
    del pre_stft
    del post_stft
    del post_istft
    gc.collect()
print('\nExport done!\n\nStart to run MossFormer_SR by ONNX Runtime.\n\nNow, loading the model...')


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
print(f"\nTest Input Audio: {test_audio}")
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(ORIGINAL_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(80000, audio_len)
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int):
        stride_step = shape_value_out
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    white_noise = np.zeros((1, 1, pad_amount), dtype=audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    white_noise = np.zeros((1, 1, INPUT_AUDIO_LENGTH - audio_len), dtype=audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]
inv_audio_len = float(100.0 / aligned_len)
super_audio_len = audio_len * SPUER_SAMPLE_RATE // ORIGINAL_SAMPLE_RATE


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Start to run MossFormer_SR
print("\nRunning the MossFormer_SR by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel generate the audio.
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
generated_wav = np.concatenate(saved, axis=-1).reshape(-1)[:super_audio_len]
end_time = time.time()

# Save the generated wav.
sf.write(save_generated_audio, generated_wav, SPUER_SAMPLE_RATE, format='WAVEX')
print(f"\nSuper Resolution Process Complete.\n\nSaving to: {save_generated_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
