import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torch.fft
from pydub import AudioSegment
from STFT_Process import STFT_Process


project_path        = "/home/DakeQQ/Downloads/Deep-echo-path-modeling-for-acoustic-echo-cancellation-main"   # The Deep_Echo_AEC GitHub project download path. https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation
onnx_model_A        = "/home/DakeQQ/Downloads/Deep_Echo_AEC_ONNX/Deep_Echo_AEC.onnx"    # The exported onnx model path.
test_near_end_audio = "./examples/nearend_mic1.wav"                                     # The near end audio path.
test_far_end_audio  = "./examples/farend_speech1.wav"                                   # The far end audio path.
save_aec_output     = "./aec.wav"                                                       # The output Acoustic Echo Cancellation audio path.


DYNAMIC_AXES       = False                          # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
IN_SAMPLE_RATE     = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE    = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH = 16001                          # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.
MAX_SIGNAL_LENGTH  = 2048 if DYNAMIC_AXES else 128  # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
MAX_THREADS        = 4                              # Number of parallel threads for test audio denoising.
NORMALIZE_AUDIO    = False                          # Normalize the input audio to a target RMS level (e.g., 8192) before processing. It can help improve the performance of the model, especially for low-volume audio. Set it to True if you want to enable it.
OPSET              = 18                             # ONNX opset.
WINDOW_TYPE        = 'hamming'                      # Type of window function used in the STFT.
NFFT               = 319                            # Number of FFT components for the STFT process.
WINDOW_LENGTH      = 319                            # Length of windowing, edit it carefully.
HOP_LENGTH         = 160                            # Number of samples between successive frames in the STFT.
ECHO_ORDER         = 10                             # Number of Deep_Echo path orders. Do not edit.


def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-7))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio


def maybe_strip_prefix(state_dict, prefix: str):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'network', 'net'):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported checkpoint format. Expected a state_dict-like mapping.')

    for prefix in ('module.', 'model.', 'network.', 'net.'):
        checkpoint = maybe_strip_prefix(checkpoint, prefix)

    return checkpoint


class CFB(torch.nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(CFB, self).__init__()
        self.conv_gate = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv_input = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, groups=1, bias=True)
        self.ceps_unit = CepsUnit(ch=out_channels)
        self.LN0 = LayerNorm(in_channels, f=160)
        self.LN1 = LayerNorm(out_channels, f=160)
        self.LN2 = LayerNorm(out_channels, f=160)

    def forward(self, x):
        g = torch.sigmoid(self.conv_gate(self.LN0(x)))
        x = self.conv_input(x)
        gx = g * x
        y = self.conv(self.LN1(gx))
        y = y + self.ceps_unit(self.LN2(x - gx))
        return y


class CepsUnit(torch.nn.Module):
    def __init__(self, ch):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.ch_lstm_f = CH_LSTM_F(ch * 2, ch, ch * 2, f=81)
        self.LN = LayerNorm(ch * 2, f=81)
        self.f = 81
        self.f2 = 81 * 2

        n_fft = NFFT // 2 + 1
        self.n_fft = n_fft
        self.hop_len = n_fft

        window = torch.ones(n_fft)
        time_idx = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        freq_idx = torch.arange(self.f, dtype=torch.float32).unsqueeze(1)
        omega = 2 * torch.pi * freq_idx * time_idx / n_fft
        cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
        self.register_buffer('stft_kernel', torch.cat([cos_kernel, sin_kernel], dim=0))

        fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
        fourier_basis_real_imag = torch.vstack([
            torch.real(fourier_basis[:self.f]),
            torch.imag(fourier_basis[:self.f])
        ]).float()
        inverse_basis_pinv = torch.linalg.pinv(fourier_basis_real_imag).T
        inverse_basis = window.unsqueeze(0) * inverse_basis_pinv.unsqueeze(1)
        self.register_buffer('inverse_basis', inverse_basis)

    def forward(self, x0):
        x_reshaped = x0.transpose(2, 3).contiguous().view(-1, 1, self.n_fft)
        stft_out = torch.nn.functional.conv1d(x_reshaped, self.stft_kernel, stride=self.hop_len)
        stft_pair = stft_out.view(self.ch, -1, 2, self.f).permute(2, 0, 3, 1).contiguous()
        stft_real, stft_imag = stft_pair.split(1, dim=0)

        lstm_output = self.ch_lstm_f(self.LN(stft_pair.view(1, self.ch * 2, self.f, -1)))
        processed_pair = lstm_output.view(2, self.ch, self.f, -1)
        processed_real, processed_imag = processed_pair.split(1, dim=0)
        out_real = processed_real * stft_real - processed_imag * stft_imag
        out_imag = processed_real * stft_imag + processed_imag * stft_real

        inp = torch.cat([out_real, out_imag], dim=2).transpose(2, 3).contiguous().view(-1, self.f2, 1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        x_out = inv.view(1, self.ch, -1, self.n_fft).transpose(2, 3).contiguous()
        return x_out


class LayerNorm(torch.nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.w = torch.nn.Parameter(torch.ones(1, c, f, 1))
        self.b = torch.nn.Parameter(torch.rand(1, c, f, 1) * 1e-4)
        denom = float(max(c * f - 1, 1))
        self.register_buffer('weight_scale', torch.tensor(denom ** 0.5, dtype=torch.float32))
        self.register_buffer('norm_eps', torch.tensor(1e-8 * denom, dtype=torch.float32))
        self.var_scale_fused = False

    def fuse_var_scale_(self):
        if self.var_scale_fused:
            return
        with torch.no_grad():
            self.w.mul_(self.weight_scale.to(dtype=self.w.dtype))
        self.var_scale_fused = True

    def forward(self, x):
        mean = x.mean((1, 2), keepdim=True)
        x = x - mean
        inv_std = torch.rsqrt(x.square().sum((1, 2), keepdim=True) + self.norm_eps)
        return x * inv_std * self.w + self.b


def fuse_layer_norm_scales_(module):
    for submodule in module.modules():
        if isinstance(submodule, LayerNorm):
            submodule.fuse_var_scale_()


class CH_LSTM_T(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1, f=160):
        super(CH_LSTM_T, self).__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.bi = 1 if not bi else 2
        self.linear = torch.nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f

    def forward(self, x):
        self.lstm2.flatten_parameters()
        x = x.permute(0, 2, 3, 1).contiguous().view(self.f, -1, self.lstm2.input_size)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        x = x.view(1, self.f, -1, self.out_ch).permute(0, 3, 1, 2).contiguous()
        return x


class CH_LSTM_F(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1, f=160):
        super(CH_LSTM_F, self).__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.linear = torch.nn.Linear(2 * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f

    def forward(self, x):
        self.lstm2.flatten_parameters()
        x = x.transpose(1, 3).contiguous().view(-1, self.f, self.lstm2.input_size)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        x = x.view(1, -1, self.f, self.out_ch).transpose(1, 3).contiguous()
        return x


class NET(torch.nn.Module):
    def __init__(self, order=ECHO_ORDER, channels=20, custom_istft=None):
        super(NET, self).__init__()
        self.order = order
        self.custom_istft = custom_istft
        self.register_buffer('delay_bank_kernel', torch.eye(order, dtype=torch.float32).unsqueeze(1))
        self.register_buffer('pad_zeros', torch.zeros(1, 1, 1, order - 1))

        self.in_ch_lstm = CH_LSTM_F(4, channels, channels)
        self.in_conv = torch.nn.Conv2d(in_channels=4 + channels, out_channels=channels, kernel_size=(1, 1))
        self.cfb_e1 = CFB(channels, channels)
        self.ln = LayerNorm(channels, 160)
        self.ch_lstm = CH_LSTM_T(in_ch=channels, feat_ch=channels * 2, out_ch=channels, num_layers=2)
        self.cfb_d1 = CFB(channels, channels)
        self.out_ch_lstm = CH_LSTM_T(2 * channels, channels, channels * 2)
        self.out_conv = torch.nn.Conv2d(in_channels=channels * 3, out_channels=order * 2, kernel_size=(1, 1), padding=(0, 0), bias=True)

    def apply_echo_path(self, far_comp, est_echo_path, batch, n_freq):
        padding_far = torch.cat([self.pad_zeros.expand(batch, 2, n_freq, -1), far_comp], dim=-1)
        delayed_far = torch.nn.functional.conv1d(padding_far.reshape(batch * 2 * n_freq, 1, -1), self.delay_bank_kernel)
        delayed_far = delayed_far.view(batch, 2, n_freq, self.order, -1).permute(0, 1, 3, 2, 4).contiguous()
        far_real, far_imag = delayed_far.split(1, dim=1)
        path_real, path_imag = est_echo_path.split(1, dim=1)
        est_echo_real = far_real * path_real - far_imag * path_imag
        est_echo_imag = far_real * path_imag + far_imag * path_real
        return torch.cat([est_echo_real.sum(dim=2), est_echo_imag.sum(dim=2)], dim=1)

    def forward(self, x, n_freq):
        mix_comp = x[:, 0::2].contiguous()
        far_comp = x[:, 1::2].contiguous()
        e0 = self.in_ch_lstm(x)
        e0 = self.in_conv(torch.cat([e0, x], dim=1))
        e1 = self.cfb_e1(e0)
        lstm_out = self.ch_lstm(self.ln(e1))
        d1 = self.cfb_d1(e1 * lstm_out)
        d0 = self.out_ch_lstm(torch.cat([e0, d1], dim=1))
        out = self.out_conv(torch.cat([d0, d1], dim=1))
        est_echo_path = out.view(1, 2, self.order, n_freq, -1)
        enhanced = mix_comp - self.apply_echo_path(far_comp, est_echo_path, 1, n_freq)
        enhanced_real, enhanced_imag = enhanced.squeeze(0).split(1, dim=0)
        aec_results = self.custom_istft(enhanced_real, enhanced_imag)
        return aec_results, est_echo_path


class DeepEchoAEC(torch.nn.Module):
    def __init__(self, iccrn, custom_stft, in_sample_rate, out_sample_rate):
        super(DeepEchoAEC, self).__init__()
        self.iccrn = iccrn
        self.custom_stft = custom_stft
        self.inv_int16 = float(1.0 / 32768.0)
        self.output_pcm_scale = 32767.0
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.out_sample_rate_scale = out_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before_centering = self.in_sample_rate_scale > 1.0
        self.resample_after_centering = self.in_sample_rate_scale < 1.0
        self.output_resample_before_pcm = self.out_sample_rate_scale > 1.0
        self.output_resample_after_pcm = self.out_sample_rate_scale < 1.0

    def forward(self, near_end_audio, far_end_audio):
        audio_pair = torch.cat([near_end_audio, far_end_audio], dim=0).float()
        if self.resample_before_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        audio_pair *= self.inv_int16
        audio_pair = audio_pair - audio_pair.mean(dim=2, keepdim=True)
        if self.resample_after_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        pair_real_part, pair_imag_part = self.custom_stft(audio_pair)
        n_freq = pair_real_part.shape[1]
        spec_input = torch.cat([pair_real_part, pair_imag_part], dim=0).unsqueeze(0)
        aec_results, _ = self.iccrn(spec_input, n_freq)

        if self.output_resample_before_pcm:
            aec_results = torch.nn.functional.interpolate(
                aec_results,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        aec_results *= self.output_pcm_scale
        if self.output_resample_after_pcm:
            aec_results = torch.nn.functional.interpolate(
                aec_results,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        return aec_results.clamp(min=-32768.0, max=32767.0).to(torch.int16)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
Path(save_aec_output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant").eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant").eval()
    iccrn = NET(order=ECHO_ORDER, custom_istft=custom_istft)
    checkpoint_path = Path(project_path).expanduser().resolve() / 'model.ckpt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)
    iccrn.load_state_dict(state_dict, strict=False)
    fuse_layer_norm_scales_(iccrn)
    iccrn = iccrn.float().eval()

    deep_echo = DeepEchoAEC(iccrn, custom_stft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE)
    near_end_audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    far_end_audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)

    torch.onnx.export(
        deep_echo,
        (near_end_audio, far_end_audio),
        onnx_model_A,
        input_names=['near_end_audio', 'far_end_audio'],
        output_names=['aec_audio'],
        dynamic_axes={
            'near_end_audio': {2: 'near_audio_len'},
            'far_end_audio': {2: 'far_audio_len'},
            'aec_audio': {2: 'aec_audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del deep_echo
    del iccrn
    del near_end_audio
    del far_end_audio
    gc.collect()
print('\nExport done!\n\nStart to run Deep_Echo by ONNX Runtime.\n\nNow, loading the model...')


session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry('session.set_denormal_as_zero', '1')
session_opts.add_session_config_entry('session.intra_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.inter_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.enable_quant_qdq_cleanup', '1')
session_opts.add_session_config_entry('session.qdq_matmulnbits_accuracy_level', '4')
session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
session_opts.add_session_config_entry('disable_synchronize_execution_providers', '1')
session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name


print(f"\nTest Input Near_End Audio: {test_near_end_audio}\nTest Input Far_End Audio: {test_far_end_audio}")
near_end_audio = np.array(AudioSegment.from_file(test_near_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
far_end_audio = np.array(AudioSegment.from_file(test_far_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
near_end_audio_len = len(near_end_audio)
far_end_audio_len = len(far_end_audio)
min_len = min(near_end_audio_len, far_end_audio_len)
if NORMALIZE_AUDIO:
    near_end_audio = normalise_audio(near_end_audio)
    far_end_audio = normalise_audio(far_end_audio)
near_end_audio = near_end_audio[:min_len]
far_end_audio = far_end_audio[:min_len]
near_end_audio = near_end_audio.reshape(1, 1, -1)
far_end_audio = far_end_audio.reshape(1, 1, -1)

shape_value_in = ort_session_A.get_inputs()[0].shape[-1]
shape_value_out = ort_session_A.get_outputs()[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = max(30 * IN_SAMPLE_RATE, min_len)  # Default to slice in 30 seconds. You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in


def align_audio(audio, audio_len):
    stride_step = INPUT_AUDIO_LENGTH
    if audio_len > INPUT_AUDIO_LENGTH:
        if (shape_value_in != shape_value_out) and isinstance(shape_value_in, int) and isinstance(shape_value_out, int) and (OUT_SAMPLE_RATE == IN_SAMPLE_RATE):
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
    return audio, aligned_len, stride_step


near_end_audio, _, _ = align_audio(near_end_audio, min_len)
far_end_audio, aligned_len, stride_step = align_audio(far_end_audio, min_len)


min_len = int(min_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
inv_audio_len = float(100.0 / min_len)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _near_end_audio, _far_end_audio):
    output = ort_session_A.run([out_name_A0], {in_name_A0: _near_end_audio[:, :, _slice_start:_slice_end], in_name_A1: _far_end_audio[:, :, _slice_start:_slice_end]})[0]
    return _slice_start * _inv_audio_len, output


print('\nRunning the Deep_Echo AEC by ONNX Runtime.')
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS if MAX_THREADS > 0 else 2) as executor:
    futures = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, near_end_audio, far_end_audio))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.3f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:min_len]
end_time = time.time()
print('Complete: 100.00%')


elapsed = end_time - start_time
audio_duration = min_len / OUT_SAMPLE_RATE
rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
sf.write(save_aec_output, denoised_wav, OUT_SAMPLE_RATE, format='WAVEX')
print(f"\nAEC Process Complete.\n\nSaving to: {save_aec_output}.\n\nTime Cost: {elapsed:.3f} Seconds\nAudio Duration: {audio_duration:.3f} Seconds\nRTF: {rtf:.4f}")