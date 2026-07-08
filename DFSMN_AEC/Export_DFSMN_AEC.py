import gc
import subprocess
import sys
from pathlib import Path
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


parent_path         = Path(__file__).resolve().parent                          # The folder that contains this script.
project_path_A      = "/home/DakeQQ/Downloads/speech_dfsmn_aec_psm_16k"         # The DFSMN AEC download path.
project_path_B      = "/home/DakeQQ/Downloads/SDAEC-main"                       # The light-AEC project download path. Keywords in this path select the backend.
                                                                              # [SDAEC, Deep_Echo, NKF] are supported. https://github.com/ZhaoF-i/SDAEC ; https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation ; https://github.com/jfsean/NKF-AEC
onnx_model_A        = str(parent_path / "DFSMN_AEC_ONNX" / "DFSMN_AEC.onnx")  # The exported onnx model path.
onnx_model_Metadata = str(metadata_path_for_model(onnx_model_A))               # The metadata carrier onnx model path.


DYNAMIC_AXES         = False                          # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
IN_SAMPLE_RATE       = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE      = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as output.
IN_AUDIO_DTYPE       = 'INT16'                        # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'INT16'                        # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16            = float(1.0 / 32768.0)
MAX_SIGNAL_LENGTH    = 2048 if DYNAMIC_AXES else 256  # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
OPSET                = 18                             # ONNX opset.
USE_BATCH_FOLD       = True                           # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
BATCH_WINDOW_SECONDS = 1.5                            # Minimum input length (seconds) that triggers window folding.
INPUT_AUDIO_LENGTH   = 32000                          # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.


# DFSMN_AEC
MODEL_SAMPLE_RATE = 16000                           # The DFSMN AEC model runs internally at 16 kHz; the Kaldi fbank is computed at this rate.
WINDOW_TYPE     = 'hamming_symmetric'               # Mask STFT/ISTFT window: symmetric hamming, matching the original torch.hamming_window(640, periodic=False).
NFFT_A          = 1024                              # Kaldi fbank FFT size = next_power_of_two(frame_length); 40 ms @ 16 kHz -> 640 -> 1024. Edit it carefully.
NFFT_A2         = 640                               # Mask STFT FFT size (= n_fft from the dey_mini.yaml loss config), edit it carefully.
WINDOW_LENGTH_A = 640                               # Kaldi frame length (40 ms @ 16 kHz) and mask STFT window length, edit it carefully.
HOP_LENGTH_A    = 320                               # Frame shift (20 ms @ 16 kHz) for both the Kaldi fbank and the mask STFT.
N_MELS          = 80                                # Number of Mel bands (= feature_size in dey_mini.yaml), edit it carefully.
PRE_EMPHASIZE   = 0.97                              # Kaldi per-frame pre-emphasis coefficient.


# Light-AEC backend
def infer_light_aec_model(project_path: str) -> str:
    project_path_key = project_path.lower().replace('-', '_').replace(' ', '_')
    if 'deep_echo' in project_path_key or ('deep' in project_path_key and 'echo' in project_path_key):
        return 'Deep_Echo'
    if 'nkf' in project_path_key:
        return 'NKF'
    if 'sdaec' in project_path_key:
        return 'SDAEC'
    raise ValueError(
        f"Unable to infer LIGHT_AEC_MODEL from project_path_B: {project_path}. "
        "Expected keywords for 'SDAEC', 'Deep_Echo', or 'NKF'.")


LIGHT_AEC_MODEL = infer_light_aec_model(project_path_B)

if LIGHT_AEC_MODEL == 'NKF':
    DYNAMIC_AXES       = False
    INPUT_AUDIO_LENGTH = max(32000, INPUT_AUDIO_LENGTH)  # NKF AEC model is recommended >= 32000

if LIGHT_AEC_MODEL == 'SDAEC':
    WINDOW_TYPE_B   = 'hamming'                                                # Type of window function used in the STFT
    NFFT_B          = 319                                                      # Number of FFT components for the STFT process
    WINDOW_LENGTH_B = 319                                                      # Length of windowing, edit it carefully.
    HOP_LENGTH_B    = 160                                                      # Number of samples between successive frames in the STFT
    ALPHA_K         = 10                                                       # The SDAEC parameter, do not edit the value.
elif LIGHT_AEC_MODEL == 'Deep_Echo':
    WINDOW_TYPE_B   = 'hamming'                                                # Type of window function used in the STFT
    NFFT_B          = 319                                                      # Number of FFT components for the STFT process
    WINDOW_LENGTH_B = 319                                                      # Length of windowing, edit it carefully.
    HOP_LENGTH_B    = 160                                                      # Number of samples between successive frames in the STFT
    ECHO_ORDER      = 10                                                       # Number of Deep_Echo path orders. Do not edit.
elif LIGHT_AEC_MODEL == 'NKF':
    checkpoint_path_B = project_path_B + "/src/nkf_epoch70.pt"                 # The pretrained checkpoint path.
    WINDOW_TYPE_B     = 'hann'                                                 # Type of window function used in the STFT
    NFFT_B            = 1024                                                   # Number of FFT components for the STFT process
    WINDOW_LENGTH_B   = 1024                                                   # Length of windowing, edit it carefully.
    HOP_LENGTH_B      = 256                                                    # Number of samples between successive frames in the STFT
    FILTER_ORDER      = 4                                                      # Kalman filter order (L), do not edit the value.
    FC_DIM            = 18                                                     # Fully-connected layer dimension, do not edit the value.
    RNN_LAYERS        = 1                                                      # Number of GRU layers, do not edit the value.
    RNN_DIM           = 18                                                     # GRU hidden dimension, do not edit the value.
else:
    raise ValueError(f"Unknown LIGHT_AEC_MODEL: {LIGHT_AEC_MODEL}. Choose from 'SDAEC', 'Deep_Echo', 'NKF'.")


# Batch-fold: for a static, same-rate (in==model==out) input at least BATCH_WINDOW_SECONDS long, fold the near/far
# waveforms into fixed-length windows and batch-process them together (each window is an independent AEC clip -> the
# adaptive filter / recurrent state stays WITHIN a window; the window count rides in the batch dimension).
FOLD_WINDOW_LENGTH   = (
    (int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH_A - 1) // HOP_LENGTH_A
) * HOP_LENGTH_A  # Per-window model-rate length, rounded UP to a mask-hop (HOP_LENGTH_A) multiple so the center=False mask STFT/ISTFT reconstructs exactly W samples per window.
EXPORT_AUDIO_LENGTH = (
    ((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH
) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is zero-padded OUTSIDE the model by the windowing loop.
if USE_BATCH_FOLD:
    # In fold mode each backend/mask ISTFT operates per window, so MAX_SIGNAL_LENGTH only needs to cover a single
    # window's frame count. The smallest backend hop is 160 (SDAEC/Deep_Echo), so W/160 + margin bounds every ISTFT
    # (SDAEC 150, NKF 94, mask 74 frames for W=24000). The default 128 truncates the per-window backend STFT.
    MAX_SIGNAL_LENGTH = FOLD_WINDOW_LENGTH // 160 + 10

if LIGHT_AEC_MODEL == 'SDAEC':
    class AlphaPredictor(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.linear1 = torch.nn.Linear(2, 1)
            self.linear2 = torch.nn.Linear(k, 1)

        def forward(self, mix_comp, far_comp, k):
            pass


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

            n_fft = NFFT_B // 2 + 1
            self.n_fft = n_fft
            self.hop_len = n_fft
            win_length = n_fft
            half_n_fft = n_fft // 2
            expected_freq_bins = half_n_fft + 1
            if self.f != expected_freq_bins:
                raise ValueError(
                    f'SDAEC CepsUnit expects {self.f} inner STFT bins, but NFFT_B={NFFT_B} produces {expected_freq_bins}.')

            window = torch.ones(win_length)
            time_idx = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
            freq_idx = torch.arange(half_n_fft + 1, dtype=torch.float32).unsqueeze(1)
            omega = 2 * torch.pi * freq_idx * time_idx / n_fft
            cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
            sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
            self.register_buffer('stft_kernel', torch.cat([cos_kernel, sin_kernel], dim=0))

            fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
            fourier_basis_real_imag = torch.vstack([
                torch.real(fourier_basis[:half_n_fft + 1]),
                torch.imag(fourier_basis[:half_n_fft + 1])
            ]).float()
            inverse_basis_pinv = torch.linalg.pinv(fourier_basis_real_imag).T
            inverse_basis = window.unsqueeze(0) * inverse_basis_pinv.unsqueeze(1)
            self.register_buffer('inverse_basis', inverse_basis)

        def forward(self, x0):
            b = x0.shape[0]
            x_reshaped = x0.transpose(2, 3).contiguous().view(-1, 1, self.n_fft)
            stft_out = torch.nn.functional.conv1d(x_reshaped, self.stft_kernel, stride=self.hop_len)
            stft_out = stft_out.view(b, self.ch, -1, self.f2).transpose(2, 3).contiguous()
            stft_pair = stft_out.view(b, self.ch, 2, self.f, -1).transpose(1, 2).contiguous()
            stft_real, stft_imag = stft_pair.split(1, dim=1)
            stft_real = stft_real.squeeze(1)
            stft_imag = stft_imag.squeeze(1)

            lstm_output = self.ch_lstm_f(self.LN(stft_pair.reshape(b, self.ch * 2, self.f, -1)))
            processed_pair = lstm_output.view(b, 2, self.ch, self.f, -1)
            processed_real, processed_imag = processed_pair.split(1, dim=1)
            processed_real = processed_real.squeeze(1)
            processed_imag = processed_imag.squeeze(1)
            out_real = processed_real * stft_real - processed_imag * stft_imag
            out_imag = processed_real * stft_imag + processed_imag * stft_real

            out_pair = torch.stack([out_real, out_imag], dim=2)
            inp = out_pair.permute(0, 1, 4, 2, 3).contiguous().view(-1, self.f2, 1)
            inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
            x_out = inv.view(b, self.ch, -1, self.n_fft).transpose(2, 3).contiguous()
            return x_out


    class LayerNorm(torch.nn.Module):
        def __init__(self, c, f):
            super(LayerNorm, self).__init__()
            self.w = torch.nn.Parameter(torch.ones(1, c, f, 1))
            self.b = torch.nn.Parameter(torch.rand(1, c, f, 1) * 1e-4)
            reduced = c * f
            self.register_buffer('mean_scale', torch.tensor(1.0 / float(reduced), dtype=torch.float32))
            weight_scale = float(max(reduced - 1, 1)) ** 0.5
            self.register_buffer('weight_scale', torch.tensor(weight_scale, dtype=torch.float32))
            self.register_buffer('norm_eps', torch.tensor(1e-6 * weight_scale * weight_scale, dtype=torch.float32))
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


    class NET(torch.nn.Module):
        def __init__(self, order=10, channels=20, max_frames=2048, custom_istft=None):
            super().__init__()
            self.act = torch.nn.ELU()
            self.order = order
            self.custom_istft = custom_istft

            self.in_ch_lstm = CH_LSTM_F(4, channels, channels)
            self.in_conv = torch.nn.Conv2d(in_channels=4 + channels, out_channels=channels, kernel_size=(1, 1))
            self.cfb_e1 = CFB(channels, channels)
            self.cfb_e2 = CFB(channels, channels)
            self.cfb_e3 = CFB(channels, channels)
            self.cfb_e4 = CFB(channels, channels)
            self.cfb_e5 = CFB(channels, channels)
            self.ln = LayerNorm(channels, 160)
            self.ch_lstm = CH_LSTM_T(in_ch=channels, feat_ch=channels * 2, out_ch=channels, num_layers=2)
            self.cfb_d5 = CFB(1 * channels, channels)
            self.cfb_d4 = CFB(2 * channels, channels)
            self.cfb_d3 = CFB(2 * channels, channels)
            self.cfb_d2 = CFB(2 * channels, channels)
            self.cfb_d1 = CFB(2 * channels, channels)
            self.out_ch_lstm = CH_LSTM_T(2 * channels, channels, channels * 2)
            self.out_conv = torch.nn.Conv2d(in_channels=channels * 3, out_channels=2, kernel_size=(1, 1), padding=(0, 0), bias=True)

        def forward(self, x):
            e0 = self.in_ch_lstm(x)
            e0 = self.in_conv(torch.cat([e0, x], 1))
            e1 = self.cfb_e1(e0)
            e2 = self.cfb_e2(e1)
            e3 = self.cfb_e3(e2)
            e4 = self.cfb_e4(e3)
            e5 = self.cfb_e5(e4)
            lstm_out = self.ch_lstm(self.ln(e5))
            d5 = self.cfb_d5(e5 * lstm_out)
            d4 = self.cfb_d4(torch.cat([e4, d5], dim=1))
            d3 = self.cfb_d3(torch.cat([e3, d4], dim=1))
            d2 = self.cfb_d2(torch.cat([e2, d3], dim=1))
            d1 = self.cfb_d1(torch.cat([e1, d2], dim=1))
            d0 = self.out_ch_lstm(torch.cat([e0, d1], dim=1))
            out = self.out_conv(torch.cat([d0, d1], dim=1))
            real, imag = out.split(1, dim=1)
            return self.custom_istft(real.squeeze(1), imag.squeeze(1))


    class CH_LSTM_T(torch.nn.Module):
        def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1, f=160):
            super().__init__()
            self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
            self.bi = 1 if not bi else 2
            self.linear = torch.nn.Linear(self.bi * feat_ch, out_ch)
            self.out_ch = out_ch
            self.f = f

        def forward(self, x):
            self.lstm2.flatten_parameters()
            b = x.shape[0]
            x = x.permute(0, 2, 3, 1).contiguous().view(b * self.f, -1, self.lstm2.input_size)
            x, _ = self.lstm2(x)
            x = self.linear(x)
            x = x.view(b, self.f, -1, self.out_ch).permute(0, 3, 1, 2).contiguous()
            return x


    class CH_LSTM_F(torch.nn.Module):
        def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1, f=160):
            super().__init__()
            self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
            self.linear = torch.nn.Linear(2 * feat_ch, out_ch)
            self.out_ch = out_ch
            self.f = f

        def forward(self, x):
            self.lstm2.flatten_parameters()
            b = x.shape[0]
            x = x.transpose(1, 3).contiguous().view(-1, self.f, self.lstm2.input_size)
            x, _ = self.lstm2(x)
            x = self.linear(x)
            x = x.view(b, -1, self.f, self.out_ch).transpose(1, 3).contiguous()
            return x
elif LIGHT_AEC_MODEL == 'Deep_Echo':
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

            n_fft = NFFT_B // 2 + 1
            self.n_fft = n_fft
            self.hop_len = n_fft
            expected_freq_bins = n_fft // 2 + 1
            if self.f != expected_freq_bins:
                raise ValueError(
                    f'Deep_Echo CepsUnit expects {self.f} inner STFT bins, but NFFT_B={NFFT_B} produces {expected_freq_bins}.')

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
            b = x0.shape[0]
            x_reshaped = x0.transpose(2, 3).contiguous().view(-1, 1, self.n_fft)
            stft_out = torch.nn.functional.conv1d(x_reshaped, self.stft_kernel, stride=self.hop_len)
            stft_pair = stft_out.view(b, self.ch, -1, 2, self.f).permute(0, 3, 1, 4, 2).contiguous()
            stft_real, stft_imag = stft_pair.split(1, dim=1)
            stft_real = stft_real.squeeze(1)
            stft_imag = stft_imag.squeeze(1)

            lstm_output = self.ch_lstm_f(self.LN(stft_pair.reshape(b, self.ch * 2, self.f, -1)))
            processed_pair = lstm_output.view(b, 2, self.ch, self.f, -1)
            processed_real, processed_imag = processed_pair.split(1, dim=1)
            processed_real = processed_real.squeeze(1)
            processed_imag = processed_imag.squeeze(1)
            out_real = processed_real * stft_real - processed_imag * stft_imag
            out_imag = processed_real * stft_imag + processed_imag * stft_real

            inp = torch.cat([out_real, out_imag], dim=2).transpose(2, 3).contiguous().view(-1, self.f2, 1)
            inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
            x_out = inv.view(b, self.ch, -1, self.n_fft).transpose(2, 3).contiguous()
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
            b = x.shape[0]
            x = x.permute(0, 2, 3, 1).contiguous().view(b * self.f, -1, self.lstm2.input_size)
            x, _ = self.lstm2(x)
            x = self.linear(x)
            x = x.view(b, self.f, -1, self.out_ch).permute(0, 3, 1, 2).contiguous()
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
            b = x.shape[0]
            x = x.transpose(1, 3).contiguous().view(-1, self.f, self.lstm2.input_size)
            x, _ = self.lstm2(x)
            x = self.linear(x)
            x = x.view(b, -1, self.f, self.out_ch).transpose(1, 3).contiguous()
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
            batch = x.shape[0]
            mix_comp = x[:, 0::2].contiguous()
            far_comp = x[:, 1::2].contiguous()
            e0 = self.in_ch_lstm(x)
            e0 = self.in_conv(torch.cat([e0, x], dim=1))
            e1 = self.cfb_e1(e0)
            lstm_out = self.ch_lstm(self.ln(e1))
            d1 = self.cfb_d1(e1 * lstm_out)
            d0 = self.out_ch_lstm(torch.cat([e0, d1], dim=1))
            out = self.out_conv(torch.cat([d0, d1], dim=1))
            est_echo_path = out.view(batch, 2, self.order, n_freq, -1)
            enhanced = mix_comp - self.apply_echo_path(far_comp, est_echo_path, batch, n_freq)
            enhanced_real, enhanced_imag = enhanced.split(1, dim=1)
            aec_results = self.custom_istft(enhanced_real.squeeze(1), enhanced_imag.squeeze(1))
            return aec_results, est_echo_path
elif LIGHT_AEC_MODEL == 'NKF':
    class ComplexGRU_Real(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True):
            super().__init__()
            self.gru_r = torch.nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first)
            self.gru_i = torch.nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first)
            self.num_layers = num_layers
            self.hidden_size = hidden_size

        def forward(self, x_real, x_imag, h_rr, h_ir, h_ri, h_ii):
            Frr, h_rr = self.gru_r(x_real, h_rr)
            Fir, h_ir = self.gru_r(x_imag, h_ir)
            Fri, h_ri = self.gru_i(x_real, h_ri)
            Fii, h_ii = self.gru_i(x_imag, h_ii)
            out_real = Frr - Fii
            out_imag = Fri + Fir
            return out_real, out_imag, h_rr, h_ir, h_ri, h_ii


    class ComplexDense_Real(torch.nn.Module):
        def __init__(self, in_channel, out_channel, bias=True):
            super().__init__()
            self.linear_real = torch.nn.Linear(in_channel, out_channel, bias=bias)
            self.linear_imag = torch.nn.Linear(in_channel, out_channel, bias=bias)
            self.out_channel = out_channel
            self._fused = False

        def fuse_weights_(self):
            with torch.no_grad():
                self.register_buffer('stacked_weight', torch.stack([self.linear_real.weight, self.linear_imag.weight], dim=0).transpose(1, 2).contiguous())
                if self.linear_real.bias is not None:
                    self.register_buffer('stacked_bias', torch.stack([self.linear_real.bias, self.linear_imag.bias], dim=0).unsqueeze(1))
                else:
                    self.register_buffer('stacked_bias', None)
            self._fused = True

        def forward(self, x_real, x_imag):
            if self._fused and x_real.shape[0] == 1 and x_imag.shape[0] == 1:
                x_stacked = torch.cat([x_real, x_imag], dim=0)
                out = torch.bmm(x_stacked, self.stacked_weight)
                if self.stacked_bias is not None:
                    out = out + self.stacked_bias
                return out.split(1, dim=0)
            return self.linear_real(x_real), self.linear_imag(x_imag)


    class ComplexPReLU_Real(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prelu = torch.nn.PReLU()
            self.register_buffer('cached_weight', torch.ones(1, dtype=torch.float32), persistent=False)
            self.use_cached_weight = False

        def cache_weight_(self):
            with torch.no_grad():
                self.cached_weight.copy_(self.prelu.weight.detach())
            self.use_cached_weight = True

        def forward(self, x_real, x_imag):
            weight = self.cached_weight if self.use_cached_weight else self.prelu.weight
            return torch.nn.functional.prelu(x_real, weight), torch.nn.functional.prelu(x_imag, weight)


    class KGNet_Real(torch.nn.Module):
        def __init__(self, L, fc_dim, rnn_layers, rnn_dim):
            super().__init__()
            self.L = L
            self.rnn_layers = rnn_layers
            self.rnn_dim = rnn_dim
            self.n_freq = NFFT_B // 2 + 1

            self.fc_in_dense = ComplexDense_Real(2 * L + 1, fc_dim, bias=True)
            self.fc_in_act = ComplexPReLU_Real()
            self.complex_gru = ComplexGRU_Real(fc_dim, rnn_dim, rnn_layers)
            self.fc_out_dense1 = ComplexDense_Real(rnn_dim, fc_dim, bias=True)
            self.fc_out_act = ComplexPReLU_Real()
            self.fc_out_dense2 = ComplexDense_Real(fc_dim, L, bias=True)

        def cache_prelu_weights_(self):
            self.fc_in_act.cache_weight_()
            self.fc_out_act.cache_weight_()

        def fuse_dense_weights_(self):
            self.fc_in_dense.fuse_weights_()
            self.fc_out_dense1.fuse_weights_()
            self.fc_out_dense2.fuse_weights_()

        def forward(self, input_real, input_imag, h_rr, h_ir, h_ri, h_ii):
            b = input_real.shape[0]
            x_real, x_imag = self.fc_in_dense(input_real, input_imag)
            x_real, x_imag = self.fc_in_act(x_real, x_imag)

            x_real = x_real.view(b * self.n_freq, 1, -1)
            x_imag = x_imag.view(b * self.n_freq, 1, -1)

            x_real, x_imag, h_rr, h_ir, h_ri, h_ii = self.complex_gru(x_real, x_imag, h_rr, h_ir, h_ri, h_ii)

            x_real = x_real.view(b, self.n_freq, -1)
            x_imag = x_imag.view(b, self.n_freq, -1)

            x_real, x_imag = self.fc_out_dense1(x_real, x_imag)
            x_real, x_imag = self.fc_out_act(x_real, x_imag)
            x_real, x_imag = self.fc_out_dense2(x_real, x_imag)

            return x_real, x_imag, h_rr, h_ir, h_ri, h_ii


    class NKF_Inner(torch.nn.Module):
        def __init__(self, L, fc_dim, rnn_layers, rnn_dim, custom_stft, custom_istft, max_frames):
            super().__init__()
            self.L = L
            self.rnn_layers = rnn_layers
            self.rnn_dim = rnn_dim
            self.custom_stft = custom_stft
            self.custom_istft = custom_istft
            self.kg_net = KGNet_Real(L, fc_dim, rnn_layers, rnn_dim)
            self.n_freq = NFFT_B // 2 + 1
            self.max_frames = max_frames

            self.register_buffer('h_prior_real_buffer', torch.zeros(1, self.n_freq, L, dtype=torch.float32))
            self.register_buffer('h_prior_imag_buffer', torch.zeros(1, self.n_freq, L, dtype=torch.float32))
            self.register_buffer('h_rr_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
            self.register_buffer('h_ir_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
            self.register_buffer('h_ri_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
            self.register_buffer('h_ii_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
            self.register_buffer('zeros_pad', torch.zeros(1, self.n_freq, L - 1, dtype=torch.float32))

        def cache_export_constants_(self):
            self.kg_net.cache_prelu_weights_()
            self.kg_net.fuse_dense_weights_()

        def forward(self, audio_pair):
            b = audio_pair.shape[0] // 2
            pair_real, pair_imag = self.custom_stft(audio_pair)
            ref_real, mic_real = pair_real.split(b, dim=0)
            ref_imag, mic_imag = pair_imag.split(b, dim=0)

            T = ref_real.shape[2]
            if T > self.max_frames:
                raise ValueError(f"Input produces {T} frames, but MAX_SIGNAL_LENGTH is {self.max_frames}.")

            h_prior_real = self.h_prior_real_buffer.expand(b, -1, -1)
            h_prior_imag = self.h_prior_imag_buffer.expand(b, -1, -1)
            h_rr = self.h_rr_buffer.repeat(1, b, 1)
            h_ir = self.h_ir_buffer.repeat(1, b, 1)
            h_ri = self.h_ri_buffer.repeat(1, b, 1)
            h_ii = self.h_ii_buffer.repeat(1, b, 1)

            ref_real_padded = torch.cat([self.zeros_pad.expand(b, -1, -1), ref_real], dim=2)
            ref_imag_padded = torch.cat([self.zeros_pad.expand(b, -1, -1), ref_imag], dim=2)

            echo_frames_real = []
            echo_frames_imag = []

            xt_real = ref_real_padded[..., :self.L]
            xt_imag = ref_imag_padded[..., :self.L]
            mic_real_t = mic_real[..., [0]]
            mic_imag_t = mic_imag[..., [0]]

            input_real = torch.cat([xt_real, mic_real_t, h_prior_real], dim=2)
            input_imag = torch.cat([xt_imag, mic_imag_t, h_prior_imag], dim=2)

            kg_real, kg_imag, h_rr, h_ir, h_ri, h_ii = self.kg_net(input_real, input_imag, h_rr, h_ir, h_ri, h_ii)

            h_post_real = kg_real * mic_real_t - kg_imag * mic_imag_t
            h_post_imag = kg_real * mic_imag_t + kg_imag * mic_real_t

            echo_frames_real.append((xt_real * h_post_real - xt_imag * h_post_imag).sum(dim=2, keepdim=True))
            echo_frames_imag.append((xt_real * h_post_imag + xt_imag * h_post_real).sum(dim=2, keepdim=True))

            for t in range(1, T):
                xt_real = ref_real_padded[..., t:t + self.L]
                xt_imag = ref_imag_padded[..., t:t + self.L]
                mic_real_t = mic_real[..., t:t + 1]
                mic_imag_t = mic_imag[..., t:t + 1]

                dh_real = h_post_real - h_prior_real
                dh_imag = h_post_imag - h_prior_imag
                h_prior_real, h_post_real = h_post_real, h_prior_real
                h_prior_imag, h_post_imag = h_post_imag, h_prior_imag

                e_real = mic_real_t - (xt_real * h_prior_real - xt_imag * h_prior_imag).sum(dim=2, keepdim=True)
                e_imag = mic_imag_t - (xt_real * h_prior_imag + xt_imag * h_prior_real).sum(dim=2, keepdim=True)

                input_real = torch.cat([xt_real, e_real, dh_real], dim=2)
                input_imag = torch.cat([xt_imag, e_imag, dh_imag], dim=2)

                kg_real, kg_imag, h_rr, h_ir, h_ri, h_ii = self.kg_net(input_real, input_imag, h_rr, h_ir, h_ri, h_ii)

                h_post_real = h_prior_real + kg_real * e_real - kg_imag * e_imag
                h_post_imag = h_prior_imag + kg_real * e_imag + kg_imag * e_real

                echo_frames_real.append((xt_real * h_post_real - xt_imag * h_post_imag).sum(dim=2, keepdim=True))
                echo_frames_imag.append((xt_real * h_post_imag + xt_imag * h_post_real).sum(dim=2, keepdim=True))

            echo_hat_real = torch.cat(echo_frames_real, dim=2)
            echo_hat_imag = torch.cat(echo_frames_imag, dim=2)

            out_real = mic_real - echo_hat_real
            out_imag = mic_imag - echo_hat_imag
            light_aec_results = self.custom_istft(out_real, out_imag)
            return light_aec_results


    def load_nkf_weights(export_model, original_state_dict):
        new_state_dict = {}

        new_state_dict['kg_net.fc_in_dense.linear_real.weight'] = original_state_dict['kg_net.fc_in.0.linear_real.weight']
        new_state_dict['kg_net.fc_in_dense.linear_real.bias'] = original_state_dict['kg_net.fc_in.0.linear_real.bias']
        new_state_dict['kg_net.fc_in_dense.linear_imag.weight'] = original_state_dict['kg_net.fc_in.0.linear_imag.weight']
        new_state_dict['kg_net.fc_in_dense.linear_imag.bias'] = original_state_dict['kg_net.fc_in.0.linear_imag.bias']
        new_state_dict['kg_net.fc_in_act.prelu.weight'] = original_state_dict['kg_net.fc_in.1.prelu.weight']

        for gru_name in ['gru_r', 'gru_i']:
            for param in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']:
                key_src = f'kg_net.complex_gru.{gru_name}.{param}'
                key_dst = f'kg_net.complex_gru.{gru_name}.{param}'
                new_state_dict[key_dst] = original_state_dict[key_src]

        new_state_dict['kg_net.fc_out_dense1.linear_real.weight'] = original_state_dict['kg_net.fc_out.0.linear_real.weight']
        new_state_dict['kg_net.fc_out_dense1.linear_real.bias'] = original_state_dict['kg_net.fc_out.0.linear_real.bias']
        new_state_dict['kg_net.fc_out_dense1.linear_imag.weight'] = original_state_dict['kg_net.fc_out.0.linear_imag.weight']
        new_state_dict['kg_net.fc_out_dense1.linear_imag.bias'] = original_state_dict['kg_net.fc_out.0.linear_imag.bias']
        new_state_dict['kg_net.fc_out_act.prelu.weight'] = original_state_dict['kg_net.fc_out.1.prelu.weight']
        new_state_dict['kg_net.fc_out_dense2.linear_real.weight'] = original_state_dict['kg_net.fc_out.2.linear_real.weight']
        new_state_dict['kg_net.fc_out_dense2.linear_real.bias'] = original_state_dict['kg_net.fc_out.2.linear_real.bias']
        new_state_dict['kg_net.fc_out_dense2.linear_imag.weight'] = original_state_dict['kg_net.fc_out.2.linear_imag.weight']
        new_state_dict['kg_net.fc_out_dense2.linear_imag.bias'] = original_state_dict['kg_net.fc_out.2.linear_imag.bias']

        export_model.load_state_dict(new_state_dict, strict=False)
        return export_model


def build_kaldi_fbank_conv(n_fft, win_length, n_mels, sample_rate, preemphasis_coefficient,
                           low_freq=20.0, high_freq=0.0):
    """Fold torchaudio.compliance.kaldi.fbank into a single Conv1d weight + Mel matrix.

    The original DFSMN-AEC preprocessor extracts features with
    ``kaldi.fbank(utt, frame_length=40, frame_shift=20, num_mel_bins=80,
    sample_frequency=16000, window_type='hamming')`` on int16-scale audio. Every
    per-frame step is linear, so DC-offset removal, 0.97 pre-emphasis, the symmetric
    Hamming window and the real/imag DFT collapse into one stride-``hop`` convolution.

    The int16 scale (32768) is baked into the weight, so the convolution accepts the
    [-1, 1] waveform used by the light-AEC backend while still producing the int16-scale
    power spectrum Kaldi expects (clamp(eps).log() then reproduces the Kaldi log-mel).

    Returns:
        conv_weight: (2 * (n_fft // 2 + 1), 1, win_length) float32 Conv1d weight.
        mel_banks:   (1, n_mels, n_fft // 2 + 1) float32 Mel filterbank.
    """
    import torchaudio.compliance.kaldi as kaldi
    f_bins = n_fft // 2 + 1
    window = torch.hamming_window(win_length, periodic=False, alpha=0.54, beta=0.46, dtype=torch.float64)
    t = torch.arange(win_length, dtype=torch.float64).unsqueeze(0)
    f = torch.arange(f_bins, dtype=torch.float64).unsqueeze(1)
    omega = (2.0 * torch.pi / n_fft) * f * t
    windowed_cos = torch.cos(omega) * window.unsqueeze(0)
    windowed_sin = -torch.sin(omega) * window.unsqueeze(0)
    # Kaldi order: per-frame DC removal (M), then pre-emphasis (P), then window + DFT.
    dc_remove = torch.eye(win_length, dtype=torch.float64) - 1.0 / win_length
    preemph = torch.eye(win_length, dtype=torch.float64)
    for j in range(1, win_length):
        preemph[j, j - 1] = -preemphasis_coefficient
    preemph[0, 0] = 1.0 - preemphasis_coefficient
    real_weight = windowed_cos @ preemph @ dc_remove
    imag_weight = windowed_sin @ preemph @ dc_remove
    conv_weight = torch.cat([real_weight, imag_weight], dim=0).unsqueeze(1) * 32768.0
    mel_banks, _ = kaldi.get_mel_banks(n_mels, n_fft, float(sample_rate), low_freq, high_freq, 100.0, -500.0, 1.0)
    mel_banks = torch.nn.functional.pad(mel_banks.to(torch.float64), (0, 1)).unsqueeze(0)
    return conv_weight.float(), mel_banks.float()


class DFSMN_AEC(torch.nn.Module):
    def __init__(self, dfsmn_aec, light_aec, light_aec_type, custom_stft_A2, custom_istft_A2, custom_stft_B, nfft_A, win_length_A, hop_length_A, pre_emphasis, in_sample_rate, out_sample_rate, n_mels, use_batch_fold=False, fold_window=0, alpha_predictor=None, k=None):
        super(DFSMN_AEC, self).__init__()
        if light_aec is None:
            raise ValueError('light_aec must be provided.')
        self.use_batch_fold = use_batch_fold          # Fold long near/far inputs into fixed windows batched on dim0
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.dfsmn_aec = dfsmn_aec.model.to('cpu').float()
        # Inlined UniDeepFsmn causal FSMN memory (replaces modeling_modified/uni_deep_fsmn.py).
        # Every deepfsmn layer shares input_dim and padding_left (= dilation * (lorder - 1)),
        # so a single zero pad template feeds all of them.
        first_uni_deep_fsmn = self.dfsmn_aec.deepfsmn[0]
        self.register_buffer('fsmn_pad', torch.zeros(1, first_uni_deep_fsmn.input_dim, first_uni_deep_fsmn.padding_left, 1, dtype=torch.float32))
        self.register_buffer('shift', dfsmn_aec.preprocessor.feature.shift.view(1, 1, -1))
        self.register_buffer('scale', dfsmn_aec.preprocessor.feature.scale.view(1, 1, -1))
        self.light_aec = light_aec
        self.light_aec_type = light_aec_type
        self.custom_stft_A2 = custom_stft_A2
        self.custom_istft_A2 = custom_istft_A2
        self.custom_stft_B = custom_stft_B
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.output_pcm_scale = 32767.0
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.out_sample_rate_scale = out_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before_centering = self.in_sample_rate_scale > 1.0
        self.resample_after_centering = self.in_sample_rate_scale < 1.0
        # Output sandwich: resample DOWN before the PCM scale and UP after it so the scale
        # multiply always runs on the smaller (model-rate) tensor (mirrors the input sandwich above).
        self.output_resample_before_pcm = self.out_sample_rate_scale < 1.0   # out < model -> downsample first
        self.output_resample_after_pcm = self.out_sample_rate_scale > 1.0    # out > model -> upsample after

        if self.light_aec_type == 'SDAEC':
            if alpha_predictor is None or custom_stft_B is None or k is None:
                raise ValueError('SDAEC backend requires alpha_predictor, custom_stft_B, and ALPHA_K.')
            linear2_weight = alpha_predictor.linear2.weight.reshape(1, k)
            linear1_weight = alpha_predictor.linear1.weight[0]
            alpha_bias = alpha_predictor.linear2.bias + linear2_weight.sum(dim=1) * alpha_predictor.linear1.bias[0]
            alpha_mix_weight = linear2_weight * linear1_weight[1]
            alpha_far_weight = linear2_weight * linear1_weight[0]
            alpha_kernel = torch.stack([alpha_mix_weight, alpha_far_weight], dim=1)
            self.register_buffer('alpha_conv_kernel', alpha_kernel)
            self.register_buffer('alpha_conv_bias', alpha_bias)
            self.register_buffer('alpha_power_pad', torch.zeros(1, 2, k - 1, dtype=torch.float32))
        elif self.light_aec_type == 'Deep_Echo':
            if custom_stft_B is None:
                raise ValueError('Deep_Echo backend requires custom_stft_B.')
        elif self.light_aec_type != 'NKF':
            raise ValueError(f'Unsupported light AEC backend: {self.light_aec_type}')

        fbank_conv_weight, mel_banks = build_kaldi_fbank_conv(nfft_A, win_length_A, n_mels, MODEL_SAMPLE_RATE, pre_emphasis)
        self.register_buffer('fbank_conv_weight', fbank_conv_weight)
        self.register_buffer('mel_banks', mel_banks)
        self.feat_hop_length = hop_length_A
        self.feat_n_freq = nfft_A // 2 + 1
        self.feat_dim = n_mels * 3
        self.log_floor = float(torch.finfo(torch.float32).eps)
        self.factor = float(1.15)  # Matches the original DFSMN echo estimate scaling.

    def _preprocess_audio_pair(self, near_end_audio, far_end_audio):
        audio_pair = torch.cat([near_end_audio, far_end_audio], dim=0).float()
        if self.resample_before_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower():
            audio_pair = audio_pair * self.inv_int16      # int16 PCM -> [-1, 1]; F16/F32 inputs already arrive normalized.
        if self.resample_after_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        return audio_pair

    def _run_sdaec_backend(self, audio_pair, batch):
        pair_real_part, pair_imag_part = self.custom_stft_B(audio_pair)
        # audio_pair is (2*batch, 1, W) with the near block first and the far block second; reshape the STFT to
        # (near_or_far, batch, real_or_imag, freq, time) so each window's near/far branches stay separate.
        pair_comp = torch.stack([pair_real_part, pair_imag_part], dim=1)
        n_freq, n_time = pair_comp.shape[-2], pair_comp.shape[-1]
        pair_comp = pair_comp.view(2, batch, 2, n_freq, n_time)
        mix_comp = pair_comp[0]
        far_comp = pair_comp[1]
        frame_power = pair_comp.square().sum(dim=(2, 3)).permute(1, 0, 2)  # (batch, [near, far], time)
        alpha_input = torch.cat([self.alpha_power_pad.expand(batch, -1, -1), frame_power], dim=-1)
        alpha = torch.nn.functional.conv1d(alpha_input, self.alpha_conv_kernel, bias=self.alpha_conv_bias).unsqueeze(2)
        far_comp = far_comp * torch.abs(alpha)
        return self.light_aec(torch.cat([mix_comp, far_comp], dim=1))

    def _run_deep_echo_backend(self, audio_pair, batch):
        pair_real_part, pair_imag_part = self.custom_stft_B(audio_pair)
        # audio_pair is (2*batch, 1, W): near block then far block. Regroup to per-window channels
        # [near_real, far_real, near_imag, far_imag] so x[:, 0::2] is near and x[:, 1::2] is far inside Deep_Echo.
        n_freq = pair_real_part.shape[1]
        pr = pair_real_part.view(2, batch, n_freq, -1)
        pi = pair_imag_part.view(2, batch, n_freq, -1)
        spec_input = torch.stack([pr, pi], dim=0).permute(2, 0, 1, 3, 4).reshape(batch, 4, n_freq, -1)
        return self.light_aec(spec_input, n_freq)[0]

    def _run_nkf_backend(self, audio_pair, batch):
        # NKF consumes raw waveforms ordered as [far, near] and runs its own STFT/ISTFT internally.
        return self.light_aec(torch.cat([audio_pair[batch:2 * batch], audio_pair[:batch]], dim=0))

    def _run_light_aec_backend(self, audio_pair, batch):
        if self.light_aec_type == 'SDAEC':
            return self._run_sdaec_backend(audio_pair, batch)
        if self.light_aec_type == 'Deep_Echo':
            return self._run_deep_echo_backend(audio_pair, batch)
        if self.light_aec_type == 'NKF':
            return self._run_nkf_backend(audio_pair, batch)
        raise RuntimeError(f'Unsupported light AEC backend: {self.light_aec_type}')

    def _uni_deep_fsmn(self, uni_deep_fsmn, x):
        # Inlined UniDeepFsmn.compute1: linear -> ReLU -> (Identity) norm -> project -> causal depthwise FSMN.
        # The zero-concat left padding reproduces the modeling_modified forward exactly while calling only
        # the pristine modelscope submodules, so no site-packages patching is required.
        hidden = uni_deep_fsmn.project(uni_deep_fsmn.norm(uni_deep_fsmn.act(uni_deep_fsmn.linear(x))))
        hidden = hidden.transpose(1, 2).unsqueeze(-1)
        memory = uni_deep_fsmn.conv1(torch.cat([self.fsmn_pad.expand(hidden.shape[0], -1, -1, -1), hidden], dim=-2))
        if uni_deep_fsmn.skip_connect:
            memory = memory + hidden
        memory = memory.transpose(1, 2).squeeze(-1)
        return x + memory

    def forward(self, near_end_audio, far_end_audio):
        # 0. Optionally fold long near/far inputs into fixed windows batched on dim0. _preprocess stacks near/far on
        #    dim0, so a fold of `batch` windows makes the pair (2*batch, 1, W): near block first, far block second.
        if self.use_batch_fold:
            near_end_audio = near_end_audio.reshape(-1, 1, self.fold_window)
            far_end_audio = far_end_audio.reshape(-1, 1, self.fold_window)
        batch = near_end_audio.shape[0]

        # 1. Convert int16 near/far inputs into the float waveform domain expected by the light-AEC backend.
        audio_pair = self._preprocess_audio_pair(near_end_audio, far_end_audio)

        # 2. Run the selected light-AEC backend to obtain a temporary echo-cancelled waveform.
        temp_aec = self._run_light_aec_backend(audio_pair, batch)
        if temp_aec.shape[-1] > audio_pair.shape[-1]:
            temp_aec = temp_aec[:, :, :audio_pair.shape[-1]]

        # 3. Align the near-end signal to the waveform length actually produced by the backend ISTFT.
        near_end_audio_f = audio_pair[:batch, :, :temp_aec.shape[-1]]

        # 4. Build the DFSMN Kaldi-fbank features from the near-end mic, the temporary AEC
        #    output (the linear-AEC residual = out_linear) and the echo estimate
        #    (out_echo ~= near - factor * temp_aec). The folded Conv1d applies per-frame DC
        #    removal, 0.97 pre-emphasis and the symmetric Hamming window, just like kaldi.fbank.
        temp_aec_real_part_A2, temp_aec_imag_part_A2 = self.custom_stft_A2(temp_aec)
        echo = near_end_audio_f - self.factor * temp_aec
        fbank_input = torch.cat([near_end_audio_f, temp_aec, echo], dim=0)
        fbank_spectrum = torch.nn.functional.conv1d(fbank_input, self.fbank_conv_weight, stride=self.feat_hop_length).square()
        real_part_A, imag_part_A = torch.split(fbank_spectrum, self.feat_n_freq, dim=1)
        power = real_part_A + imag_part_A
        mel_energies = torch.matmul(self.mel_banks, power).clamp(min=self.log_floor).log()
        feat = mel_energies.reshape(3, batch, self.feat_dim // 3, -1).permute(1, 0, 2, 3).reshape(batch, self.feat_dim, -1).transpose(1, 2)
        feat = (feat + self.shift) * self.scale
        x1 = self.dfsmn_aec.linear1(feat)
        x2 = self.dfsmn_aec.relu(x1)
        x3 = x2
        for uni_deep_fsmn in self.dfsmn_aec.deepfsmn:
            x3 = self._uni_deep_fsmn(uni_deep_fsmn, x3)
        masks = self.dfsmn_aec.sig(self.dfsmn_aec.linear2(x3)).transpose(1, 2)

        # 5. Apply the DFSMN mask to the raw temporary AEC spectrum and synthesize the final waveform.
        temp_aec_real_part_A2 *= masks
        temp_aec_imag_part_A2 *= masks
        audio = self.custom_istft_A2(temp_aec_real_part_A2, temp_aec_imag_part_A2)
        if self.output_resample_before_pcm:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            audio = audio * self.output_pcm_scale      # [-1, 1] -> int16 PCM; F16/F32 outputs stay normalized.
        if self.output_resample_after_pcm:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            audio = audio.reshape(1, 1, -1)
        if "int" in OUT_AUDIO_DTYPE.lower():
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return audio
        return audio.to(torch.float16)




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_DFSMN_ONNX_AEC.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    custom_stft_A2 = STFT_Process(model_type='stft_B', n_fft=NFFT_A2, hop_len=HOP_LENGTH_A, win_length=WINDOW_LENGTH_A, max_frames=0, window_type=WINDOW_TYPE, center_pad=False, pad_mode='constant').eval()
    custom_istft_A2 = STFT_Process(model_type='istft_B', n_fft=NFFT_A2, hop_len=HOP_LENGTH_A, win_length=WINDOW_LENGTH_A, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=False, pad_mode='constant').eval()
    custom_stft_B = STFT_Process(model_type='stft_B', n_fft=NFFT_B, hop_len=HOP_LENGTH_B, win_length=WINDOW_LENGTH_B, max_frames=0, window_type=WINDOW_TYPE_B, center_pad=True, pad_mode='constant').eval()
    custom_istft_B = STFT_Process(model_type='istft_B', n_fft=NFFT_B, hop_len=HOP_LENGTH_B, win_length=WINDOW_LENGTH_B, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE_B, center_pad=True, pad_mode='constant').eval()
    light_aec = None
    alpha_predictor = None

    if LIGHT_AEC_MODEL == 'SDAEC':
        light_aec = NET(max_frames=MAX_SIGNAL_LENGTH, custom_istft=custom_istft_B)
        light_aec.load_state_dict(torch.load(project_path_B + '/Model/ICCRN.ckpt', map_location='cpu'), strict=False)
        for module in light_aec.modules():
            if isinstance(module, LayerNorm):
                module.fuse_var_scale_()
        light_aec = light_aec.float().eval()
        alpha_predictor = AlphaPredictor(ALPHA_K)
        alpha_predictor.load_state_dict(torch.load(project_path_B + '/Model/alpha.ckpt', map_location='cpu'), strict=False)
        alpha_predictor = alpha_predictor.float().eval()
    elif LIGHT_AEC_MODEL == 'Deep_Echo':
        light_aec = NET(order=ECHO_ORDER, custom_istft=custom_istft_B)
        checkpoint_path = Path(project_path_B).expanduser().resolve() / 'model.ckpt'
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = extract_state_dict(checkpoint)
        light_aec.load_state_dict(state_dict, strict=False)
        fuse_layer_norm_scales_(light_aec)
        light_aec = light_aec.float().eval()
    else:
        light_aec = NKF_Inner(
            L=FILTER_ORDER,
            fc_dim=FC_DIM,
            rnn_layers=RNN_LAYERS,
            rnn_dim=RNN_DIM,
            custom_stft=custom_stft_B,
            custom_istft=custom_istft_B,
            max_frames=MAX_SIGNAL_LENGTH
        ).eval()
        original_state_dict = torch.load(checkpoint_path_B, map_location='cpu')
        light_aec = load_nkf_weights(light_aec, original_state_dict)
        light_aec = light_aec.float().eval()
        light_aec.cache_export_constants_()

    dfsmn_pipeline = pipeline(
        Tasks.acoustic_echo_cancellation,
        model=project_path_A,
        device='cpu',
        trust_remote_code=True
    )

    dfsmn_aec = DFSMN_AEC(
        dfsmn_pipeline,
        light_aec=light_aec,
        light_aec_type=LIGHT_AEC_MODEL,
        custom_stft_A2=custom_stft_A2,
        custom_istft_A2=custom_istft_A2,
        custom_stft_B=custom_stft_B if LIGHT_AEC_MODEL != 'NKF' else None,
        nfft_A=NFFT_A,
        win_length_A=WINDOW_LENGTH_A,
        hop_length_A=HOP_LENGTH_A,
        pre_emphasis=PRE_EMPHASIZE,
        in_sample_rate=IN_SAMPLE_RATE,
        out_sample_rate=OUT_SAMPLE_RATE,
        n_mels=N_MELS,
        use_batch_fold=USE_BATCH_FOLD,
        fold_window=FOLD_WINDOW_LENGTH,
        alpha_predictor=alpha_predictor,
        k=ALPHA_K if LIGHT_AEC_MODEL == 'SDAEC' else None
    )
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    near_end_audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    far_end_audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)

    torch.onnx.export(
        dfsmn_aec,
        (near_end_audio, far_end_audio),
        onnx_model_A,
        input_names=['near_end_audio', 'far_end_audio'],
        output_names=['aec_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'near_end_audio': {2: 'audio_len'},
            'far_end_audio': {2: 'audio_len'},
            'aec_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del dfsmn_aec
    del dfsmn_pipeline
    del light_aec
    del alpha_predictor
    del near_end_audio
    del far_end_audio
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="DFSMN_AEC", task="aec", model_family="dfsmn_aec",
    max_dynamic_audio_seconds=30, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=2, feature_kind="kaldi_fbank_stft_aec", center_pad=False, pad_mode="constant",
    extra={
        "light_aec_model": LIGHT_AEC_MODEL, "n_mels": N_MELS, "nfft_a": NFFT_A, "nfft_a2": NFFT_A2,
        "window_length_a": WINDOW_LENGTH_A, "hop_length_a": HOP_LENGTH_A, "nfft_b": NFFT_B,
        "window_length_b": WINDOW_LENGTH_B, "hop_length_b": HOP_LENGTH_B, "window_type_b": WINDOW_TYPE_B,
        "preemphasize": PRE_EMPHASIZE, "alpha_k": globals().get("ALPHA_K"),
        "echo_order": globals().get("ECHO_ORDER"), "filter_order": globals().get("FILTER_ORDER"),
    },
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
