import gc
import subprocess
import sys
from pathlib import Path

import torch
import torch.fft
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


project_path           = "/home/DakeQQ/Downloads/SDAEC-main"             # The SDAEC GitHub project download path. https://github.com/ZhaoF-i/SDAEC
parent_path            = Path(__file__).resolve().parent               # The folder that contains this script.
onnx_model_A           = str(parent_path / "SDAEC_ONNX" / "SDAEC.onnx") # The exported onnx model path.
onnx_model_Metadata    = str(metadata_path_for_model(onnx_model_A))      # The metadata carrier onnx model path.


DYNAMIC_AXES          = False                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
IN_SAMPLE_RATE        = 16000                         # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE       = 16000                         # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
MODEL_SAMPLE_RATE     = 16000                         # The SDAEC model runs internally at 16 kHz.
INPUT_AUDIO_LENGTH    = 32000                         # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.
MAX_SIGNAL_LENGTH     = 2048 if DYNAMIC_AXES else 256 # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.

WINDOW_TYPE           = 'hamming'                     # Type of window function used in the STFT
NFFT                  = 319                           # Number of FFT components for the STFT process
WINDOW_LENGTH         = 319                           # Length of windowing, edit it carefully.
HOP_LENGTH            = 160                           # Number of samples between successive frames in the STFT
ALPHA_K               = 10                            # The SDAEC parameter, do not edit the value.

IN_AUDIO_DTYPE        = 'INT16'                       # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE       = 'INT16'                       # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16             = float(1.0 / 32768.0)
OPSET                 = 18                            # ONNX opset.

BATCH_WINDOW_SECONDS  = 1.5                           # Minimum input length (seconds) that triggers window folding.
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length rounded UP to a HOP multiple.
USE_BATCH_FOLD        = False                         # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH
if USE_BATCH_FOLD:
    MAX_SIGNAL_LENGTH = FOLD_WINDOW_LENGTH // HOP_LENGTH + 10  # Per-window STFT frame count (+margin); the default 128 would truncate a folded window's backend STFT.


class AlphaPredictor(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.linear2 = torch.nn.Linear(k, 1)
        # self.ReLU = nn.ReLU()

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

        # --- Pre-computation and buffer registration ---
        n_fft = NFFT // 2 + 1
        self.n_fft = n_fft
        self.hop_len = n_fft
        win_length = n_fft
        half_n_fft = n_fft // 2

        # Fused STFT Kernel (cos + sin in single tensor: one conv1d instead of two)
        window = torch.ones(win_length)
        time_idx = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        freq_idx = torch.arange(half_n_fft + 1, dtype=torch.float32).unsqueeze(1)
        omega = 2 * torch.pi * freq_idx * time_idx / n_fft
        cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
        self.register_buffer('stft_kernel', torch.cat([cos_kernel, sin_kernel], dim=0))

        # ISTFT Basis
        fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
        fourier_basis_real_imag = torch.vstack([
            torch.real(fourier_basis[:half_n_fft + 1]),
            torch.imag(fourier_basis[:half_n_fft + 1])
        ]).float()
        inverse_basis_pinv = torch.linalg.pinv(fourier_basis_real_imag).T
        inverse_basis = window.unsqueeze(0) * inverse_basis_pinv.unsqueeze(1)
        self.register_buffer('inverse_basis', inverse_basis)

    def forward(self, x0):
        # --- Fused STFT: single conv1d with combined cos+sin kernel ---
        b = x0.shape[0]
        x_reshaped = x0.transpose(2, 3).contiguous().view(-1, 1, self.n_fft)
        stft_out = torch.nn.functional.conv1d(x_reshaped, self.stft_kernel, stride=self.hop_len)
        stft_out = stft_out.view(b, self.ch, -1, self.f2).transpose(2, 3).contiguous()
        stft_pair = stft_out.view(b, self.ch, 2, self.f, -1).transpose(1, 2).contiguous()
        stft_real, stft_imag = stft_pair.split(1, dim=1)
        stft_real = stft_real.squeeze(1)
        stft_imag = stft_imag.squeeze(1)

        # --- CepsUnit Logic (single reshape replaces slice+cat re-packing) ---
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

        # --- Model Layers ---
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


class SDAEC(torch.nn.Module):
    def __init__(self, iccrn, alpha_predictor, custom_stft, nfft, k, max_len, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(SDAEC, self).__init__()
        self.iccrn = iccrn
        self.use_batch_fold = use_batch_fold          # Fold long near/far inputs into fixed windows batched on dim0
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.custom_stft = custom_stft
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

        # --- Fully fused alpha predictor: causal 2-channel conv1d ---
        linear2_weight = alpha_predictor.linear2.weight.reshape(1, k)
        linear1_weight = alpha_predictor.linear1.weight[0]
        alpha_bias = alpha_predictor.linear2.bias + linear2_weight.sum(dim=1) * alpha_predictor.linear1.bias[0]
        alpha_mix_weight = linear2_weight * linear1_weight[1]
        alpha_far_weight = linear2_weight * linear1_weight[0]
        alpha_kernel = torch.stack([alpha_mix_weight, alpha_far_weight], dim=1)
        self.register_buffer('alpha_conv_kernel', alpha_kernel)
        self.register_buffer('alpha_conv_bias', alpha_bias)
        self.register_buffer('alpha_power_pad', torch.zeros(1, 2, k - 1, dtype=torch.float32))

    def forward(self, near_end_audio, far_end_audio):
        # Optionally fold long near/far inputs into fixed windows batched on dim0 (each window is an independent AEC
        # clip). cat stacks near/far on dim0, so a fold of `batch` windows makes the pair (2*batch, 1, W).
        if self.use_batch_fold:
            near_end_audio = near_end_audio.reshape(-1, 1, self.fold_window)
            far_end_audio = far_end_audio.reshape(-1, 1, self.fold_window)
        batch = near_end_audio.shape[0]
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
        pair_comp = torch.cat([pair_real_part, pair_imag_part], dim=1).view(2, batch, 2, n_freq, -1)
        mix_comp = pair_comp[0]
        far_comp = pair_comp[1]

        frame_power = pair_comp.square().sum(dim=(2, 3)).permute(1, 0, 2)
        alpha_input = torch.cat([self.alpha_power_pad.expand(batch, -1, -1), frame_power], dim=-1)
        alpha = torch.nn.functional.conv1d(alpha_input, self.alpha_conv_kernel, bias=self.alpha_conv_bias).unsqueeze(2)

        far_comp = far_comp * torch.abs(alpha)
        audio = self.iccrn(torch.cat([mix_comp, far_comp], dim=1))
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
    inference_script = Path(__file__).resolve().with_name('Inference_SDAEC_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant").eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant").eval()
    iccrn = NET(max_frames=MAX_SIGNAL_LENGTH, custom_istft=custom_istft)
    iccrn.load_state_dict(torch.load(project_path + '/Model/ICCRN.ckpt', map_location='cpu'), strict=False)
    for module in iccrn.modules():
        if isinstance(module, LayerNorm):
            module.fuse_var_scale_()
    iccrn = iccrn.float().eval()
    alpha_predictor = AlphaPredictor(ALPHA_K)
    alpha_predictor.load_state_dict(torch.load(project_path + '/Model/alpha.ckpt', map_location='cpu'), strict=False)
    alpha_predictor = alpha_predictor.float().eval()

    sdaec = SDAEC(iccrn, alpha_predictor, custom_stft, NFFT, ALPHA_K, MAX_SIGNAL_LENGTH, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH)
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    near_end_audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    far_end_audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)

    torch.onnx.export(
        sdaec,
        (near_end_audio, far_end_audio),
        onnx_model_A,
        input_names=['near_end_audio', 'far_end_audio'],
        output_names=['aec_audio'],
        dynamic_axes={
            'near_end_audio': {2: 'audio_len'},
            'far_end_audio': {2: 'audio_len'},
            'aec_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del sdaec
    del iccrn
    del alpha_predictor
    del near_end_audio
    del far_end_audio
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="SDAEC", task="aec", model_family="sdaec",
    max_dynamic_audio_seconds=30, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=2, feature_kind="stft_alpha_predictor", center_pad=True, pad_mode="constant", extra={"alpha_k": ALPHA_K},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
