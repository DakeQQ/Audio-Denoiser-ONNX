import gc
import subprocess
import sys
from pathlib import Path

import torch
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


project_path           = str(Path.home() / "Downloads" / "SDAEC-main") # The SDAEC GitHub project download path. https://github.com/ZhaoF-i/SDAEC
parent_path            = Path(__file__).resolve().parent               # The folder that contains this script.
onnx_model_A           = str(parent_path / "SDAEC_ONNX" / "SDAEC.onnx") # The exported onnx model path.
onnx_model_Metadata    = str(metadata_path_for_model(onnx_model_A))      # The metadata carrier onnx model path.


DYNAMIC_AXES          = False                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
IN_SAMPLE_RATE        = 16000                         # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE       = 16000                         # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
MODEL_SAMPLE_RATE     = 16000                         # The SDAEC model runs internally at 16 kHz.
INPUT_AUDIO_LENGTH    = 32000                         # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.

WINDOW_TYPE           = 'hamming'                     # Type of window function used in the STFT
NFFT                  = 319                           # Number of FFT components for the STFT process
WINDOW_LENGTH         = 319                           # Length of windowing, edit it carefully.
HOP_LENGTH            = 160                           # Number of samples between successive frames in the STFT
STATIC_SIGNAL_LENGTH  = None if DYNAMIC_AXES else (INPUT_AUDIO_LENGTH + 2 * (NFFT // 2) - NFFT) // HOP_LENGTH + 1
MAX_SIGNAL_LENGTH     = 2048 if DYNAMIC_AXES else STATIC_SIGNAL_LENGTH # Exact centered-STFT frame count for static export.
ALPHA_K               = 10                            # The SDAEC parameter, do not edit the value.

IN_AUDIO_DTYPE        = 'INT16'                       # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE       = 'INT16'                       # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16             = float(1.0 / 32768.0)
OPSET                 = 20                            # ONNX opset.

BATCH_WINDOW_SECONDS  = 1.5                           # Minimum input length (seconds) that triggers window folding.
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length rounded UP to a HOP multiple.
USE_BATCH_FOLD        = False                         # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH
STATIC_MODEL_BATCH    = None if DYNAMIC_AXES else (EXPORT_AUDIO_LENGTH // FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else 1)
if DYNAMIC_AXES:
    raise ValueError("The optimized SDAEC exporter requires a static audio-length profile; export separate fixed profiles instead.")
if USE_BATCH_FOLD:
    STATIC_SIGNAL_LENGTH = (FOLD_WINDOW_LENGTH + 2 * (NFFT // 2) - NFFT) // HOP_LENGTH + 1
    MAX_SIGNAL_LENGTH = STATIC_SIGNAL_LENGTH


class AlphaPredictor(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.linear2 = torch.nn.Linear(k, 1)
        # self.ReLU = nn.ReLU()

    def forward(self, mix_comp, far_comp, k):
        pass


class CFB(torch.nn.Module):
    def __init__(self, in_channels=None, out_channels=None, model_batch=1, signal_length=200):
        super(CFB, self).__init__()
        self.conv_gate = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv_input = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, groups=1, bias=True)
        self.ceps_unit = CepsUnit(ch=out_channels, model_batch=model_batch, signal_length=signal_length)
        self.LN0 = LayerNorm(in_channels, f=160)
        self.LN1 = LayerNorm(out_channels, f=160)
        self.LN2 = LayerNorm(out_channels, f=160)
        self.model_batch = model_batch
        self.signal_length = signal_length

    def forward(self, x, freq_initial_state, cepstral_dft_weight, cepstral_idft_weight):
        # Canonical internal layout is (batch, time, frequency, channels).
        batch = self.model_batch if self.model_batch is not None else x.shape[0]
        frames = self.signal_length if self.signal_length is not None else x.shape[1]
        g = torch.sigmoid(torch.nn.functional.linear(
            self.LN0(x), self.conv_gate.weight[:, :, 0, 0], self.conv_gate.bias
        ))
        x = torch.nn.functional.linear(x, self.conv_input.weight[:, :, 0, 0], self.conv_input.bias)
        gx = g * x
        y = self.LN1(gx).reshape(-1, 160, self.conv.out_channels).transpose(1, 2)
        y = torch.nn.functional.conv1d(
            y, self.conv.weight.squeeze(-1), self.conv.bias, padding=1
        )
        y = y.transpose(1, 2)
        y = y.reshape(batch, frames, 160, self.conv.out_channels)
        y = y + self.ceps_unit(
            self.LN2(x - gx), freq_initial_state, cepstral_dft_weight, cepstral_idft_weight
        )
        return y


class CepsUnit(torch.nn.Module):
    def __init__(self, ch, model_batch=1, signal_length=200):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.ch_lstm_f = CH_LSTM_F(
            ch * 2, ch, ch * 2, f=81, model_batch=model_batch, signal_length=signal_length
        )
        self.LN = LayerNorm(ch * 2, f=81)
        self.f = 81
        self.f2 = 81 * 2
        self.model_batch = model_batch
        self.signal_length = signal_length

    def forward(self, x0, freq_initial_state, cepstral_dft_weight, cepstral_idft_weight):
        # Interleaved real/imaginary rows make both Fourier transforms one MatMul each.
        batch = self.model_batch if self.model_batch is not None else x0.shape[0]
        frames = self.signal_length if self.signal_length is not None else x0.shape[1]
        stft_pair = torch.matmul(cepstral_dft_weight, x0).reshape(
            batch, frames, self.f, self.ch * 2
        )
        stft_real, stft_imag = stft_pair.split(self.ch, dim=-1)
        lstm_output = self.ch_lstm_f(self.LN(stft_pair), freq_initial_state)
        processed_real, processed_imag = lstm_output.split(self.ch, dim=-1)
        out_real = processed_real * stft_real - processed_imag * stft_imag
        out_imag = processed_real * stft_imag + processed_imag * stft_real
        packed = torch.cat((out_real, out_imag), dim=2)
        return torch.matmul(cepstral_idft_weight, packed)


class LayerNorm(torch.nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.c = c
        self.f = f
        self.reduced = c * f
        self.w = torch.nn.Parameter(torch.ones(1, c, f, 1))
        self.b = torch.nn.Parameter(torch.rand(1, c, f, 1) * 1e-4)
        reduced = c * f
        self.register_buffer('mean_scale', torch.tensor(1.0 / float(reduced), dtype=torch.float32))
        weight_scale = float(max(reduced - 1, 1)) ** 0.5
        self.register_buffer('weight_scale', torch.tensor(weight_scale, dtype=torch.float32))
        self.register_buffer('norm_eps', torch.tensor(1e-6 * weight_scale * weight_scale, dtype=torch.float32))
        self.var_scale_fused = False
        self.export_ready = False

    def fuse_var_scale_(self):
        if self.var_scale_fused:
            return
        with torch.no_grad():
            self.w.mul_(self.weight_scale.to(dtype=self.w.dtype))
        self.var_scale_fused = True

    def prepare_for_export_(self):
        if not self.var_scale_fused:
            raise RuntimeError("Call fuse_var_scale_() after loading the checkpoint first.")
        self.register_buffer(
            'export_w',
            (self.w.detach().squeeze(0).squeeze(-1).transpose(0, 1) / float(self.reduced) ** 0.5).contiguous(),
        )
        self.register_buffer(
            'export_b',
            self.b.detach().squeeze(0).squeeze(-1).transpose(0, 1).contiguous(),
        )
        self.export_eps = 1e-6 * float(self.reduced - 1) / float(self.reduced)
        self.export_ready = True

    def forward(self, x):
        if not self.export_ready:
            raise RuntimeError("Call prepare_for_export_() before inference or export.")
        return torch.nn.functional.layer_norm(
            x, (self.f, self.c), self.export_w, self.export_b, self.export_eps
        )


class NET(torch.nn.Module):
    def __init__(self, order=10, channels=20, max_frames=2048, custom_istft=None):
        super().__init__()
        self.act = torch.nn.ELU()
        self.order = order
        self.custom_istft = custom_istft
        self.model_batch = STATIC_MODEL_BATCH
        self.signal_length = STATIC_SIGNAL_LENGTH

        # --- Model Layers ---
        self.in_ch_lstm = CH_LSTM_F(
            4, channels, channels, model_batch=self.model_batch, signal_length=self.signal_length
        )
        self.in_conv = torch.nn.Conv2d(in_channels=4 + channels, out_channels=channels, kernel_size=(1, 1))
        self.cfb_e1 = CFB(channels, channels, self.model_batch, self.signal_length)
        self.cfb_e2 = CFB(channels, channels, self.model_batch, self.signal_length)
        self.cfb_e3 = CFB(channels, channels, self.model_batch, self.signal_length)
        self.cfb_e4 = CFB(channels, channels, self.model_batch, self.signal_length)
        self.cfb_e5 = CFB(channels, channels, self.model_batch, self.signal_length)
        self.ln = LayerNorm(channels, 160)
        self.ch_lstm = CH_LSTM_T(
            in_ch=channels, feat_ch=channels * 2, out_ch=channels, num_layers=2,
            model_batch=self.model_batch, signal_length=self.signal_length,
        )
        self.cfb_d5 = CFB(1 * channels, channels, self.model_batch, self.signal_length)
        self.cfb_d4 = CFB(2 * channels, channels, self.model_batch, self.signal_length)
        self.cfb_d3 = CFB(2 * channels, channels, self.model_batch, self.signal_length)
        self.cfb_d2 = CFB(2 * channels, channels, self.model_batch, self.signal_length)
        self.cfb_d1 = CFB(2 * channels, channels, self.model_batch, self.signal_length)
        self.out_ch_lstm = CH_LSTM_T(
            2 * channels, channels, channels * 2,
            model_batch=self.model_batch, signal_length=self.signal_length,
        )
        self.out_conv = torch.nn.Conv2d(in_channels=channels * 3, out_channels=2, kernel_size=(1, 1), padding=(0, 0), bias=True)

        cep_n_fft = NFFT // 2 + 1
        cep_bins = cep_n_fft // 2 + 1
        t = torch.arange(cep_n_fft, dtype=torch.float32).unsqueeze(0)
        f = torch.arange(cep_bins, dtype=torch.float32).unsqueeze(1)
        omega = 2.0 * torch.pi * f * t / cep_n_fft
        cep_real = torch.cos(omega)
        cep_imag = -torch.sin(omega)
        cepstral_dft_weight = torch.stack((cep_real, cep_imag), dim=1).reshape(cep_bins * 2, cep_n_fft)
        fourier_basis = torch.fft.fft(torch.eye(cep_n_fft, dtype=torch.float32))
        inverse_cat = torch.linalg.pinv(torch.vstack((
            torch.real(fourier_basis[:cep_bins]), torch.imag(fourier_basis[:cep_bins])
        )).float()).T
        self.register_buffer('cepstral_dft_weight', cepstral_dft_weight)
        self.register_buffer('cepstral_idft_weight', inverse_cat.transpose(0, 1).contiguous())
        if self.model_batch is not None:
            self.register_buffer(
                'freq_initial_state',
                torch.zeros(2, self.model_batch * self.signal_length, channels),
            )
        else:
            self.freq_initial_state = None
        self.register_buffer('in_fused_weight', torch.empty(0), persistent=False)
        self.register_buffer('in_fused_bias', torch.empty(0), persistent=False)
        self.register_buffer('out_fused_weight', torch.empty(0), persistent=False)
        self.register_buffer('out_fused_bias', torch.empty(0), persistent=False)

    def prepare_for_export_(self):
        for module in self.modules():
            if isinstance(module, LayerNorm):
                module.prepare_for_export_()
        with torch.no_grad():
            in_conv_weight = self.in_conv.weight[:, :, 0, 0].double()
            in_linear = self.in_ch_lstm.linear
            in_projected = in_conv_weight[:, :in_linear.out_features]
            self.in_fused_weight = torch.cat((
                in_projected @ in_linear.weight.double(),
                in_conv_weight[:, in_linear.out_features:],
            ), dim=1).float().contiguous()
            self.in_fused_bias = (
                self.in_conv.bias.double() + in_projected @ in_linear.bias.double()
            ).float().contiguous()

            out_conv_weight = self.out_conv.weight[:, :, 0, 0].double()
            out_linear = self.out_ch_lstm.linear
            out_projected = out_conv_weight[:, :out_linear.out_features]
            self.out_fused_weight = torch.cat((
                out_projected @ out_linear.weight.double(),
                out_conv_weight[:, out_linear.out_features:],
            ), dim=1).float().contiguous()
            self.out_fused_bias = (
                self.out_conv.bias.double() + out_projected @ out_linear.bias.double()
            ).float().contiguous()

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        e0 = self.in_ch_lstm.forward_recurrent(x, self.freq_initial_state)
        e0 = torch.nn.functional.linear(
            torch.cat([e0, x], dim=-1), self.in_fused_weight, self.in_fused_bias
        )
        e1 = self.cfb_e1(e0, self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        e2 = self.cfb_e2(e1, self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        e3 = self.cfb_e3(e2, self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        e4 = self.cfb_e4(e3, self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        e5 = self.cfb_e5(e4, self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        lstm_out = self.ch_lstm(self.ln(e5))
        d5 = self.cfb_d5(e5 * lstm_out, self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        d4 = self.cfb_d4(torch.cat([e4, d5], dim=-1), self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        d3 = self.cfb_d3(torch.cat([e3, d4], dim=-1), self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        d2 = self.cfb_d2(torch.cat([e2, d3], dim=-1), self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        d1 = self.cfb_d1(torch.cat([e1, d2], dim=-1), self.freq_initial_state, self.cepstral_dft_weight, self.cepstral_idft_weight)
        d0 = self.out_ch_lstm.forward_recurrent(torch.cat([e0, d1], dim=-1))
        out = torch.nn.functional.linear(
            torch.cat([d0, d1], dim=-1), self.out_fused_weight, self.out_fused_bias
        ).transpose(1, 3)
        return self.custom_istft.forward_packed(out.reshape(
            self.model_batch, 2 * 160, self.signal_length
        ))


class CH_LSTM_T(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1, f=160, model_batch=1, signal_length=200):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=False, bidirectional=bi)
        self.bi = 1 if not bi else 2
        self.linear = torch.nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f
        self.model_batch = model_batch
        self.signal_length = signal_length
        if model_batch is not None:
            directions = 2 if bi else 1
            self.register_buffer(
                'initial_state',
                torch.zeros(num_layers * directions, model_batch * f, feat_ch),
            )
        else:
            self.initial_state = None

    def forward(self, x):
        return self.linear(self.forward_recurrent(x))

    def forward_recurrent(self, x):
        self.lstm2.flatten_parameters()
        frames = self.signal_length if self.signal_length is not None else x.shape[1]
        x = x.transpose(0, 1).contiguous().view(frames, -1, self.lstm2.input_size)
        if self.initial_state is None:
            x, _ = self.lstm2(x)
        else:
            x, _ = self.lstm2(x, (self.initial_state, self.initial_state))
        x = x.view(frames, -1, self.f, self.linear.in_features).transpose(0, 1).contiguous()
        return x


class CH_LSTM_F(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1, f=160, model_batch=1, signal_length=200):
        super().__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=False, bidirectional=bi)
        self.linear = torch.nn.Linear(2 * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f
        self.model_batch = model_batch
        self.signal_length = signal_length

    def forward(self, x, initial_state=None):
        return self.linear(self.forward_recurrent(x, initial_state))

    def forward_recurrent(self, x, initial_state=None):
        self.lstm2.flatten_parameters()
        frames = self.signal_length if self.signal_length is not None else x.shape[1]
        x = x.permute(2, 0, 1, 3).contiguous().view(self.f, -1, self.lstm2.input_size)
        if initial_state is None:
            x, _ = self.lstm2(x)
        else:
            x, _ = self.lstm2(x, (initial_state, initial_state))
        x = x.view(self.f, -1, frames, self.linear.in_features).permute(1, 2, 0, 3).contiguous()
        return x


class SDAEC(torch.nn.Module):
    def __init__(self, iccrn, alpha_predictor, custom_stft, nfft, k, max_len, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(SDAEC, self).__init__()
        self.iccrn = iccrn
        self.use_batch_fold = use_batch_fold          # Fold long near/far inputs into fixed windows batched on dim0
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.custom_stft = custom_stft
        self.input_pcm_scale_folded = "int" in IN_AUDIO_DTYPE.lower() and custom_stft.input_scale != 1.0
        self.output_pcm_scale_folded = (
            "int" in OUT_AUDIO_DTYPE.lower()
            and hasattr(iccrn.custom_istft, 'inv_win_sum')
            and iccrn.custom_istft.output_scale != 1.0
        )
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
        batch = STATIC_MODEL_BATCH if STATIC_MODEL_BATCH is not None else near_end_audio.shape[0]
        audio_pair = torch.cat([near_end_audio, far_end_audio], dim=0).float()
        if self.resample_before_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower() and not self.input_pcm_scale_folded:
            audio_pair = audio_pair * INV_INT16
        audio_pair = audio_pair - audio_pair.mean(dim=2, keepdim=True)
        if self.resample_after_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        pair_comp = self.custom_stft(audio_pair).view(
            2 * batch, 2, NFFT // 2 + 1, -1
        )
        frames = STATIC_SIGNAL_LENGTH if STATIC_SIGNAL_LENGTH is not None else pair_comp.shape[-1]
        frame_power = pair_comp.square().sum(dim=(1, 2)).view(2, batch, frames).transpose(0, 1)
        if STATIC_MODEL_BATCH is not None:
            mix_comp, far_comp = pair_comp.split(STATIC_MODEL_BATCH, dim=0)
        else:
            mix_comp, far_comp = pair_comp.chunk(2, dim=0)
        alpha_pad = self.alpha_power_pad if batch == 1 else self.alpha_power_pad.expand(batch, -1, -1)
        alpha_input = torch.cat([alpha_pad, frame_power], dim=-1)
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
        if "int" in OUT_AUDIO_DTYPE.lower() and not self.output_pcm_scale_folded:
            audio = audio * 32767.0
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
    custom_stft = STFT_Process(
        model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH,
        max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant",
        # Keep PCM scaling as an explicit FP32 safety island. Folding 1/32768 into
        # the DFT kernel leaves ~3e-5 weights that are poorly represented in FP16.
        input_scale=1.0,
        packed_output=True,
    ).eval()
    custom_istft = STFT_Process(
        model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH,
        max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant",
        static_norm=not DYNAMIC_AXES,
        # Do not fold 32767 into COLA normalization: edge reciprocals exceed five
        # million and are clipped by FP16 conversion. Scale the waveform afterward.
        output_scale=1.0,
        output_length=FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else int(round(EXPORT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)),
    ).eval()
    iccrn = NET(max_frames=MAX_SIGNAL_LENGTH, custom_istft=custom_istft)
    iccrn.load_state_dict(torch.load(project_path + '/Model/ICCRN.ckpt', map_location='cpu'), strict=False)
    for module in iccrn.modules():
        if isinstance(module, LayerNorm):
            module.fuse_var_scale_()
    iccrn.prepare_for_export_()
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
