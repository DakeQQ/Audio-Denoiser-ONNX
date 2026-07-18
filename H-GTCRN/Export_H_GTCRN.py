import gc
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


parent_path          = Path(__file__).resolve().parent                             # The folder that contains this script.
model_path           = str(Path.home() / "Downloads" / "H-GTCRN-main")            # The H-GTCRN download path.
onnx_model_A         = str(parent_path / "H_GTCRN_ONNX" / "H_GTCRN.onnx")          # The exported onnx model path.
onnx_model_Metadata  = str(metadata_path_for_model(onnx_model_A))                  # The metadata carrier onnx model path.


DYNAMIC_AXES         = False                          # False exports a fixed windowed model; set True to keep dynamic audio length so WPE/AuxIVA can use full-sequence statistics.
OPSET                = 20                             # ONNX opset.
IN_SAMPLE_RATE       = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE      = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
MODEL_SAMPLE_RATE    = 16000                          # The internal processing sample rate of the model. STFT/ISTFT, WPE/AuxIVA and the network always run at this rate; inputs are resampled to it.
INPUT_AUDIO_LENGTH   = 32000                          # Dummy export length when dynamic axes are enabled. Keep it as an integer multiple of HOP_LENGTH.
N_CHANNELS           = 2                              # Number of input microphone channels for the original WPE/AuxIVA stereo front-end.
WINDOW_TYPE          = 'hann'                         # Type of window function used in the STFT (matches original H-GTCRN torch.hann_window).
PAD_MODE             = 'reflect'                      # ['constant', 'reflect'] torch.stft defaults to 'reflect' (original repo uses the default), so match it.
NFFT                 = 512                            # Number of FFT components for the STFT process.
WINDOW_LENGTH        = 512                            # Length of windowing, edit it carefully.
HOP_LENGTH           = 256                            # Number of samples between successive frames in the STFT.
BATCH_WINDOW_SECONDS = 1.5                            # When the configured input audio length is >= this many seconds, the audio is folded into fixed-length windows and batch-processed together to accelerate inference. WPE/AuxIVA then run per window.
FOLD_WINDOW_LENGTH   = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window length (model-rate samples) for batch folding, rounded up to a multiple of HOP_LENGTH so every window reconstructs exactly through STFT -> ISTFT.
USE_BATCH_FOLD       = False                           # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
EXPORT_AUDIO_LENGTH  = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length: in fold mode it is rounded UP to a whole number of windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop.
MODEL_AUDIO_LENGTH   = int(EXPORT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE) if not DYNAMIC_AXES else 0  # Static model-rate waveform length after interpolation.
MAX_SIGNAL_LENGTH    = (FOLD_WINDOW_LENGTH // HOP_LENGTH + 1) if USE_BATCH_FOLD else (4096 if DYNAMIC_AXES else MODEL_AUDIO_LENGTH // HOP_LENGTH + 1)  # Max STFT frames (per-window count in fold mode). Sizes the WPE delay templates AND the ISTFT COLA trim.
FRONTEND_BATCH       = (MODEL_AUDIO_LENGTH // FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else 1  # WPE/AuxIVA batch: one item per folded window.
WPE_RT60             = 0.3                            # WPE reverberation time parameter.
WPE_DELAY            = 2                              # WPE prediction delay parameter.
WPE_ITER             = 1                              # WPE number of iterations.
IVA_ITER             = 10                             # AuxIVA number of iterations (must match training: 10 iterations for proper source separation).
CG_SOLVE_ITER        = 6                              # Inner CG steps for the WPE linear solve.

IN_AUDIO_DTYPE       = 'INT16'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'INT16'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16            = float(1.0 / 32768.0)
FOLD_INPUT_PCM_SCALE = False  # Keep PCM normalization before the DFT kernel for exact checkpointed output parity.
FOLD_OUTPUT_PCM_SCALE = False  # Reassociating COLA division with the non-power-of-two PCM scale can change int16 rounding.


def pad_audio_tail_with_context(audio: np.ndarray, target_length: int) -> np.ndarray:
    current_length = audio.shape[-1]
    if current_length >= target_length:
        return audio

    pad_amount = target_length - current_length
    if current_length == 0:
        padding = np.zeros((*audio.shape[:-1], pad_amount), dtype=audio.dtype)
    elif current_length == 1:
        padding = np.repeat(audio[..., -1:], pad_amount, axis=-1)
    else:
        padding = np.pad(audio, ((0, 0), (0, 0), (0, pad_amount)), mode='reflect')[..., current_length:]

    return np.concatenate((audio, padding.astype(audio.dtype, copy=False)), axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# H-GTCRN core model (inlined from modeling_modified/hgtcrn_optimized.py)
# ShuffleNetV2 + SFE + TRA + 2 DPGRNN (multi-channel, 6-channel input variant)
#   - Weight fusion: BatchNorm fused into Conv/ConvTranspose weights (fuse_bn_)
#   - Scale fusion: ERB weights pre-transposed as buffers (avoid runtime .T)
#   - Use split instead of chunk/slice/gather; constants pre-computed as buffers
# ═══════════════════════════════════════════════════════════════════════════

class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        self.nfreqs = nfreqs
        self.high_split = nfreqs - erb_subband_1
        # nn.Linear kept for state_dict compatibility; forward uses pre-transposed buffers
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)
        # Pre-transposed for direct matmul (no runtime .T)
        self.register_buffer('erb_weight_t', erb_filters.T.contiguous())
        self.register_buffer('ierb_weight_t', erb_filters.contiguous())

    def hz2erb(self, freq_hz):
        return 24.7 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 24.7) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                           / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
                                                       / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
                                                           / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B,C,T,F) -> (B,C,T, erb_subband_1 + erb_subband_2)"""
        x_low, x_high = x.split([self.erb_subband_1, self.high_split], dim=-1)
        return torch.cat([x_low, torch.matmul(x_high, self.erb_weight_t)], dim=-1)

    def bs(self, x_erb):
        """x: (B,C,T,F_erb) -> (B,C,T, nfreqs)"""
        x_erb_low, x_erb_high = x_erb.split([self.erb_subband_1, self.erb_subband_2], dim=-1)
        return torch.cat([x_erb_low, torch.matmul(x_erb_high, self.ierb_weight_t)], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, in_channels, kernel_size=3, stride=1, batch_size=1, n_frames=None, width=33):
        super().__init__()
        if kernel_size != 3 or stride != 1:
            raise ValueError("The optimized SFE requires kernel_size=3 and stride=1.")
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.width = width
        self.register_buffer(
            'zero_col',
            None if n_frames is None else torch.zeros(batch_size, in_channels, n_frames, 1),
        )

    def forward(self, x, zero_col=None):
        """x: (B,C,T,F) -> (B, C*kernel_size, T, F)"""
        static_zero = self.zero_col if zero_col is None else zero_col
        padded = (
            torch.nn.functional.pad(x, (1, 1, 0, 0))
            if static_zero is None
            else torch.cat([static_zero, x, static_zero], dim=-1)
        )
        neighborhoods = torch.stack(
            [
                padded[..., :self.width],
                padded[..., 1:self.width + 1],
                padded[..., 2:self.width + 2],
            ],
            dim=2,
        )
        frames = -1 if self.n_frames is None else self.n_frames
        return neighborhoods.reshape(
            self.batch_size,
            self.in_channels * self.kernel_size,
            frames,
            self.width,
        )


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)

    def forward(self, x, h0=None):
        """x: (B,C,T,F) -> (B,C,T,F)"""
        zt = x.square().mean(dim=-1).transpose(1, 2)
        gru_out = self.att_gru(zt, h0)[0] if h0 is not None else self.att_gru(zt)[0]
        at = torch.sigmoid(self.att_fc(gru_out)).transpose(1, 2).unsqueeze(-1)
        return x * at


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        self.is_deconv = use_deconv
        self.groups = groups
        self.out_channels = out_channels
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
        self.fused = False

    def fuse_bn_(self):
        """Fuse BatchNorm into Conv weights for inference."""
        if self.fused:
            return
        conv = self.conv
        bn = self.bn
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std

        if self.is_deconv:
            groups = self.groups
            out_per_group = self.out_channels // groups
            in_per_group = conv.in_channels // groups
            w = conv.weight.view(groups, in_per_group, out_per_group, conv.weight.shape[2], conv.weight.shape[3])
            fused_weight = (w * scale.view(groups, 1, out_per_group, 1, 1)).view_as(conv.weight)
        else:
            fused_weight = conv.weight * scale.view(-1, 1, 1, 1)

        fused_bias = bn.bias - bn.running_mean * scale if conv.bias is None else (conv.bias - bn.running_mean) * scale + bn.bias

        conv.weight = nn.Parameter(fused_weight)
        conv.bias = nn.Parameter(fused_bias)
        self.bn = nn.Identity()
        self.fused = True

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution - Optimized (ConvBlock wrappers for state_dict compatibility)"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, width=33, batch_size=1, n_frames=None):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        self.half_channels = in_channels // 2
        self.full_channels = in_channels
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.width = width

        self.sfe = SFE(
            self.half_channels,
            kernel_size=3,
            stride=1,
            batch_size=batch_size,
            n_frames=n_frames,
            width=width,
        )

        # Use ConvBlock wrappers to match checkpoint key paths:
        # point_conv1.conv.weight, point_conv1.bn.weight, point_conv1.act.weight, etc.
        self.point_conv1 = ConvBlock(in_channels // 2 * 3, hidden_channels, 1)
        self.depth_conv = ConvBlock(hidden_channels, hidden_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=hidden_channels)
        self.point_conv2 = ConvBlock(hidden_channels, in_channels // 2, 1)
        self.point_conv2.act = nn.Identity()  # No activation after point_conv2 (matches original)

        self.tra = TRA(in_channels // 2)

        # Pre-allocated static zero tensor for causal padding
        self.register_buffer('pad_zeros', torch.zeros(batch_size, hidden_channels, self.pad_size, width), persistent=False)

    def fuse_bn_(self):
        """Fuse all BatchNorm layers into their preceding Conv layers."""
        self.point_conv1.fuse_bn_()
        self.depth_conv.fuse_bn_()
        self.point_conv2.fuse_bn_()

    def forward(self, x, tra_h0=None, sfe_zero=None):
        """x: (B, C, T, F)"""
        x1, x2 = x.split(self.half_channels, dim=1)

        x1 = self.sfe(x1, sfe_zero)
        h1 = self.point_conv1(x1)
        h1 = torch.cat([self.pad_zeros, h1], dim=2)
        h1 = self.depth_conv(h1)
        h1 = self.point_conv2(h1)
        h1 = self.tra(h1, tra_h0)

        # ShuffleNet channel shuffle: interleave h1 and x2 along channel dim
        if self.n_frames is None:
            return torch.stack([h1, x2], dim=2).reshape(
                self.batch_size,
                self.full_channels,
                -1,
                self.width,
            )
        return torch.stack([h1, x2], dim=2).reshape(
            self.batch_size,
            self.full_channels,
            self.n_frames,
            self.width,
        )


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.half_hidden = hidden_size // 2
        self.half_input = input_size // 2
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.h_dim = num_layers * (2 if bidirectional else 1)
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h0=None):
        """x: (B, seq_length, input_size)"""
        x1, x2 = x.split(self.half_input, dim=-1)
        if h0 is None:
            y1, _ = self.rnn1(x1)
            y2, _ = self.rnn2(x2)
        else:
            y1, _ = self.rnn1(x1, h0)
            y2, _ = self.rnn2(x2, h0)
        return torch.cat([y1, y2], dim=-1)


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, batch_size=1, n_frames=None, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_frames = n_frames

        self.intra_rnn = GRNN(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=False,
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x, intra_h0=None, inter_h0=None):
        """x: (B, C, T, F)"""
        if self.n_frames is None:
            t = x.shape[2]
            batch_time = -1
            batch_freq = -1
        else:
            t = self.n_frames
            batch_time = self.batch_size * self.n_frames
            batch_freq = self.batch_size * self.width

        ## Intra RNN
        x_perm = x.permute(0, 2, 3, 1)                    # (B, T, F, C)
        intra_in = x_perm.reshape(batch_time, self.width, self.hidden_size)  # (B*T, F, C)
        intra_x = self.intra_fc(self.intra_rnn(intra_in, intra_h0))      # (B*T, F, C)
        intra_x = self.intra_ln(intra_x.reshape(self.batch_size, t, self.width, self.hidden_size))
        intra_out = x_perm + intra_x                       # (B, T, F, C)

        ## Inter RNN
        inter_in = intra_out.transpose(1, 2).reshape(batch_freq, t, self.hidden_size)  # (B*F, T, C)
        inter_x = self.inter_fc(self.inter_rnn(inter_in, inter_h0))      # (B*F, T, C)
        inter_x = self.inter_ln(inter_x.reshape(self.batch_size, self.width, t, self.hidden_size).transpose(1, 2))
        inter_out = intra_out + inter_x                    # (B, T, F, C)

        return inter_out.permute(0, 3, 1, 2)              # (B, C, T, F)


class Encoder(nn.Module):
    def __init__(self, batch_size=1, n_frames=None):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(6 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), batch_size=batch_size, n_frames=n_frames),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), batch_size=batch_size, n_frames=n_frames),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), batch_size=batch_size, n_frames=n_frames)
        ])

    def fuse_bn_(self):
        for conv in self.en_convs:
            conv.fuse_bn_()

    def forward(self, x, tra_h0=None, sfe_zero=None):
        e0 = self.en_convs[0](x)
        e1 = self.en_convs[1](e0)
        e2 = self.en_convs[2](e1, tra_h0, sfe_zero)
        e3 = self.en_convs[3](e2, tra_h0, sfe_zero)
        e4 = self.en_convs[4](e3, tra_h0, sfe_zero)
        return e4, (e0, e1, e2, e3, e4)


class Decoder(nn.Module):
    def __init__(self, batch_size=1, n_frames=None):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), batch_size=batch_size, n_frames=n_frames),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), batch_size=batch_size, n_frames=n_frames),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), batch_size=batch_size, n_frames=n_frames),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
        ])

    def fuse_bn_(self):
        for conv in self.de_convs:
            conv.fuse_bn_()

    def forward(self, x, en_outs, tra_h0=None, sfe_zero=None):
        x = self.de_convs[0](x + en_outs[4], tra_h0, sfe_zero)
        x = self.de_convs[1](x + en_outs[3], tra_h0, sfe_zero)
        x = self.de_convs[2](x + en_outs[2], tra_h0, sfe_zero)
        x = self.de_convs[3](x + en_outs[1])
        x = self.de_convs[4](x + en_outs[0])
        return x


class GTCRN_IVA(nn.Module):
    """
    H-GTCRN core neural network (ERB + SFE + Encoder + DPGRNN + Decoder + CRM).
    6-channel input: [ref_real, ref_imag, ch2_real, ch2_imag, sel_log_mag, unsel_log_mag]

    Input:  spec_features (1, 6, T, F=257) float32
    Output: m (1, 2, T, F=257) float32 — complex ratio mask
    """
    def __init__(self, batch_size=1, n_frames=None):
        super().__init__()
        self.n_fft = 512
        self.hop_len = 256
        self.win_len = 512

        self.erb = ERB(65, 64)
        self.sfe = SFE(6, 3, 1, batch_size=batch_size, n_frames=n_frames, width=129)
        self.encoder = Encoder(batch_size=batch_size, n_frames=n_frames)
        self.dpgrnn1 = DPGRNN(16, 33, 16, batch_size=batch_size, n_frames=n_frames)
        self.dpgrnn2 = DPGRNN(16, 33, 16, batch_size=batch_size, n_frames=n_frames)
        self.decoder = Decoder(batch_size=batch_size, n_frames=n_frames)
        self.register_buffer(
            'gt_sfe_zero',
            None if n_frames is None else torch.zeros(batch_size, 8, n_frames, 1),
        )
        self.register_buffer('tra_h0', torch.zeros(1, batch_size, 16))
        self.register_buffer(
            'intra_h0',
            None if n_frames is None else torch.zeros(2, batch_size * n_frames, 4),
        )
        self.register_buffer('inter_h0', torch.zeros(1, batch_size * 33, 8))

    def fuse_bn_(self):
        """Fuse all BatchNorm layers into Conv weights for inference."""
        self.encoder.fuse_bn_()
        self.decoder.fuse_bn_()

    def forward(self, spec_features):
        """
        spec_features: (1, 6, T, F=257)
        Returns: s_real (1, F, T), s_imag (1, F, T)
        """
        # ERB band mapping
        feat = self.erb.bm(spec_features)        # (1, 6, T, 129)
        # Subband Feature Extraction
        feat = self.sfe(feat)                    # (1, 18, T, 129)
        # Encoder
        feat, en_outs = self.encoder(feat, self.tra_h0, self.gt_sfe_zero)
        # Dual-path Grouped RNN
        feat = self.dpgrnn1(feat, self.intra_h0, self.inter_h0)  # (1, 16, T, 33)
        feat = self.dpgrnn2(feat, self.intra_h0, self.inter_h0)  # (1, 16, T, 33)
        # Decoder
        m_feat = self.decoder(feat, en_outs, self.tra_h0, self.gt_sfe_zero)  # (1, 2, T, F_erb=129)
        # ERB band synthesis — keep in (T,F) layout to avoid transposing large tensors
        m = self.erb.bs(m_feat)                  # (1, 2, T, F=257)

        # Split mask and ref in native (T,F) layout (no transpose needed)
        m_real, m_imag = m.split(1, dim=1)                             # each (1, 1, T, F)
        ref_real, ref_imag, _ = spec_features.split([1, 1, 4], dim=1)  # each ref: (1, 1, T, F)

        # CRM in (T,F) layout, then transpose only the smaller output tensors to (F,T)
        s_real = (ref_real * m_real - ref_imag * m_imag).squeeze(1).transpose(-1, -2)  # (1, F, T)
        s_imag = (ref_imag * m_real + ref_real * m_imag).squeeze(1).transpose(-1, -2)  # (1, F, T)

        return s_real, s_imag


# ═══════════════════════════════════════════════════════════════════════════
# ONNX-friendly complex arithmetic helpers (real-valued representation)
# Complex tensors are represented as separate real/imag tensors or (..., 2) pairs
# ═══════════════════════════════════════════════════════════════════════════

def batched_complex_solve_cg(R_r, R_i, P_r, P_i, zero, n_iter=36):
    """
    Solve R @ G = P for G using Conjugate Gradient (CG).
    ONNX-friendly: uses only matmul, element-wise ops, and reductions.
    Guaranteed to converge for Hermitian positive definite R.

    R: (B, F, N, N) complex Hermitian positive definite (R_r + j*R_i)
    P: (B, F, N, M) complex (P_r + j*P_i)
    Returns: G_r, G_i of same shape as P
    """
    # Initialize: x=0, r=P, p=r
    x_r = zero
    x_i = zero
    r_r = P_r
    r_i = P_i
    p_r = P_r
    p_i = P_i

    # rr = sum_n |r_n|^2 per (B, F, M) = r^H @ r per column
    rr = (r_r * r_r + r_i * r_i).sum(dim=-2) + 1e-12  # (B, F, M)

    for _ in range(n_iter):
        # Ap = R @ p (complex matmul)
        Ap_r = torch.matmul(R_r, p_r) - torch.matmul(R_i, p_i)  # (B, F, N, M)
        Ap_i = torch.matmul(R_r, p_i) + torch.matmul(R_i, p_r)

        # pAp = p^H @ Ap = sum_n conj(p_n)*Ap_n per column (real for HPD R)
        # For HPD R, p^H R p is real and positive
        pAp = (p_r * Ap_r + p_i * Ap_i).sum(dim=-2) + 1e-12  # (B, F, M)

        # alpha = rr / pAp
        alpha = rr / pAp # (B, F, M)
        alpha_expanded = alpha.unsqueeze(-2)  # (B, F, 1, M)

        # x = x + alpha * p
        x_r = x_r + alpha_expanded * p_r
        x_i = x_i + alpha_expanded * p_i

        # r = r - alpha * Ap
        r_r = r_r - alpha_expanded * Ap_r
        r_i = r_i - alpha_expanded * Ap_i

        # rr_new = r^H @ r
        rr_new = (r_r * r_r + r_i * r_i).sum(dim=-2) + 1e-12  # (B, F, M)

        # beta = rr_new / rr
        beta = rr_new / rr  # (B, F, M)
        beta_expanded = beta.unsqueeze(-2)  # (B, F, 1, M)

        # p = r + beta * p
        p_r = r_r + beta_expanded * p_r
        p_i = r_i + beta_expanded * p_i

        rr = rr_new

    return x_r, x_i


def solve_2x2_complex(A, b, batch_size, n_freq_bins):
    """
    Solve A @ x = b for a 2x2 complex A using Cramer's rule.

    This deliberately preserves the baseline operation order. A unit-vector
    specialization is algebraically equivalent and bit-exact in PyTorch, but
    its raw ORT graph diverges for ill-conditioned sparse inputs after several
    AuxIVA iterations.
    """
    # Flatten only the two matrix axes. Keeping a singleton coefficient axis
    # lets every scalar expression broadcast naturally without Squeeze nodes.
    a, b_mat, c, d = A.reshape(batch_size, n_freq_bins, 4, 2).split(1, dim=-2)

    a_r, a_i = a.split(1, dim=-1)
    b_mat_r, b_mat_i = b_mat.split(1, dim=-1)
    c_r, c_i = c.split(1, dim=-1)
    d_r, d_i = d.split(1, dim=-1)

    # det = a*d - b*c (complex)
    ad = torch.cat([a_r * d_r - a_i * d_i, a_r * d_i + a_i * d_r], dim=-1)
    bc = torch.cat([b_mat_r * c_r - b_mat_i * c_i, b_mat_r * c_i + b_mat_i * c_r], dim=-1)
    det = ad - bc
    det_r, det_i = det.split(1, dim=-1)
    det_abs_sq = 1.0 / ((det_r ** 2 + det_i ** 2) + 1e-12)
    inv_det_r = det_r * det_abs_sq
    inv_det_i = -det_i * det_abs_sq

    b0, b1 = b.split(1, dim=-2)
    b0_r, b0_i = b0.split(1, dim=-1)
    b1_r, b1_i = b1.split(1, dim=-1)

    num0_r = (d_r * b0_r - d_i * b0_i) - (b_mat_r * b1_r - b_mat_i * b1_i)
    num0_i = (d_r * b0_i + d_i * b0_r) - (b_mat_r * b1_i + b_mat_i * b1_r)
    num1_r = (a_r * b1_r - a_i * b1_i) - (c_r * b0_r - c_i * b0_i)
    num1_i = (a_r * b1_i + a_i * b1_r) - (c_r * b0_i + c_i * b0_r)

    x0 = torch.cat([num0_r * inv_det_r - num0_i * inv_det_i,
                    num0_r * inv_det_i + num0_i * inv_det_r], dim=-1)
    x1 = torch.cat([num1_r * inv_det_r - num1_i * inv_det_i,
                    num1_r * inv_det_i + num1_i * inv_det_r], dim=-1)
    return torch.cat([x0, x1], dim=-2)


class OnnxFriendlyWPE(torch.nn.Module):
    """
    ONNX-exportable Weighted Prediction Error (WPE) dereverberation.
    Replaces torch.linalg.inv with complex Conjugate Gradient.

    Optimizations:
      - Pre-compute: eye matrix and delay templates registered as buffers
      - Hoist loop-invariant: Xp transpose computed once outside iteration loop
    """
    def __init__(self, n_channels=2, rt60=0.3, hop_length=256, delay=2, sample_rate=16000, num_iter=1, ns_iter=36,
            n_freq_bins=NFFT // 2 + 1, max_frames=MAX_SIGNAL_LENGTH, batch_size=1, dynamic_frames=False):
        super().__init__()
        self.M = n_channels
        self.Lg = int(rt60 * sample_rate / hop_length)
        self.D = delay
        self.num_iter = num_iter
        self.solve_iter = ns_iter
        self.MLg = self.M * self.Lg
        self.n_freq_bins = n_freq_bins
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.dynamic_frames = dynamic_frames
        # Pre-compute identity matrix as buffer (avoids runtime torch.eye allocation)
        self.register_buffer('eye_MLg', torch.eye(self.MLg, dtype=torch.float32))

        # Static delay-bank construction. Each output position gathers its delayed
        # source frame once; invalid leading positions are selected as exact zeros.
        frame_ids = torch.arange(max_frames, dtype=torch.int32).unsqueeze(0)
        lag_offsets = delay + torch.arange(self.Lg, dtype=torch.int32).unsqueeze(1)
        delay_indices = frame_ids - lag_offsets
        self.register_buffer('delay_indices', delay_indices.clamp_min(0).reshape(-1))
        self.register_buffer('delay_valid', (delay_indices >= 0).view(1, 1, self.Lg, 1, max_frames))
        self.register_buffer(
            'cg_zero',
            torch.zeros(batch_size, n_freq_bins, self.MLg, n_channels, dtype=torch.float32),
        )

    def forward(self, X_real, X_imag):
        """
        X_real, X_imag: (B, M, F, T) — multi-channel STFT real/imag parts.
        Returns: Y_real, Y_imag of same shape — dereverberated.
        """
        # Permute to (B, F, M, T)
        Xp_r = X_real.permute(0, 2, 1, 3)  # (B, F, M, T)
        Xp_i = X_imag.permute(0, 2, 1, 3)

        # Build delay matrix: (B, F, M*Lg, T)
        if self.dynamic_frames:
            T = X_real.shape[-1]
            delayed_r = []
            delayed_i = []
            for l_idx in range(self.Lg):
                shift = self.D + l_idx
                delayed_r.append(torch.nn.functional.pad(Xp_r[..., :-shift], (shift, 0)).unsqueeze(2))
                delayed_i.append(torch.nn.functional.pad(Xp_i[..., :-shift], (shift, 0)).unsqueeze(2))
            X_delay_r = torch.cat(delayed_r, dim=2).reshape(
                self.batch_size,
                self.n_freq_bins,
                self.MLg,
                T,
            )
            X_delay_i = torch.cat(delayed_i, dim=2).reshape(
                self.batch_size,
                self.n_freq_bins,
                self.MLg,
                T,
            )
        else:
            T = self.max_frames
            gathered_r = torch.index_select(Xp_r, -1, self.delay_indices).reshape(
                self.batch_size,
                self.n_freq_bins,
                self.M,
                self.Lg,
                self.max_frames,
            ).permute(0, 1, 3, 2, 4)
            gathered_i = torch.index_select(Xp_i, -1, self.delay_indices).reshape(
                self.batch_size,
                self.n_freq_bins,
                self.M,
                self.Lg,
                self.max_frames,
            ).permute(0, 1, 3, 2, 4)
            X_delay_r = torch.where(self.delay_valid, gathered_r, 0.0).reshape(
                self.batch_size,
                self.n_freq_bins,
                self.MLg,
                self.max_frames,
            )
            X_delay_i = torch.where(self.delay_valid, gathered_i, 0.0).reshape(
                self.batch_size,
                self.n_freq_bins,
                self.MLg,
                self.max_frames,
            )

        # Compute eps matching original: 1e-3 * mean_over_F(max_over(M,T)(|X|^2)).
        # Keep it PER-WINDOW (per batch element) so folded windows stay independent
        # (batched result == per-window loop); for batch=1 this equals the original scalar.
        mag_sq = Xp_r * Xp_r + Xp_i * Xp_i  # (B, F, M, T)
        eps_val = (1e-3 * mag_sq.amax(dim=(-2, -1)).mean(dim=-1)).reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # Y = Xp initially
        Y_r = Xp_r  # (B, F, M, T)
        Y_i = Xp_i

        # Hoist loop-invariant: Xp transposed doesn't change across iterations
        Xp_rT = Xp_r.transpose(-2, -1)  # (B, F, T, M)
        Xp_iT = Xp_i.transpose(-2, -1)

        for _ in range(self.num_iter):
            # lambda = mean(|Y|^2, dim=channels), shape (B, F, 1, T)
            Y_pow = (Y_r * Y_r + Y_i * Y_i).mean(dim=2, keepdim=True).clamp(min=eps_val)  # (B, F, 1, T)

            # temp = X_delay / lambda: (B, F, MLg, T)
            inv_lambda = 1.0 / Y_pow  # (B, F, 1, T)
            temp_r = X_delay_r * inv_lambda
            temp_i = X_delay_i * inv_lambda

            # R = temp @ conj(X_delay)^H: (B, F, MLg, MLg)
            Xd_rT = X_delay_r.transpose(-2, -1)  # (B, F, T, MLg)
            Xd_iT = X_delay_i.transpose(-2, -1)
            R_real = torch.matmul(temp_r, Xd_rT) + torch.matmul(temp_i, Xd_iT)
            R_imag = torch.matmul(temp_i, Xd_rT) - torch.matmul(temp_r, Xd_iT)

            # P = temp @ conj(Xp)^H: (B, F, MLg, M)
            P_real = torch.matmul(temp_r, Xp_rT) + torch.matmul(temp_i, Xp_iT)
            P_imag = torch.matmul(temp_i, Xp_rT) - torch.matmul(temp_r, Xp_iT)

            # Add eps * I to R for regularization (use pre-computed buffer)
            R_real = R_real + eps_val * self.eye_MLg

            # Solve R @ G = P using Conjugate Gradient.
            G_r, G_i = batched_complex_solve_cg(
                R_real,
                R_imag,
                P_real,
                P_imag,
                self.cg_zero,
                n_iter=self.solve_iter,
            )

            # Y = Xp - conj(G)^T @ X_delay
            G_conj_T_real = G_r.transpose(-2, -1)   # (B, F, M, MLg)
            G_conj_T_imag = -G_i.transpose(-2, -1)  # (B, F, M, MLg)

            pred_r = torch.matmul(G_conj_T_real, X_delay_r) - torch.matmul(G_conj_T_imag, X_delay_i)
            pred_i = torch.matmul(G_conj_T_imag, X_delay_r) + torch.matmul(G_conj_T_real, X_delay_i)

            Y_r = Xp_r - pred_r
            Y_i = Xp_i - pred_i

        # Permute back to (B, M, F, T)
        return Y_r.permute(0, 2, 1, 3), Y_i.permute(0, 2, 1, 3)


class OnnxFriendlyAuxIVA(torch.nn.Module):
    """
    ONNX-exportable AuxIVA source separation for 2-channel input.
    Replaces torch.linalg.solve with analytical 2x2 complex solve.

    Optimizations:
      - Pre-compute static demixing state as immutable buffers
      - Hoist: X transpose computed once outside iteration loop
      - Functional row updates avoid mutation-driven ScatterND graphs
    """
    def __init__(self, n_iter=10, n_channels=2, n_freq_bins=NFFT // 2 + 1, batch_size=1, n_frames=MAX_SIGNAL_LENGTH):
        super().__init__()
        self.n_iter = n_iter
        self.M = n_channels
        self.n_freq_bins = n_freq_bins
        self.batch_size = batch_size
        self.n_frames = n_frames
        # Pre-computed constants as buffers
        eye_M = torch.eye(n_channels, dtype=torch.float32)
        self.register_buffer('proj_back_one', torch.ones(1, 1, dtype=torch.float32))
        self.register_buffer('proj_back_zero', torch.zeros(1, 1, dtype=torch.float32))
        e_s_all = torch.zeros(n_channels, batch_size, n_freq_bins, n_channels, 2)
        for source in range(n_channels):
            e_s_all[source, :, :, source, 0] = 1.0
        self.register_buffer('e_s', e_s_all)
        self.eps = 1e-10
        self.register_buffer(
            'eps_eye',
            (self.eps * eye_M).view(1, 1, n_channels, n_channels),
        )
        self.register_buffer(
            'init_W_r',
            eye_M.view(1, 1, n_channels, n_channels).expand(batch_size, n_freq_bins, n_channels, n_channels).clone(),
        )
        self.register_buffer(
            'init_W_i',
            torch.zeros(batch_size, n_freq_bins, n_channels, n_channels, dtype=torch.float32),
        )

    def forward(self, X_real, X_imag):
        """
        X_real, X_imag: (B, M=2, F, T) — dereverberated STFT.
        Returns: Y_real, Y_imag (B, M=2, F, T) — separated sources.
        """
        inv_T = (1.0 / X_real.shape[-1]) if self.n_frames is None else (1.0 / self.n_frames)

        # Reshape to (B, F, M, T) for processing
        X_r = X_real.permute(0, 2, 1, 3)  # (B, F, M, T)
        X_i = X_imag.permute(0, 2, 1, 3)

        # Hoist loop-invariant: X transposed (constant across all iterations)
        X_rT = X_r.transpose(-2, -1)  # (B, F, T, M)
        X_iT = X_i.transpose(-2, -1)

        # Keep each demixing row as an SSA tensor. Python list replacement below
        # updates graph references, not tensor storage, so export emits no ScatterND.
        W_rows_r = list(self.init_W_r.split(1, dim=2))
        W_rows_i = list(self.init_W_i.split(1, dim=2))

        # Y = W @ X (W starts as identity, so Y = X initially)
        Y_r = X_r
        Y_i = X_i

        for iter_idx in range(self.n_iter):
            # r = 2 * L2_norm(Y over F): (B, M, T)
            Y_pow = Y_r * Y_r + Y_i * Y_i  # (B, F, M, T)
            r = 2.0 * torch.sqrt(Y_pow.sum(dim=1) + self.eps)  # (B, M, T)
            r_inv = 1.0 / r  # (B, M, T)
            r_inv_sources = r_inv.split(1, dim=1)

            for s in range(self.M):
                # r_inv for source s: (B, 1, 1, T)
                w_s = r_inv_sources[s].unsqueeze(1)  # (B, 1, 1, T)

                # weighted X: (B, F, M, T)
                wX_r = X_r * w_s
                wX_i = X_i * w_s

                # V = wX @ conj(X)^H / T: (B, F, M, M) complex
                V_r = (torch.matmul(wX_r, X_rT) + torch.matmul(wX_i, X_iT)) * inv_T
                V_i = (torch.matmul(wX_i, X_rT) - torch.matmul(wX_r, X_iT)) * inv_T

                # On the very first source-step, W starts as the identity and W_i is zero.
                if iter_idx == 0 and s == 0:
                    WV_r = V_r
                    WV_i = V_i
                else:
                    W_r = torch.cat(W_rows_r, dim=2)
                    W_i = torch.cat(W_rows_i, dim=2)
                    # WV = W @ V: (B, F, M, M)
                    WV_r = torch.matmul(W_r, V_r) - torch.matmul(W_i, V_i)
                    WV_i = torch.matmul(W_r, V_i) + torch.matmul(W_i, V_r)

                # Solve (WV + eps*I) @ w_new = e_s using pre-computed buffers
                WV_r_reg = WV_r + self.eps_eye

                A_solve = torch.stack([WV_r_reg, WV_i], dim=-1)
                w_new = solve_2x2_complex(
                    A_solve,
                    self.e_s[s],
                    self.batch_size,
                    self.n_freq_bins,
                )
                w_new_r, w_new_i = w_new.split(1, dim=-1)

                # W[s] = conj(w_new)
                conj_w_r = w_new_r   # (B, F, M, 1)
                conj_w_i = -w_new_i

                # Normalize: denom = conj(w) @ V @ w
                Vw_r = torch.matmul(V_r, w_new_r) - torch.matmul(V_i, w_new_i)
                Vw_i = torch.matmul(V_r, w_new_i) + torch.matmul(V_i, w_new_r)

                denom_r = (conj_w_r * Vw_r - conj_w_i * Vw_i).sum(dim=-2, keepdim=True)

                # For HPD V, w^H V w is real and non-negative; normalizing with the
                # real scalar avoids phase drift from a noisy complex sqrt.
                norm_scale = torch.rsqrt(denom_r.clamp(min=0.0) + self.eps)
                final_r = conj_w_r * norm_scale
                final_i = conj_w_i * norm_scale

                # Update one functional row; no tensor mutation reaches ONNX.
                W_rows_r[s] = final_r.reshape(self.batch_size, self.n_freq_bins, 1, self.M)
                W_rows_i[s] = final_i.reshape(self.batch_size, self.n_freq_bins, 1, self.M)

            # Recompute Y = W @ X
            W_r = torch.cat(W_rows_r, dim=2)
            W_i = torch.cat(W_rows_i, dim=2)
            Y_r = torch.matmul(W_r, X_r) - torch.matmul(W_i, X_i)
            Y_i = torch.matmul(W_r, X_i) + torch.matmul(W_i, X_r)

        # Projection back to align with reference channel (channel 0)
        ref_r, _ = X_r.split(1, dim=2)  # (B, F, 1, T), broadcast over sources
        ref_i, _ = X_i.split(1, dim=2)
        num_r = (ref_r * Y_r + ref_i * Y_i).sum(dim=-1)
        num_i = (ref_r * Y_i - ref_i * Y_r).sum(dim=-1)
        denom = (Y_r * Y_r + Y_i * Y_i).sum(dim=-1)
        valid = denom > 0.0
        safe_denom = 1.0 / torch.where(valid, denom, self.proj_back_one)
        c_r = torch.where(valid, num_r * safe_denom, self.proj_back_one).unsqueeze(-1)
        c_i = torch.where(valid, num_i * safe_denom, self.proj_back_zero).unsqueeze(-1)
        Y_out_r = c_r * Y_r + c_i * Y_i
        Y_out_i = c_r * Y_i - c_i * Y_r

        return Y_out_r.permute(0, 2, 1, 3), Y_out_i.permute(0, 2, 1, 3)


class H_GTCRN_CUSTOM(torch.nn.Module):
    """
    Fully fused H-GTCRN pipeline for end-to-end ONNX export:
      int16 2-channel audio -> STFT -> WPE -> AuxIVA -> feature construction -> GTCRN -> CRM -> iSTFT -> int16 mono audio.

    All preprocessing (WPE, AuxIVA) is implemented with ONNX-friendly operators
    (no linalg.inv/solve — uses Jacobi iteration and analytical 2x2 Cramer's rule).

    Optimizations:
      - Weight fusion: BN already fused in GTCRN via fuse_bn_()
      - Scale fusion: inv_int16 / output_pcm_scale as registered buffers
      - Reduce dim changes: view instead of squeeze+unsqueeze; single transpose for feature layout
      - Use split instead of slice for channel extraction
      - Pre-compute: n_freq_bins constant avoids runtime .shape; log10 fused with sqrt
      - Fuse concat+transpose: build all 6 features in (F,T) then single transpose to (T,F)

    Input:  noisy_audio (1, N_CHANNELS, audio_len) int16
    Output: denoised_audio (1, 1, audio_len) int16
    """
    def __init__(self, gtcrn_core, stft_model, istft_model, wpe_module, iva_module, n_fft=512, in_sample_rate=16000, out_sample_rate=16000, use_batch_fold=False, fold_window=0, model_audio_length=0, n_frames=None, frontend_batch=1, fold_input_pcm_scale=False, fold_output_pcm_scale=False):
        super(H_GTCRN_CUSTOM, self).__init__()
        self.gtcrn = gtcrn_core
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.wpe = wpe_module
        self.iva = iva_module
        # Pre-computed constants as buffers (no runtime computation)
        self.register_buffer('inv_int16', torch.tensor([INV_INT16], dtype=torch.float32))
        self.register_buffer('output_pcm_scale', torch.tensor(32767.0))
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
        self.n_freq_bins = n_fft // 2 + 1  # Avoid runtime .shape query for F dimension
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.model_audio_length = model_audio_length
        self.n_frames = n_frames
        self.frontend_batch = frontend_batch
        self.fold_input_pcm_scale = fold_input_pcm_scale
        self.fold_output_pcm_scale = fold_output_pcm_scale

    def forward(self, audio):
        """
        audio: (1, 2, audio_len) int16 — 2-channel raw audio
        """
        # ─── 1. Resample to the 16 kHz model rate and normalize ──────────
        audio_f = audio.float()  # (1, 2, L)
        if self.resample_before_centering:
            audio_f = torch.nn.functional.interpolate(
                audio_f,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower() and not self.fold_input_pcm_scale:
            audio_f = audio_f * self.inv_int16      # int16 PCM -> [-1, 1]; F16/F32 inputs already arrive normalized.
        audio_f = audio_f - torch.mean(audio_f)
        if self.resample_after_centering:
            audio_f = torch.nn.functional.interpolate(
                audio_f,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # ─── 2. Multi-channel STFT via Conv1d ─────────────────────────────
        if self.use_batch_fold:
            # Fold (1, 2, num_window*W) -> (num_window*2, 1, W): window-major, channel-minor,
            # so the STFT output rows regroup cleanly to (num_window, 2, F, T) for WPE/AuxIVA.
            stft_in = audio_f.reshape(2, self.frontend_batch, self.fold_window).transpose(0, 1).reshape(
                self.frontend_batch * 2,
                1,
                self.fold_window,
            )
        else:
            # view replaces squeeze(0)+unsqueeze(1): (1,2,L) -> (2,1,L) in one op
            stft_in = audio_f.reshape(2, 1, self.model_audio_length) if self.model_audio_length else audio_f.reshape(2, 1, -1)
        real_parts, imag_parts = self.stft_model(stft_in)  # each (B*2, F, T)  [B=num_window fold / 1 non-fold]
        n_frames = real_parts.shape[-1] if self.n_frames is None else self.n_frames

        # Regroup once and reuse in WPE plus network feature construction.
        stft_real = real_parts.reshape(self.frontend_batch, 2, self.n_freq_bins, n_frames)
        stft_imag = imag_parts.reshape(self.frontend_batch, 2, self.n_freq_bins, n_frames)

        # ─── 3. WPE dereverberation ──────────────────────────────────────
        # Regroup channels for the multi-channel front-end: (B*2, F, T) -> (B, 2, F, T)
        drb_real, drb_imag = self.wpe(stft_real, stft_imag)

        # ─── 4. AuxIVA source separation ─────────────────────────────────
        iva_real, iva_imag = self.iva(drb_real, drb_imag)  # each (B, 2, F, T)

        # ─── 5. Compute both source powers once, then reorder logs ─────────
        iva_power = iva_real * iva_real + iva_imag * iva_imag
        energy = iva_power.sum(dim=(2, 3))  # (B, 2)
        energy_0, energy_1 = energy.split(1, dim=1)  # each (B, 1)
        pred = (energy_0 < energy_1).reshape(self.frontend_batch, 1, 1, 1)

        # ─── 6. Fused log-magnitude: log10(sqrt(x)) = 0.5*log10(x) ──────
        # Original clamps the MAGNITUDE at 1e-12 (norm(...).clamp(1e-12)). Because sqrt is
        # monotonic, clamp(sqrt(x),1e-12) == sqrt(clamp(x,1e-24)), so clamp the squared
        # magnitude at (1e-12)^2 = 1e-24 to stay bit-exact with the original feature floor.
        log_magnitude = 0.5 * torch.log10(iva_power.clamp(min=1e-24))
        log_0, log_1 = log_magnitude.split(1, dim=1)
        sel_log = torch.where(pred, log_0, log_1)
        unsel_log = torch.where(pred, log_1, log_0)

        # ─── 7. Feature construction: one final allocation + transpose ───
        # Ordering: [ch0_real, ch0_imag, ch1_real, ch1_imag, selected_log, other_log].
        real_0, real_1 = stft_real.split(1, dim=1)
        imag_0, imag_1 = stft_imag.split(1, dim=1)
        spec_features = torch.cat(
            [real_0, imag_0, real_1, imag_1, sel_log, unsel_log],
            dim=1,
        ).transpose(-1, -2)  # (B, 6, T, F)

        # ─── 8. GTCRN network (ERB + Encoder + DPGRNN + Decoder + CRM) ───
        s_real, s_imag = self.gtcrn(spec_features)  # each (B, F, T)

        # ─── 9. iSTFT -> time-domain audio ───────────────────────────────
        audio_out = self.istft_model(s_real, s_imag)  # (B, 1, W)
        if self.use_batch_fold:
            audio_out = audio_out.reshape(1, 1, self.model_audio_length)   # stitch windows back

        # ─── 10. Resample output and scale to int16 PCM ──────────────────
        if self.output_resample_before_pcm:
            audio_out = torch.nn.functional.interpolate(
                audio_out,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower() and not self.fold_output_pcm_scale:
            audio_out = audio_out * self.output_pcm_scale      # [-1, 1] -> int16 PCM; F16/F32 outputs stay normalized.
        if self.output_resample_after_pcm:
            audio_out = torch.nn.functional.interpolate(
                audio_out,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        # The WPE/AuxIVA front-end can produce NaN for a fully silent window. Replace NaN
        # explicitly; the clamp below maps +/-Inf to the same limits as nan_to_num. This is
        # numerically equivalent at the int16 output and avoids the exporter's two
        # float -> double -> IsInf side chains, which otherwise add needless mixed-precision
        # Cast nodes.
        audio_out = torch.where(torch.isnan(audio_out), torch.zeros_like(audio_out), audio_out)
        if "int" in OUT_AUDIO_DTYPE.lower():
            return audio_out.clamp(-32768.0, 32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return audio_out
        return audio_out.to(torch.float16)




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_H_GTCRN_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
with torch.inference_mode():
    static_n_frames = None if DYNAMIC_AXES else MAX_SIGNAL_LENGTH
    custom_stft = STFT_Process(
        model_type='stft_B',
        n_fft=NFFT,
        hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        max_frames=0,
        window_type=WINDOW_TYPE,
        center_pad=True,
        pad_mode=PAD_MODE,
        input_scale=INV_INT16 if FOLD_INPUT_PCM_SCALE else 1.0,
    ).eval()
    custom_istft = STFT_Process(
        model_type='istft_B',
        n_fft=NFFT,
        hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        max_frames=MAX_SIGNAL_LENGTH,
        window_type=WINDOW_TYPE,
        center_pad=True,
        pad_mode=PAD_MODE,
        output_scale=32767.0 if FOLD_OUTPUT_PCM_SCALE else 1.0,
        static_cola=not DYNAMIC_AXES,
    ).eval()
    wpe_module = OnnxFriendlyWPE(
        n_channels=N_CHANNELS,
        rt60=WPE_RT60,
        hop_length=HOP_LENGTH,
        delay=WPE_DELAY,
        sample_rate=MODEL_SAMPLE_RATE,
        num_iter=WPE_ITER,
        ns_iter=CG_SOLVE_ITER,
        n_freq_bins=NFFT // 2 + 1,
        max_frames=MAX_SIGNAL_LENGTH,
        batch_size=FRONTEND_BATCH,
        dynamic_frames=DYNAMIC_AXES,
    ).eval()
    iva_module = OnnxFriendlyAuxIVA(
        n_iter=IVA_ITER,
        n_channels=N_CHANNELS,
        batch_size=FRONTEND_BATCH,
        n_frames=static_n_frames,
    ).eval()
    gtcrn_iva = GTCRN_IVA(batch_size=FRONTEND_BATCH, n_frames=static_n_frames).eval()
    ckpt = torch.load(model_path + "/checkpoints/best_model_0121.tar", map_location='cpu')
    gtcrn_iva.load_state_dict(ckpt['model'], strict=False)
    gtcrn_iva.fuse_bn_()  # Fuse BatchNorm into Conv weights for optimized inference
    model = H_GTCRN_CUSTOM(
        gtcrn_iva,
        custom_stft,
        custom_istft,
        wpe_module,
        iva_module,
        n_fft=NFFT,
        in_sample_rate=IN_SAMPLE_RATE,
        out_sample_rate=OUT_SAMPLE_RATE,
        use_batch_fold=USE_BATCH_FOLD,
        fold_window=FOLD_WINDOW_LENGTH,
        model_audio_length=MODEL_AUDIO_LENGTH,
        n_frames=static_n_frames,
        frontend_batch=FRONTEND_BATCH,
        fold_input_pcm_scale=FOLD_INPUT_PCM_SCALE,
        fold_output_pcm_scale=FOLD_OUTPUT_PCM_SCALE,
    ).eval()
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.ones((1, N_CHANNELS, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    # If torch.onnx.export produced external data, re-save as a single self-contained file.
    import onnx
    data_file = os.path.basename(onnx_model_A) + ".data"
    data_path = os.path.join(os.path.dirname(onnx_model_A), data_file)
    if os.path.exists(data_path):
        onnx_model = onnx.load(onnx_model_A, load_external_data=True)
        for tensor in onnx_model.graph.initializer:
            if tensor.external_data:
                del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.DEFAULT
        onnx.save(onnx_model, onnx_model_A)
        os.remove(data_path)
        del onnx_model
    del model
    del gtcrn_iva
    del custom_stft
    del custom_istft
    del wpe_module
    del iva_module
    del audio
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="H_GTCRN", task="denoise", model_family="h_gtcrn",
    max_dynamic_audio_seconds=30, normalize_audio_default=False, input_channels=N_CHANNELS, output_channels=1,
    num_audio_inputs=1, feature_kind="stft_wpe_auxiva", center_pad=True, pad_mode=PAD_MODE,
    extra={"wpe_rt60": WPE_RT60, "wpe_delay": WPE_DELAY, "wpe_iter": WPE_ITER, "iva_iter": IVA_ITER, "cg_solve_iter": CG_SOLVE_ITER},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
