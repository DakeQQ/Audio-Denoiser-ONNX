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
model_path           = r"/home/DakeQQ/Downloads/H-GTCRN-main"                        # The H-GTCRN download path.
onnx_model_A         = str(parent_path / "H_GTCRN_ONNX" / "H_GTCRN.onnx")          # The exported onnx model path.
onnx_model_Metadata  = str(metadata_path_for_model(onnx_model_A))                  # The metadata carrier onnx model path.


DYNAMIC_AXES         = False                          # False exports a fixed windowed model; set True to keep dynamic audio length so WPE/AuxIVA can use full-sequence statistics.
OPSET                = 18                             # ONNX opset.
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
MAX_SIGNAL_LENGTH    = (FOLD_WINDOW_LENGTH // HOP_LENGTH + 1) if USE_BATCH_FOLD else (4096 if DYNAMIC_AXES else INPUT_AUDIO_LENGTH // HOP_LENGTH + 1)  # Max STFT frames (per-window count in fold mode). Sizes the WPE delay templates AND the ISTFT COLA trim.
WPE_RT60             = 0.3                            # WPE reverberation time parameter.
WPE_DELAY            = 2                              # WPE prediction delay parameter.
WPE_ITER             = 1                              # WPE number of iterations.
IVA_ITER             = 10                             # AuxIVA number of iterations (must match training: 10 iterations for proper source separation).
CG_SOLVE_ITER        = 6                              # Inner CG steps for the WPE linear solve.

IN_AUDIO_DTYPE       = 'INT16'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'INT16'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16            = float(1.0 / 32768.0)


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
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        """x: (B,C,T,F) -> (B, C*kernel_size, T, F)"""
        b, _, t, f = x.shape          # read dims once; batch generalizes for fold (batch>1)
        return self.unfold(x).view(b, -1, t, f)


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)

    def forward(self, x):
        """x: (B,C,T,F) -> (B,C,T,F)"""
        zt = x.square().mean(dim=-1).transpose(1, 2)
        at = torch.sigmoid(self.att_fc(self.att_gru(zt)[0])).transpose(1, 2).unsqueeze(-1)
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
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, width=33):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        self.half_channels = in_channels // 2
        self.full_channels = in_channels

        self.sfe = SFE(kernel_size=3, stride=1)

        # Use ConvBlock wrappers to match checkpoint key paths:
        # point_conv1.conv.weight, point_conv1.bn.weight, point_conv1.act.weight, etc.
        self.point_conv1 = ConvBlock(in_channels // 2 * 3, hidden_channels, 1)
        self.depth_conv = ConvBlock(hidden_channels, hidden_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=hidden_channels)
        self.point_conv2 = ConvBlock(hidden_channels, in_channels // 2, 1)
        self.point_conv2.act = nn.Identity()  # No activation after point_conv2 (matches original)

        self.tra = TRA(in_channels // 2)

        # Pre-allocated static zero tensor for causal padding
        self.register_buffer('pad_zeros', torch.zeros(1, hidden_channels, self.pad_size, width), persistent=False)

    def fuse_bn_(self):
        """Fuse all BatchNorm layers into their preceding Conv layers."""
        self.point_conv1.fuse_bn_()
        self.depth_conv.fuse_bn_()
        self.point_conv2.fuse_bn_()

    def forward(self, x):
        """x: (B, C, T, F)"""
        b, _, _, f = x.shape          # batch & freq — read once, reused (batch>1 under fold)
        x1, x2 = x.split(self.half_channels, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_conv1(x1)
        h1 = torch.cat([self.pad_zeros.expand(b, -1, -1, -1), h1], dim=2)
        h1 = self.depth_conv(h1)
        h1 = self.point_conv2(h1)
        h1 = self.tra(h1)

        # ShuffleNet channel shuffle: interleave h1 and x2 along channel dim
        return torch.stack([h1, x2], dim=2).view(b, self.full_channels, -1, f)


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

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (h_dim, B, hidden_size)
        """
        if h is None:
            h = torch.zeros(self.h_dim, x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)

        x1, x2 = x.split(self.half_input, dim=-1)
        h1, h2 = h.split(self.half_hidden, dim=-1)

        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        t = x.shape[2]                                    # time frames — read once, reused for every reshape (batch dim uses -1)

        ## Intra RNN
        x_perm = x.permute(0, 2, 3, 1)                    # (B, T, F, C)
        intra_in = x_perm.reshape(-1, self.width, self.hidden_size)  # (B*T, F, C) — reshape: x_perm is permuted (non-contiguous)
        intra_x = self.intra_fc(self.intra_rnn(intra_in)[0])      # (B*T, F, C)
        intra_x = self.intra_ln(intra_x.reshape(-1, t, self.width, self.hidden_size))  # (B, T, F, C) — batch via -1
        intra_out = x_perm + intra_x                       # (B, T, F, C)

        ## Inter RNN
        inter_in = intra_out.transpose(1, 2).reshape(-1, t, self.hidden_size)  # (B*F, T, C) — reshape: transposed (non-contiguous)
        inter_x = self.inter_fc(self.inter_rnn(inter_in)[0])      # (B*F, T, C)
        inter_x = self.inter_ln(inter_x.reshape(-1, self.width, t, self.hidden_size).transpose(1, 2))  # (B, T, F, C) — batch via -1
        inter_out = intra_out + inter_x                    # (B, T, F, C)

        return inter_out.permute(0, 3, 1, 2)              # (B, C, T, F)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(6 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1))
        ])

    def fuse_bn_(self):
        for conv in self.en_convs:
            conv.fuse_bn_()

    def forward(self, x):
        e0 = self.en_convs[0](x)
        e1 = self.en_convs[1](e0)
        e2 = self.en_convs[2](e1)
        e3 = self.en_convs[3](e2)
        e4 = self.en_convs[4](e3)
        return e4, (e0, e1, e2, e3, e4)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
        ])

    def fuse_bn_(self):
        for conv in self.de_convs:
            conv.fuse_bn_()

    def forward(self, x, en_outs):
        x = self.de_convs[0](x + en_outs[4])
        x = self.de_convs[1](x + en_outs[3])
        x = self.de_convs[2](x + en_outs[2])
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
    def __init__(self):
        super().__init__()
        self.n_fft = 512
        self.hop_len = 256
        self.win_len = 512

        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = Decoder()

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
        feat, en_outs = self.encoder(feat)
        # Dual-path Grouped RNN
        feat = self.dpgrnn1(feat)                # (1, 16, T, 33)
        feat = self.dpgrnn2(feat)                # (1, 16, T, 33)
        # Decoder
        m_feat = self.decoder(feat, en_outs)     # (1, 2, T, F_erb=129)
        # ERB band synthesis — keep in (T,F) layout to avoid transposing large tensors
        m = self.erb.bs(m_feat)                  # (1, 2, T, F=257)

        # Split mask and ref in native (T,F) layout (no transpose needed)
        m_real, m_imag = m.split(1, dim=1)                             # each (1, 1, T, F)
        ref, _ = spec_features.split([2, 4], dim=1)                    # (1, 2, T, F)
        ref_real, ref_imag = ref.split(1, dim=1)                       # each (1, 1, T, F)

        # CRM in (T,F) layout, then transpose only the smaller output tensors to (F,T)
        s_real = (ref_real * m_real - ref_imag * m_imag).squeeze(1).transpose(-1, -2)  # (1, F, T)
        s_imag = (ref_imag * m_real + ref_real * m_imag).squeeze(1).transpose(-1, -2)  # (1, F, T)

        return s_real, s_imag


# ═══════════════════════════════════════════════════════════════════════════
# ONNX-friendly complex arithmetic helpers (real-valued representation)
# Complex tensors are represented as separate real/imag tensors or (..., 2) pairs
# ═══════════════════════════════════════════════════════════════════════════

def batched_complex_solve_cg(R_r, R_i, P_r, P_i, n_iter=36):
    """
    Solve R @ G = P for G using Conjugate Gradient (CG).
    ONNX-friendly: uses only matmul, element-wise ops, and reductions.
    Guaranteed to converge for Hermitian positive definite R.

    R: (B, F, N, N) complex Hermitian positive definite (R_r + j*R_i)
    P: (B, F, N, M) complex (P_r + j*P_i)
    Returns: G_r, G_i of same shape as P
    """
    # Initialize: x=0, r=P, p=r
    x_r = torch.zeros_like(P_r)
    x_i = torch.zeros_like(P_i)
    r_r = P_r.clone()
    r_i = P_i.clone()
    p_r = P_r.clone()
    p_i = P_i.clone()

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


def solve_2x2_complex(A, b):
    """
    Solve A @ x = b for x, where A is (..., 2, 2, 2) complex and b is (..., 2, 2) complex.
    Uses Cramer's rule for 2x2 system. ONNX-friendly (no linalg ops).

    A: (..., 2, 2, 2) - 2x2 complex matrix (last dim is real/imag)
    b: (..., 2, 2)    - 2-element complex vector (last dim is real/imag)
    Returns: x (..., 2, 2) complex vector
    """
    # A = [[a, b], [c, d]], det = ad - bc
    A_row0, A_row1 = A.split(1, dim=-3)  # each (..., 1, 2, 2)
    a, b_mat = A_row0.squeeze(-3).split(1, dim=-2)  # each (..., 1, 2)
    a, b_mat = a.squeeze(-2), b_mat.squeeze(-2)  # each (..., 2)
    c, d = A_row1.squeeze(-3).split(1, dim=-2)  # each (..., 1, 2)
    c, d = c.squeeze(-2), d.squeeze(-2)  # each (..., 2)

    # Split all complex pairs into real/imag once to avoid repeated gather
    a_r, a_i = a.split(1, dim=-1)      # each (..., 1)
    b_mat_r, b_mat_i = b_mat.split(1, dim=-1)
    c_r, c_i = c.split(1, dim=-1)
    d_r, d_i = d.split(1, dim=-1)

    # det = a*d - b*c (complex)
    ad = torch.cat([a_r * d_r - a_i * d_i, a_r * d_i + a_i * d_r], dim=-1)
    bc = torch.cat([b_mat_r * c_r - b_mat_i * c_i, b_mat_r * c_i + b_mat_i * c_r], dim=-1)
    det = ad - bc  # (..., 2)

    # inv_det = conj(det) / |det|^2
    det_r, det_i = det.split(1, dim=-1)
    det_abs_sq = 1.0 / ((det_r ** 2 + det_i ** 2) + 1e-12)
    inv_det_r = det_r * det_abs_sq   # (..., 1)
    inv_det_i = -det_i * det_abs_sq  # (..., 1)

    # x0 = (d * b[0] - b_mat * b[1]) * inv_det
    # x1 = (a * b[1] - c * b[0]) * inv_det
    b0, b1 = b.split(1, dim=-2)  # each (..., 1, 2)
    b0, b1 = b0.squeeze(-2), b1.squeeze(-2)  # each (..., 2)
    b0_r, b0_i = b0.split(1, dim=-1)
    b1_r, b1_i = b1.split(1, dim=-1)

    # num0 = d * b0 - b_mat * b1
    num0_r = (d_r * b0_r - d_i * b0_i) - (b_mat_r * b1_r - b_mat_i * b1_i)
    num0_i = (d_r * b0_i + d_i * b0_r) - (b_mat_r * b1_i + b_mat_i * b1_r)

    # num1 = a * b1 - c * b0
    num1_r = (a_r * b1_r - a_i * b1_i) - (c_r * b0_r - c_i * b0_i)
    num1_i = (a_r * b1_i + a_i * b1_r) - (c_r * b0_i + c_i * b0_r)

    # Multiply by inv_det
    x0 = torch.cat([num0_r * inv_det_r - num0_i * inv_det_i,
                    num0_r * inv_det_i + num0_i * inv_det_r], dim=-1)
    x1 = torch.cat([num1_r * inv_det_r - num1_i * inv_det_i,
                    num1_r * inv_det_i + num1_i * inv_det_r], dim=-1)

    return torch.stack([x0, x1], dim=-2)  # (..., 2, 2)


class OnnxFriendlyWPE(torch.nn.Module):
    """
    ONNX-exportable Weighted Prediction Error (WPE) dereverberation.
    Replaces torch.linalg.inv with complex Conjugate Gradient.

    Optimizations:
      - Pre-compute: eye matrix and delay templates registered as buffers
      - Hoist loop-invariant: Xp transpose computed once outside iteration loop
    """
    def __init__(self, n_channels=2, rt60=0.3, hop_length=256, delay=2, sample_rate=16000, num_iter=1, ns_iter=36,
         n_freq_bins=NFFT // 2 + 1, max_frames=MAX_SIGNAL_LENGTH):
        super().__init__()
        self.M = n_channels
        self.Lg = int(rt60 * sample_rate / hop_length)
        self.D = delay
        self.num_iter = num_iter
        self.solve_iter = ns_iter
        self.MLg = self.M * self.Lg
        self.n_freq_bins = n_freq_bins
        self.max_frames = max_frames
        # Pre-compute identity matrix as buffer (avoids runtime torch.eye allocation)
        self.register_buffer('eye_MLg', torch.eye(self.MLg, dtype=torch.float32))
        self.register_buffer(
            'delay_template_r',
            torch.zeros(1, n_freq_bins, self.MLg, max_frames, dtype=torch.float32),
        )
        self.register_buffer(
            'delay_template_i',
            torch.zeros(1, n_freq_bins, self.MLg, max_frames, dtype=torch.float32),
        )

    def forward(self, X_real, X_imag):
        """
        X_real, X_imag: (B, M, F, T) — multi-channel STFT real/imag parts.
        Returns: Y_real, Y_imag of same shape — dereverberated.
        """
        B, M, F, T = X_real.shape
        # Permute to (B, F, M, T)
        Xp_r = X_real.permute(0, 2, 1, 3)  # (B, F, M, T)
        Xp_i = X_imag.permute(0, 2, 1, 3)

        # Build delay matrix: (B, F, M*Lg, T)
        if F != self.n_freq_bins or T > self.max_frames:
            raise ValueError(
                f"WPE delay buffers require F={self.n_freq_bins} and T<={self.max_frames}, got F={F}, T={T}."
            )

        MLg = self.MLg
        # Expand the batch-1 zero delay template to the actual batch (num_window under fold);
        # for batch=1 this is a no-op. .clone() materializes a writable per-window buffer.
        X_delay_r = self.delay_template_r[:, :, :, :T].expand(B, -1, -1, -1).clone()
        X_delay_i = self.delay_template_i[:, :, :, :T].expand(B, -1, -1, -1).clone()

        for l_idx in range(self.Lg):
            start_col = self.D + l_idx
            row_start = l_idx * M
            row_end = row_start + M
            X_delay_r[:, :, row_start:row_end, start_col:] = Xp_r[:, :, :, :-start_col]
            X_delay_i[:, :, row_start:row_end, start_col:] = Xp_i[:, :, :, :-start_col]

        # Compute eps matching original: 1e-3 * mean_over_F(max_over(M,T)(|X|^2)).
        # Keep it PER-WINDOW (per batch element) so folded windows stay independent
        # (batched result == per-window loop); for batch=1 this equals the original scalar.
        mag_sq = Xp_r * Xp_r + Xp_i * Xp_i  # (B, F, M, T)
        eps_val = (1e-3 * mag_sq.amax(dim=(-2, -1)).mean(dim=-1)).reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # Y = Xp initially
        Y_r = Xp_r.clone()  # (B, F, M, T)
        Y_i = Xp_i.clone()

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
      - Pre-compute: eye_M, e_s unit vectors registered as buffers
      - Hoist: X transpose computed once outside iteration loop
      - Use split instead of slice for source channel extraction
    """
    def __init__(self, n_iter=10, n_channels=2, n_freq_bins=NFFT // 2 + 1):
        super().__init__()
        self.n_iter = n_iter
        self.M = n_channels
        self.n_freq_bins = n_freq_bins
        # Pre-computed constants as buffers
        eye_M = torch.eye(n_channels, dtype=torch.float32)
        self.register_buffer('eye_M', eye_M)
        self.register_buffer('proj_back_one', torch.ones(1, 1, dtype=torch.float32))
        self.register_buffer('proj_back_zero', torch.zeros(1, 1, dtype=torch.float32))
        # Pre-compute e_s unit vectors across the fixed frequency axis.
        e_s_all = torch.zeros(n_channels, n_freq_bins, n_channels, 2)
        for s in range(n_channels):
            e_s_all[s, :, s, 0] = 1.0
        self.register_buffer('e_s', e_s_all)
        self.eps = 1e-10
        self.register_buffer(
            'eps_eye',
            (self.eps * eye_M).view(1, 1, n_channels, n_channels).expand(1, n_freq_bins, n_channels, n_channels).clone(),
        )
        self.register_buffer(
            'init_W_r',
            eye_M.view(1, 1, n_channels, n_channels).expand(1, n_freq_bins, n_channels, n_channels).clone(),
        )
        self.register_buffer(
            'init_W_i',
            torch.zeros(1, n_freq_bins, n_channels, n_channels, dtype=torch.float32),
        )

    def forward(self, X_real, X_imag):
        """
        X_real, X_imag: (B, M=2, F, T) — dereverberated STFT.
        Returns: Y_real, Y_imag (B, M=2, F, T) — separated sources.
        """
        B, M, F, T = X_real.shape
        inv_T = 1.0 / T

        # Reshape to (B, F, M, T) for processing
        X_r = X_real.permute(0, 2, 1, 3)  # (B, F, M, T)
        X_i = X_imag.permute(0, 2, 1, 3)

        # Hoist loop-invariant: X transposed (constant across all iterations)
        X_rT = X_r.transpose(-2, -1)  # (B, F, T, M)
        X_iT = X_i.transpose(-2, -1)

        # Initialize W as identity: (B, F, M, M). Expand the batch-1 buffers to the actual
        # batch (num_window under fold; no-op for batch=1) so the per-source in-place W[s]
        # updates below write batch-B values into a writable buffer.
        W_r = self.init_W_r.expand(B, -1, -1, -1).clone()
        W_i = self.init_W_i.expand(B, -1, -1, -1).clone()

        # Y = W @ X (W starts as identity, so Y = X initially)
        Y_r = X_r.clone()
        Y_i = X_i.clone()

        for iter_idx in range(self.n_iter):
            # r = 2 * L2_norm(Y over F): (B, M, T)
            Y_pow = Y_r * Y_r + Y_i * Y_i  # (B, F, M, T)
            r = 2.0 * torch.sqrt(Y_pow.sum(dim=1) + self.eps)  # (B, M, T)
            r_inv = 1.0 / r  # (B, M, T)

            for s in range(M):
                # r_inv for source s: (B, 1, 1, T)
                w_s = r_inv[:, s:s+1, :].unsqueeze(1)  # (B, 1, 1, T)

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
                    # WV = W @ V: (B, F, M, M)
                    WV_r = torch.matmul(W_r, V_r) - torch.matmul(W_i, V_i)
                    WV_i = torch.matmul(W_r, V_i) + torch.matmul(W_i, V_r)

                # Solve (WV + eps*I) @ w_new = e_s using pre-computed buffers
                WV_r_reg = WV_r + self.eps_eye

                # Stack WV as (B, F, 2, 2, 2) for solve_2x2_complex
                A_solve = torch.stack([WV_r_reg, WV_i], dim=-1)  # (B, F, M, M, 2)
                w_new = solve_2x2_complex(A_solve, self.e_s[s])  # (B, F, M, 2)

                # w_new split into real/imag
                w_new_r, w_new_i = w_new.split(1, dim=-1)  # each (B, F, M, 1)
                w_new_r = w_new_r.squeeze(-1)  # (B, F, M)
                w_new_i = w_new_i.squeeze(-1)

                # W[s] = conj(w_new)
                conj_w_r = w_new_r   # (B, F, M)
                conj_w_i = -w_new_i

                # Normalize: denom = conj(w) @ V @ w
                wn_r_col = w_new_r.unsqueeze(-1)  # (B, F, M, 1)
                wn_i_col = w_new_i.unsqueeze(-1)
                Vw_r = torch.matmul(V_r, wn_r_col) - torch.matmul(V_i, wn_i_col)  # (B, F, M, 1)
                Vw_i = torch.matmul(V_r, wn_i_col) + torch.matmul(V_i, wn_r_col)

                denom_r = (conj_w_r * Vw_r.squeeze(-1) - conj_w_i * Vw_i.squeeze(-1)).sum(dim=-1)

                # For HPD V, w^H V w is real and non-negative; normalizing with the
                # real scalar avoids phase drift from a noisy complex sqrt.
                norm_scale = torch.rsqrt(denom_r.clamp(min=0.0) + self.eps).unsqueeze(-1)
                final_r = conj_w_r * norm_scale
                final_i = conj_w_i * norm_scale

                # Update W[s]
                W_r[:, :, s, :] = final_r
                W_i[:, :, s, :] = final_i

            # Recompute Y = W @ X
            Y_r = torch.matmul(W_r, X_r) - torch.matmul(W_i, X_i)
            Y_i = torch.matmul(W_r, X_i) + torch.matmul(W_i, X_r)

        # Projection back to align with reference channel (channel 0)
        ref_r = X_r[:, :, 0, :]  # (B, F, T)
        ref_i = X_i[:, :, 0, :]

        # Use split for channel extraction instead of indexing + zeros_like
        Y_s_list_r = Y_r.split(1, dim=2)  # list of (B, F, 1, T)
        Y_s_list_i = Y_i.split(1, dim=2)

        out_r_list = []
        out_i_list = []
        for s in range(M):
            Ys_r = Y_s_list_r[s].squeeze(2)  # (B, F, T)
            Ys_i = Y_s_list_i[s].squeeze(2)

            num_r = (ref_r * Ys_r + ref_i * Ys_i).sum(dim=-1)
            num_i = (ref_r * Ys_i - ref_i * Ys_r).sum(dim=-1)
            denom = (Ys_r * Ys_r + Ys_i * Ys_i).sum(dim=-1)
            valid = denom > 0.0
            safe_denom = 1.0 / torch.where(valid, denom, self.proj_back_one)

            c_r = torch.where(valid, num_r * safe_denom, self.proj_back_one).unsqueeze(-1)  # (B, F, 1)
            c_i = torch.where(valid, num_i * safe_denom, self.proj_back_zero).unsqueeze(-1)

            out_r_list.append(c_r * Ys_r + c_i * Ys_i)
            out_i_list.append(c_r * Ys_i - c_i * Ys_r)

        Y_out_r = torch.stack(out_r_list, dim=2)
        Y_out_i = torch.stack(out_i_list, dim=2)

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
    def __init__(self, gtcrn_core, stft_model, istft_model, wpe_module, iva_module, n_fft=512, in_sample_rate=16000, out_sample_rate=16000, use_batch_fold=False, fold_window=0):
        super(H_GTCRN_CUSTOM, self).__init__()
        self.gtcrn = gtcrn_core
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.wpe = wpe_module
        self.iva = iva_module
        # Pre-computed constants as buffers (no runtime computation)
        self.register_buffer('inv_int16', torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32))
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
        if "int" in IN_AUDIO_DTYPE.lower():
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
            stft_in = audio_f.reshape(2, -1, self.fold_window).transpose(0, 1).reshape(-1, 1, self.fold_window)
        else:
            # view replaces squeeze(0)+unsqueeze(1): (1,2,L) -> (2,1,L) in one op
            stft_in = audio_f.view(2, 1, -1)
        real_parts, imag_parts = self.stft_model(stft_in)  # each (B*2, F, T)  [B=num_window fold / 1 non-fold]
        n_frames = real_parts.shape[-1]

        # ─── 3. WPE dereverberation ──────────────────────────────────────
        # Regroup channels for the multi-channel front-end: (B*2, F, T) -> (B, 2, F, T)
        drb_real, drb_imag = self.wpe(
            real_parts.reshape(-1, 2, self.n_freq_bins, n_frames),
            imag_parts.reshape(-1, 2, self.n_freq_bins, n_frames),
        )

        # ─── 4. AuxIVA source separation ─────────────────────────────────
        iva_real, iva_imag = self.iva(drb_real, drb_imag)  # each (B, 2, F, T)

        # ─── 5. Channel selection via torch.where (no float mul) ──────────
        # Use split instead of slice for ONNX-friendly channel extraction
        iva_r0, iva_r1 = iva_real.split(1, dim=1)  # each (B, 1, F, T)
        iva_i0, iva_i1 = iva_imag.split(1, dim=1)
        energy = (iva_real * iva_real + iva_imag * iva_imag).sum(dim=(2, 3))  # (B, 2)
        # pred broadcast shape fused: use split instead of gather for ONNX-friendly extraction
        energy_0, energy_1 = energy.split(1, dim=1)  # each (B, 1)
        pred = (energy_0 < energy_1).reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)
        sel_real = torch.where(pred, iva_r0, iva_r1)      # (B, 1, F, T)
        sel_imag = torch.where(pred, iva_i0, iva_i1)
        unsel_real = torch.where(pred, iva_r1, iva_r0)
        unsel_imag = torch.where(pred, iva_i1, iva_i0)

        # ─── 6. Fused log-magnitude: log10(sqrt(x)) = 0.5*log10(x) ──────
        # Original clamps the MAGNITUDE at 1e-12 (norm(...).clamp(1e-12)). Because sqrt is
        # monotonic, clamp(sqrt(x),1e-12) == sqrt(clamp(x,1e-24)), so clamp the squared
        # magnitude at (1e-12)^2 = 1e-24 to stay bit-exact with the original feature floor.
        sel_log = 0.5 * torch.log10((sel_real * sel_real + sel_imag * sel_imag).clamp(min=1e-24))
        unsel_log = 0.5 * torch.log10((unsel_real * unsel_real + unsel_imag * unsel_imag).clamp(min=1e-24))

        # ─── 7. Feature construction: single stack+cat+transpose ─────────
        # Stack real/imag interleaved -> (B*2,2,F,T) -> reshape (B,4,F,T)
        # Ordering: [ch0_real, ch0_imag, ch1_real, ch1_imag] (matches original)
        spec_4ch = torch.stack([real_parts, imag_parts], dim=1).reshape(-1, 4, self.n_freq_bins, n_frames)
        # Combine all 6 features in (F,T) layout, then single transpose to (T,F)
        spec_features = torch.cat([spec_4ch, sel_log, unsel_log], dim=1).transpose(-1, -2)  # (B, 6, T, F)

        # ─── 8. GTCRN network (ERB + Encoder + DPGRNN + Decoder + CRM) ───
        s_real, s_imag = self.gtcrn(spec_features)  # each (B, F, T)

        # ─── 9. iSTFT -> time-domain audio ───────────────────────────────
        audio_out = self.istft_model(s_real, s_imag)  # (B, 1, W)
        if self.use_batch_fold:
            audio_out = audio_out.reshape(1, 1, -1)   # stitch windows back

        # ─── 10. Resample output and scale to int16 PCM ──────────────────
        if self.output_resample_before_pcm:
            audio_out = torch.nn.functional.interpolate(
                audio_out,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            audio_out = audio_out * self.output_pcm_scale      # [-1, 1] -> int16 PCM; F16/F32 outputs stay normalized.
        if self.output_resample_after_pcm:
            audio_out = torch.nn.functional.interpolate(
                audio_out,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        # The WPE/AuxIVA front-end divides by a determinant that is zero for a fully silent
        # window, so guard the output against NaN/Inf (the int16 cast already maps NaN -> 0, so
        # this keeps the int16 output identical while making the float output finite on silence).
        audio_out = torch.nan_to_num(audio_out, nan=0.0, posinf=32767.0, neginf=-32768.0)
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
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode=PAD_MODE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode=PAD_MODE).eval()
    wpe_module = OnnxFriendlyWPE(
        n_channels=N_CHANNELS,
        rt60=WPE_RT60,
        hop_length=HOP_LENGTH,
        delay=WPE_DELAY,
        sample_rate=IN_SAMPLE_RATE,
        num_iter=WPE_ITER,
        ns_iter=CG_SOLVE_ITER,
        n_freq_bins=NFFT // 2 + 1,
        max_frames=MAX_SIGNAL_LENGTH,
    ).eval()
    iva_module = OnnxFriendlyAuxIVA(n_iter=IVA_ITER, n_channels=N_CHANNELS).eval()
    gtcrn_iva = GTCRN_IVA().eval()
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
        opset_version=17,
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
