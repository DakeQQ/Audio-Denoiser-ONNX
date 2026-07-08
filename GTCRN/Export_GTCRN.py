import gc
import subprocess
import sys
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


model_path           = "/home/DakeQQ/Downloads/gtcrn-main"                 # The GTCRN download path.
parent_path          = Path(__file__).resolve().parent                    # The folder that contains this script.
onnx_model_A         = str(parent_path / "GTCRN_ONNX" / "GTCRN.onnx")    # The exported onnx model path.
onnx_model_Metadata  = str(metadata_path_for_model(onnx_model_A))         # The metadata carrier onnx model path.


DYNAMIC_AXES         = False                          # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET                = 18                             # ONNX opset.
IN_SAMPLE_RATE       = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE      = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
MODEL_SAMPLE_RATE    = 16000                          # The internal processing sample rate of the model. STFT/ISTFT and the network always run at this rate; inputs are resampled to it.
INPUT_AUDIO_LENGTH   = 32000                          # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the HOP_LENGTH value.
WINDOW_TYPE          = 'hann_sqrt'                    # Type of window function used in the STFT
N_MELS               = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT                 = 512                            # Number of FFT components for the STFT process
WINDOW_LENGTH        = 512                            # Length of windowing, edit it carefully.
HOP_LENGTH           = 256                            # Number of samples between successive frames in the STFT
BATCH_WINDOW_SECONDS = 1.5                            # When the configured input audio length is >= this many seconds, the audio is folded into fixed-length windows and batch-processed together to accelerate inference.
USE_BATCH_FOLD       = False                          # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
FOLD_WINDOW_LENGTH   = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window length (model-rate samples) for batch folding, rounded up to a multiple of HOP_LENGTH so every window reconstructs exactly through STFT -> ISTFT.
EXPORT_AUDIO_LENGTH  = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length: in fold mode it is rounded UP to a whole number of windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop, so no padding op is needed inside the graph.
MAX_SIGNAL_LENGTH    = (FOLD_WINDOW_LENGTH // HOP_LENGTH + 1) if USE_BATCH_FOLD else (4096 if DYNAMIC_AXES else INPUT_AUDIO_LENGTH // HOP_LENGTH + 1)  # Max STFT frames for the ISTFT COLA trim (per-window frame count in fold mode).
IN_AUDIO_DTYPE       = 'INT16'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'INT16'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16            = float(1.0 / 32768.0)


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
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

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
        # Read the shape once and reuse the dims; -1 infers the expanded channel count
        b, _, t, f = x.shape
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
        # (1,C,T,F) -> mean over F -> (1,C,T) -> transpose(1,2) -> (1,T,C)
        zt = x.square().mean(dim=-1).transpose(1, 2)
        # GRU + FC + Sigmoid fused; result (1,T,C) -> transpose back (1,C,T) -> unsqueeze(-1)
        at = torch.sigmoid(self.att_fc(self.att_gru(zt)[0])).transpose(1, 2).unsqueeze(-1)
        return x * at


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        self.is_deconv = use_deconv
        self.groups = groups
        self.out_channels = out_channels
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
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
    """Group Temporal Convolution - Optimized"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        self.half_channels = in_channels // 2
        self.full_channels = in_channels
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)

        self.tra = TRA(in_channels // 2)
        self.fused = False

    def fuse_bn_(self):
        """Fuse all BatchNorm layers into their preceding Conv layers."""
        if self.fused:
            return
        for conv_name, bn_name in [('point_conv1', 'point_bn1'), ('depth_conv', 'depth_bn'), ('point_conv2', 'point_bn2')]:
            conv = getattr(self, conv_name)
            bn = getattr(self, bn_name)
            std = torch.sqrt(bn.running_var + bn.eps)
            scale = bn.weight / std

            if isinstance(conv, nn.ConvTranspose2d):
                groups = conv.groups
                out_per_group = conv.out_channels // groups
                in_per_group = conv.in_channels // groups
                w = conv.weight.view(groups, in_per_group, out_per_group, conv.weight.shape[2], conv.weight.shape[3])
                fused_weight = (w * scale.view(groups, 1, out_per_group, 1, 1)).view_as(conv.weight)
            else:
                fused_weight = conv.weight * scale.view(-1, 1, 1, 1)

            fused_bias = bn.bias - bn.running_mean * scale if conv.bias is None else (conv.bias - bn.running_mean) * scale + bn.bias

            conv.weight = nn.Parameter(fused_weight)
            conv.bias = nn.Parameter(fused_bias)
            setattr(self, bn_name, nn.Identity())
        self.fused = True

    def forward(self, x):
        """x: (B, C, T, F)"""
        b, _, _, f = x.shape          # batch & freq — read once, reused for the channel-shuffle reshape
        # Use split (not chunk) with known constant sizes
        x1, x2 = x.split(self.half_channels, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)

        # ShuffleNet channel shuffle: interleave h1 and x2 along channel dim
        # stack on dim=2 -> (B, C//2, 2, T, F), then reshape to (B, C, T, F)
        # This avoids: stack+transpose+contiguous+reshape (original uses 4 ops)
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

        # Use split with known constant sizes
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
        # (B,C,T,F) -> (B,T,F,C) -> (B*T,F,C)
        x_perm = x.permute(0, 2, 3, 1)                    # (B, T, F, C)
        intra_in = x_perm.reshape(-1, self.width, self.hidden_size)  # (B*T, F, C) — reshape: x_perm is a permuted (non-contiguous) view
        intra_x = self.intra_fc(self.intra_rnn(intra_in)[0])      # (B*T, F, C)
        intra_x = self.intra_ln(intra_x.reshape(-1, t, self.width, self.hidden_size))  # (B, T, F, C) — batch via -1
        intra_out = x_perm + intra_x                       # (B, T, F, C)

        ## Inter RNN
        # (B,T,F,C) -> (B,F,T,C) -> (B*F,T,C)
        inter_in = intra_out.transpose(1, 2).reshape(-1, t, self.hidden_size)  # (B*F, T, C) — reshape: transposed (non-contiguous)
        inter_x = self.inter_fc(self.inter_rnn(inter_in)[0])      # (B*F, T, C)
        inter_x = self.inter_ln(inter_x.reshape(-1, self.width, t, self.hidden_size).transpose(1, 2))  # (B, T, F, C) — batch via -1
        inter_out = intra_out + inter_x                    # (B, T, F, C)

        return inter_out.permute(0, 3, 1, 2)              # (B, C, T, F)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False)
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
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
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


class GTCRN(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, magnitude, real_part, imag_part):
        """
        magnitude: (1, F, T)
        real_part: (1, F, T)
        imag_part: (1, F, T)
        Returns: s_real (1, F, T), s_imag (1, F, T)
        """
        # Stack then single transpose (1 transpose of 1 tensor, not 3 separate transposes)
        feat = torch.stack([magnitude, real_part, imag_part], dim=1).transpose(-1, -2)  # (1, 3, T, F)

        feat = self.erb.bm(feat)   # (1, 3, T, 129)
        feat = self.sfe(feat)      # (1, 9, T, 129)

        feat, en_outs = self.encoder(feat)

        feat = self.dpgrnn1(feat)  # (1, 16, T, 33)
        feat = self.dpgrnn2(feat)  # (1, 16, T, 33)

        m_feat = self.decoder(feat, en_outs)  # (1, 2, T, F_erb)

        # ERB band synthesis + mask application
        # (B, 2, T, F_erb) -> ERB synthesis -> (B, 2, T, 257) -> transpose(-1,-2) -> (B, 2, 257, T)
        m = self.erb.bs(m_feat).transpose(-1, -2)  # (B, 2, F, T)

        # Split the two mask channels along dim=1: each (B, F, T)
        m0, m1 = (mask.squeeze(1) for mask in m.split(1, dim=1))

        # Complex Ratio Mask (fused, no separate Mask class)
        s_real = real_part * m0 - imag_part * m1
        s_imag = imag_part * m0 + real_part * m1

        return s_real, s_imag


class GTCRN_CUSTOM(torch.nn.Module):
    def __init__(self, gtcrn, stft_model, istft_model, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(GTCRN_CUSTOM, self).__init__()
        self.gtcrn = gtcrn
        self.stft_model = stft_model
        self.istft_model = istft_model
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
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding

    def forward(self, audio):
        audio = audio.float()
        if self.resample_before_centering:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower():
            audio = audio * self.inv_int16      # int16 PCM -> [-1, 1]; F16/F32 inputs already arrive normalized.
        audio = audio - torch.mean(audio) # Remove DC Offset
        if self.resample_after_centering:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            # Input length is already a whole number of windows (the tail was padded OUTSIDE
            # the model, in numpy, by the windowing loop), so fold (1, 1, num_window*W) ->
            # (num_window, 1, W) and run the whole batch at once. No padding op inside the graph.
            audio = audio.reshape(-1, 1, self.fold_window)              # (num_window, 1, W)

        # Shared STFT -> network -> ISTFT (batch = num_window when folded, else 1)
        real_part, imag_part = self.stft_model(audio)
        magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part + 1e-12)
        s_real, s_imag = self.gtcrn(magnitude, real_part, imag_part)
        audio = self.istft_model(s_real, s_imag)

        if self.use_batch_fold:
            audio = audio.reshape(1, 1, -1)                             # stitch windows back
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
        if "int" in OUT_AUDIO_DTYPE.lower():
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return audio
        return audio.to(torch.float16)




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_GTCRN_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode="reflect").eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode="reflect").eval()
    gtcrn = GTCRN().eval()
    ckpt = torch.load(model_path + "/checkpoints/model_trained_on_dns3.tar", map_location='cpu')
    gtcrn.load_state_dict(ckpt['model'], strict=False)
    gtcrn.fuse_bn_()  # Fuse BatchNorm into Conv weights for optimized inference
    gtcrn = GTCRN_CUSTOM(gtcrn.float(), custom_stft, custom_istft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH)
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    torch.onnx.export(
        gtcrn,
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
    del gtcrn
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="GTCRN", task="denoise", model_family="gtcrn",
    max_dynamic_audio_seconds=30, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=1, feature_kind="stft", center_pad=True, pad_mode="reflect", extra={"n_mels": N_MELS},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
