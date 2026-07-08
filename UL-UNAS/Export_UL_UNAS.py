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


model_path             = r"/home/DakeQQ/Downloads/ul-unas-main"                # The UL-UNAS download path.
parent_path            = Path(__file__).resolve().parent                     # The folder that contains this script.
onnx_model_A           = str(parent_path / "UL_UNAS_ONNX" / "UL_UNAS.onnx")  # The exported onnx model path.
onnx_model_Metadata    = str(metadata_path_for_model(onnx_model_A))           # The metadata carrier onnx model path.


DYNAMIC_AXES          = False                        # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
IN_SAMPLE_RATE        = 16000                        # UL-UNAS is designed for 16kHz only.
OUT_SAMPLE_RATE       = 16000                        # UL-UNAS is designed for 16kHz only.
MODEL_SAMPLE_RATE     = 16000                        # The internal processing sample rate of the model. STFT/ISTFT and the network always run at this rate; inputs are resampled to it.
INPUT_AUDIO_LENGTH    = 32000                        # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the HOP_LENGTH value.
WINDOW_TYPE           = 'hann'                       # Type of window function used in the STFT. UL-UNAS uses standard hann.
STFT_PAD_MODE         = 'reflect'                   # ["constant", "reflect"]
N_MELS                = 100                          # Number of Mel bands to generate in the Mel-spectrogram
NFFT                  = 512                          # Number of FFT components for the STFT process
WINDOW_LENGTH         = 512                          # Length of windowing, edit it carefully.
HOP_LENGTH            = 256                          # Number of samples between successive frames in the STFT
BATCH_WINDOW_SECONDS  = 1.5                          # When the configured input audio length is >= this many seconds, the audio is folded into fixed-length windows and batch-processed together to accelerate inference.
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window length (model-rate samples) for batch folding, rounded up to a multiple of HOP_LENGTH so every window reconstructs exactly through STFT -> ISTFT.
USE_BATCH_FOLD        = False                         # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length: in fold mode it is rounded UP to a whole number of windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop, so no padding op is needed inside the graph.
MAX_SIGNAL_LENGTH     = (FOLD_WINDOW_LENGTH // HOP_LENGTH + 1) if USE_BATCH_FOLD else (4096 if DYNAMIC_AXES else 512)  # Max STFT frames for the ISTFT COLA trim (per-window frame count in fold mode).
OPSET                 = 18                           # ONNX opset version. Set it to 17 for better performance and compatibility. You can adjust it if you encounter issues with certain providers.
IN_AUDIO_DTYPE        = 'INT16'                      # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE       = 'INT16'                      # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16             = float(1.0 / 32768.0)
REMOVE_DC_OFFSET      = False                        # Keep disabled for parity with the original UL-UNAS inference path.


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        self.nfreqs = nfreqs
        self.high_split = nfreqs - erb_subband_1
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
        return (10 ** (erb_f * 0.046728972) - 1) * 228.832951945

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


class AffinePReLU(nn.Module):
    def __init__(self, channels, width, init=0.25):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(1, channels, 1, width))
        self.affine_bias = nn.Parameter(torch.zeros(1, channels, 1, width))
        self.slope_weight = nn.Parameter(torch.empty(1, channels, 1, 1))
        nn.init.constant_(self.slope_weight, init)

    def forward(self, x):
        y = self.affine_weight * x + self.affine_bias
        y = y + torch.where(x > 0, x, self.slope_weight * x)
        return y


class FA(nn.Module):
    def __init__(self, nfreq, freq_comp_ratio=4):
        super().__init__()
        self.r = freq_comp_ratio

        self.gru = nn.GRU(self.r, self.r, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * self.r, self.r)
        self.nfreq = nfreq

        remainder = nfreq % self.r
        if remainder != 0:
            self.pad_len = self.r - remainder
        else:
            self.pad_len = 0

        self.F_pad = nfreq + self.pad_len
        self.H = self.F_pad // self.r

    def forward(self, x):
        """x: (B, C, T, F)"""
        b = x.shape[0]                     # batch — read once, reused for the final reshape (batch>1 under fold)
        x = torch.mean(x.square(), dim=1)  # (B, T, F)

        x = nn.functional.pad(x, (0, self.pad_len))
        x = x.view(1, -1, self.H, self.r)

        # Replace einops: 'b t h c -> (b t) h c'
        x = x.reshape(-1, self.H, self.r)
        x, _ = self.gru(x)  # (BT, H, 2r)
        x = self.fc(x)      # (BT, H, r)

        x = x.reshape(b, 1, -1, self.F_pad)  # (B, 1, T, F)

        if self.pad_len > 0:
            x = x[..., :self.nfreq]

        return x


class cTFA(nn.Module):
    """causal time-frequency attention"""
    def __init__(self, channels, width):
        super().__init__()
        self.channels = channels
        self.ta_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.ta_fc = nn.Linear(channels * 2, channels)

        self.fa = FA(width)

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.ta_gru(zt.transpose(1, 2))[0]
        at = self.ta_fc(at).transpose(1, 2)  # (B,C,T)
        at = torch.sigmoid(at).unsqueeze(-1)

        af = self.fa(x)
        af = torch.sigmoid(af)

        return at * x * af


class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """x: (B,2C,T,F) -> channel shuffle"""
        b, C2, _, F = x.shape
        C = C2 // 2
        x1, x2 = x.split(C, dim=1)
        # Replace einops: stack then interleave
        # x1,x2 each (B,C,T,F) -> stack on dim=2 -> (B,C,2,T,F) -> reshape to (B,2C,T,F)
        x = torch.stack([x1, x2], dim=2)  # (B, C, 2, T, F)
        x = x.reshape(b, C2, -1, F)
        return x


class XConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        kernel_size,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.g = groups
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        kt = kernel_size[0]

        if use_deconv:
            pt = kt - 1
            conv_module = nn.ConvTranspose2d
        else:
            pt = 0
            conv_module = nn.Conv2d

        pf = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d([0, 0, kt - 1, 0])
        self.conv = conv_module(in_channels, out_channels, kernel_size,
                                stride=(1, stride), padding=(pt, pf), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = AffinePReLU(out_channels, width) if not is_last else nn.Identity()
        self.ctfa = cTFA(out_channels, width)
        self.shuffle = Shuffle() if (not is_last and groups == 2) else nn.Identity()

    def fuse_bn_(self):
        if isinstance(self.bn, nn.Identity):
            return
        conv = self.conv
        bn = self.bn
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
        self.bn = nn.Identity()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.ctfa(x)
        x = self.shuffle(x)
        return x


class XDWSBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        kernel_size,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        kt = kernel_size[0]

        if use_deconv:
            pt = kt - 1
            conv_module = nn.ConvTranspose2d
        else:
            pt = 0
            conv_module = nn.Conv2d

        pf = kernel_size[1] // 2

        if stride == 2:
            if not use_deconv:
                in_width = width * 2 - 1
            else:
                in_width = width // 2 + 1
        else:
            in_width = width

        self.pconv_conv = nn.Conv2d(in_channels, out_channels, 1, groups=groups)
        self.pconv_bn = nn.BatchNorm2d(out_channels)
        self.pconv_act = AffinePReLU(out_channels, in_width)
        self.pconv_shuffle = Shuffle() if groups == 2 else nn.Identity()

        self.dconv_pad = nn.ZeroPad2d([0, 0, kt - 1, 0])
        self.dconv_conv = conv_module(out_channels, out_channels, kernel_size,
                                      stride=(1, stride), padding=(pt, pf), groups=out_channels)
        self.dconv_bn = nn.BatchNorm2d(out_channels)
        self.dconv_act = AffinePReLU(out_channels, width) if not is_last else nn.Identity()
        self.dconv_ctfa = cTFA(out_channels, width)

    def fuse_bn_(self):
        for conv_attr, bn_attr in [('pconv_conv', 'pconv_bn'), ('dconv_conv', 'dconv_bn')]:
            bn = getattr(self, bn_attr)
            if isinstance(bn, nn.Identity):
                continue
            conv = getattr(self, conv_attr)
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
            setattr(self, bn_attr, nn.Identity())

    def forward(self, x):
        """x: (B, C, T, F)"""
        h = self.pconv_conv(x)
        h = self.pconv_bn(h)
        h = self.pconv_act(h)
        h = self.pconv_shuffle(h)

        h = self.dconv_pad(h)
        h = self.dconv_conv(h)
        h = self.dconv_bn(h)
        h = self.dconv_act(h)
        h = self.dconv_ctfa(h)
        return h


class XMBBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        kernel_size,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        kt = kernel_size[0]

        if use_deconv:
            pt = kt - 1
            conv_module = nn.ConvTranspose2d
        else:
            pt = 0
            conv_module = nn.Conv2d

        pf = kernel_size[1] // 2

        if stride == 2:
            if not use_deconv:
                in_width = width * 2 - 1
            else:
                in_width = width // 2 + 1
        else:
            in_width = width

        self.pconv1_conv = nn.Conv2d(in_channels, out_channels, 1, groups=groups)
        self.pconv1_bn = nn.BatchNorm2d(out_channels)
        self.pconv1_act = AffinePReLU(out_channels, in_width)
        self.pconv1_shuffle = Shuffle() if groups == 2 else nn.Identity()

        self.dconv_pad = nn.ZeroPad2d([0, 0, kt - 1, 0])
        self.dconv_conv = conv_module(out_channels, out_channels, kernel_size,
                                      stride=(1, stride), padding=(pt, pf), groups=out_channels)
        self.dconv_bn = nn.BatchNorm2d(out_channels)
        self.dconv_act = AffinePReLU(out_channels, width)

        self.pconv2_conv = nn.Conv2d(out_channels, out_channels, 1, groups=groups)
        self.pconv2_bn = nn.BatchNorm2d(out_channels)
        self.pconv2_ctfa = cTFA(out_channels, width)
        self.shuffle = Shuffle() if (not is_last and groups == 2) else nn.Identity()

    def fuse_bn_(self):
        for conv_attr, bn_attr in [('pconv1_conv', 'pconv1_bn'), ('dconv_conv', 'dconv_bn'), ('pconv2_conv', 'pconv2_bn')]:
            bn = getattr(self, bn_attr)
            if isinstance(bn, nn.Identity):
                continue
            conv = getattr(self, conv_attr)
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
            setattr(self, bn_attr, nn.Identity())

    def forward(self, x):
        input_x = x
        x = self.pconv1_conv(x)
        x = self.pconv1_bn(x)
        x = self.pconv1_act(x)
        x = self.pconv1_shuffle(x)

        x = self.dconv_pad(x)
        x = self.dconv_conv(x)
        x = self.dconv_bn(x)
        x = self.dconv_act(x)

        x = self.pconv2_conv(x)
        x = self.pconv2_bn(x)
        x = self.pconv2_ctfa(x)

        if x.shape == input_x.shape:
            x = x + input_x

        x = self.shuffle(x)
        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False, max_batch_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.half_hidden = hidden_size // 2
        self.half_input = input_size // 2
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.h_dim = num_layers * (2 if bidirectional else 1)
        self.max_batch_size = max_batch_size
        self.register_buffer(
            'hidden_buffer',
            torch.zeros([self.h_dim, max_batch_size, hidden_size], dtype=torch.int8),
            persistent=False,
        )
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (h_dim, B, hidden_size)
        """
        if h is None:
            batch_size = x.shape[0]
            if batch_size > self.hidden_buffer.shape[1]:
                self.hidden_buffer = self.hidden_buffer.new_zeros(self.h_dim, batch_size, self.hidden_size)
            h = self.hidden_buffer[:, :batch_size, :].float()
            if h.device != x.device or h.dtype != x.dtype:
                h = h.to(device=x.device, dtype=x.dtype)

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
        self.intra_fc = nn.Linear(hidden_size, input_size)
        self.intra_ln = nn.LayerNorm((width, input_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, input_size)
        self.inter_ln = nn.LayerNorm((width, input_size), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        t = x.shape[2]                                    # time frames — read once, reused for every reshape (batch dim uses -1)

        ## Intra RNN
        x_perm = x.permute(0, 2, 3, 1)  # (B, T, F, C)
        intra_in = x_perm.reshape(-1, self.width, self.input_size)  # (B*T, F, C)
        intra_x = self.intra_fc(self.intra_rnn(intra_in)[0])       # (B*T, F, C)
        intra_x = self.intra_ln(intra_x.reshape(-1, t, self.width, self.input_size))  # (B, T, F, C) — batch via -1 (reshape: intra_x is contiguous but keep parity with permuted path)
        intra_out = x_perm + intra_x  # (B, T, F, C)

        ## Inter RNN
        inter_in = intra_out.transpose(1, 2).reshape(-1, t, self.input_size)  # (B*F, T, C) — reshape: transposed (non-contiguous)
        inter_x = self.inter_fc(self.inter_rnn(inter_in)[0])  # (B*F, T, C)
        inter_x = self.inter_ln(inter_x.reshape(-1, self.width, t, self.input_size).transpose(1, 2))  # (B, T, F, C) — batch via -1
        inter_out = intra_out + inter_x  # (B, T, F, C)

        return inter_out.permute(0, 3, 1, 2)  # (B, C, T, F)


class Encoder(nn.Module):
    def __init__(
        self,
        types,
        channels,
        widths,
        kernels,
        strides,
        groups
    ):
        super().__init__()
        block_types = [XConvBlock, XDWSBlock, XMBBlocks]
        n_blocks = len(types)
        en_convs = []
        in_channels = 1
        for i in range(n_blocks):
            module = block_types[types[i]]
            out_channels = channels[i]
            en_convs.append(module(in_channels, out_channels, widths[i],
                                   kernels[i], strides[i], groups=groups[i]))
            in_channels = out_channels

        self.en_convs = nn.ModuleList(en_convs)

    def fuse_bn_(self):
        for conv in self.en_convs:
            conv.fuse_bn_()

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(
        self,
        types,
        channels,
        widths,
        kernels,
        strides,
        groups,
        final_width
    ):
        super().__init__()
        block_types = [XConvBlock, XDWSBlock, XMBBlocks]
        n_blocks = len(types)
        de_convs = []
        in_channels = channels[-1]

        for i in range(n_blocks - 1, 0, -1):
            module = block_types[types[i]]
            out_channels = channels[i - 1]
            de_convs.append(module(in_channels, out_channels, widths[i - 1],
                                   kernels[i], strides[i], groups[i], use_deconv=True))
            in_channels = out_channels

        module = block_types[types[0]]
        de_convs.append(module(in_channels, 1, final_width, kernels[0], strides[0], groups[0], use_deconv=True, is_last=True))

        self.de_convs = nn.ModuleList(de_convs)

    def fuse_bn_(self):
        for conv in self.de_convs:
            conv.fuse_bn_()

    def forward(self, x, en_outs):
        n_blocks = len(self.de_convs)
        for i in range(n_blocks):
            x = self.de_convs[i](x + en_outs[n_blocks - i - 1])
        x = torch.sigmoid(x)
        return x


class ULUNAS(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        erb_low=65,
        erb_high=64,
        types=[0, 2, 1, 2, 1],
        strides=[2, 2, 1, 1, 1],
        groups=[1, 2, 2, 2, 2],
        channels=[12, 24, 24, 32, 16],
        kernels=[(3, 3), (2, 3), (2, 3), (1, 5), (1, 5)],
        widths=[65, 33, 33, 33, 33]
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len

        self.erb = ERB(erb_low, erb_high, nfft=n_fft, high_lim=8000, fs=16000)

        self.encoder = Encoder(types, channels, widths, kernels, strides, groups)

        self.dpgrnn = nn.Sequential(
            *[DPGRNN(channels[-1], widths[-1], channels[-1]) for _ in range(2)]
        )

        self.decoder = Decoder(types, channels, widths, kernels, strides, groups, final_width=erb_low + erb_high)

    def fuse_bn_(self):
        """Fuse all BatchNorm layers into Conv weights for inference."""
        self.encoder.fuse_bn_()
        self.decoder.fuse_bn_()

    def forward(self, magnitude):
        """
        Modified forward for ONNX export.
        STFT/ISTFT handled externally by STFT_Process.

        Args:
            magnitude: (1, F, T) from external STFT, where F = n_fft//2 + 1

        Returns:
            mask: (1, F, T) sigmoid mask to apply to real and imag parts
        """
        # magnitude: (1, F, T) -> (1, 1, T, F) for 2D conv processing
        feat = torch.log10(magnitude.unsqueeze(1).transpose(-1, -2))  # (1, 1, T, F=257)

        feat = self.erb.bm(feat)  # (1, 1, T, 129)

        feat, en_outs = self.encoder(feat)

        feat = self.dpgrnn(feat)  # (1, 16, T, 33)

        m_feat = self.decoder(feat, en_outs)  # (1, 1, T, 129) with sigmoid already applied

        m = self.erb.bs(m_feat)  # (1, 1, T, 257)

        # Transpose mask back to (1, F, T) to match external real/imag shape
        return m.squeeze(1).transpose(-1, -2)  # (1, 257, T) = (1, F, T)

def convert_state_dict(original_sd, types):
    """
    Convert state dict keys from the original ULUNAS model (which uses nn.Sequential)
    to the optimized model format (which uses flat named attributes).

    The original model uses:
      - XConvBlock:  ops.{0=pad,1=conv,2=bn,3=act,4=ctfa,5=shuffle}
      - XDWSBlock:   pconv.{0=conv,1=bn,2=act,3=shuffle} / dconv.{0=pad,1=conv,2=bn,3=act,4=ctfa}
      - XMBBlocks:   pconv1.{0=conv,1=bn,2=act,3=shuffle} / dconv.{0=pad,1=conv,2=bn,3=act} / pconv2.{0=conv,1=bn,2=ctfa}

    The optimized model uses:
      - XConvBlock:  pad, conv, bn, act, ctfa, shuffle
      - XDWSBlock:   pconv_conv, pconv_bn, pconv_act, pconv_shuffle / dconv_pad, dconv_conv, dconv_bn, dconv_act, dconv_ctfa
      - XMBBlocks:   pconv1_conv, pconv1_bn, pconv1_act, pconv1_shuffle / dconv_pad, dconv_conv, dconv_bn, dconv_act / pconv2_conv, pconv2_bn, pconv2_ctfa
    """
    # Key mappings for each block type
    xconv_map = [
        ('ops.1.', 'conv.'),
        ('ops.2.', 'bn.'),
        ('ops.3.', 'act.'),
        ('ops.4.', 'ctfa.'),
    ]
    xdws_map = [
        ('pconv.0.', 'pconv_conv.'),
        ('pconv.1.', 'pconv_bn.'),
        ('pconv.2.', 'pconv_act.'),
        ('dconv.1.', 'dconv_conv.'),
        ('dconv.2.', 'dconv_bn.'),
        ('dconv.3.', 'dconv_act.'),
        ('dconv.4.', 'dconv_ctfa.'),
    ]
    xmb_map = [
        ('pconv1.0.', 'pconv1_conv.'),
        ('pconv1.1.', 'pconv1_bn.'),
        ('pconv1.2.', 'pconv1_act.'),
        ('dconv.1.', 'dconv_conv.'),
        ('dconv.2.', 'dconv_bn.'),
        ('dconv.3.', 'dconv_act.'),
        ('pconv2.0.', 'pconv2_conv.'),
        ('pconv2.1.', 'pconv2_bn.'),
        ('pconv2.2.', 'pconv2_ctfa.'),
    ]
    block_maps = {0: xconv_map, 1: xdws_map, 2: xmb_map}

    # Decoder block types (reversed order from encoder)
    n_blocks = len(types)
    decoder_types = [types[i] for i in range(n_blocks - 1, 0, -1)] + [types[0]]

    new_sd = {}
    for key, value in original_sd.items():
        new_key = key

        # Handle encoder blocks
        if key.startswith('encoder.en_convs.'):
            parts = key.split('.', 3)  # ['encoder', 'en_convs', idx, remainder]
            block_idx = int(parts[2])
            remainder = parts[3]
            block_type = types[block_idx]
            for old_prefix, new_prefix in block_maps[block_type]:
                if remainder.startswith(old_prefix):
                    new_key = f'encoder.en_convs.{block_idx}.{new_prefix}{remainder[len(old_prefix):]}'
                    break

        # Handle decoder blocks
        elif key.startswith('decoder.de_convs.'):
            parts = key.split('.', 3)  # ['decoder', 'de_convs', idx, remainder]
            block_idx = int(parts[2])
            remainder = parts[3]
            block_type = decoder_types[block_idx]
            for old_prefix, new_prefix in block_maps[block_type]:
                if remainder.startswith(old_prefix):
                    new_key = f'decoder.de_convs.{block_idx}.{new_prefix}{remainder[len(old_prefix):]}'
                    break

        if new_key.endswith('affine_weight') or new_key.endswith('affine_bias'):
            value = value.view(1, value.shape[0], 1, value.shape[1])
        elif new_key.endswith('slope_weight'):
            value = value.view(1, value.shape[0], 1, 1)

        new_sd[new_key] = value

    return new_sd


class ULUNAS_CUSTOM(torch.nn.Module):
    def __init__(self, ulunas, stft_model, istft_model, in_sample_rate, out_sample_rate, remove_dc_offset=False, use_batch_fold=False, fold_window=0):
        super(ULUNAS_CUSTOM, self).__init__()
        self.ulunas = ulunas
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
        self.remove_dc_offset = remove_dc_offset
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding

    def forward(self, audio):
        audio = audio.float()
        audio_len = audio.shape[-1] if DYNAMIC_AXES else INPUT_AUDIO_LENGTH
        if self.resample_before_centering:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower():
            audio = audio * self.inv_int16      # int16 PCM -> [-1, 1]; F16/F32 inputs already arrive normalized.
        if self.resample_after_centering:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if self.remove_dc_offset:
            audio = audio - torch.mean(audio, dim=-1, keepdim=True)
        if self.use_batch_fold:
            # Input length is already a whole number of windows (the tail was padded OUTSIDE
            # the model, in numpy, by the windowing loop), so fold (1, 1, num_window*W) ->
            # (num_window, 1, W) and run the whole batch at once. No padding op inside the graph.
            audio = audio.reshape(-1, 1, self.fold_window)              # (num_window, 1, W)
        real_part, imag_part = self.stft_model(audio)
        magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part + 1e-12)
        # UL-UNAS returns a real-valued sigmoid mask (1, F, T)
        mask = self.ulunas(magnitude)
        # Apply mask to both real and imag (simple magnitude masking)
        s_real = real_part * mask
        s_imag = imag_part * mask
        audio = self.istft_model(s_real, s_imag)
        if self.use_batch_fold:
            audio = audio.reshape(1, 1, -1)                             # stitch windows back
        audio = audio[..., :audio_len]
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
        audio = torch.nan_to_num(audio, nan=0.0, posinf=32767.0, neginf=-32768.0)
        if "int" in OUT_AUDIO_DTYPE.lower():
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return audio
        return audio.to(torch.float16)


def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_UL_UNAS_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode=STFT_PAD_MODE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode=STFT_PAD_MODE).eval()
    ulunas = ULUNAS().eval()
    ckpt = torch.load(model_path + "/checkpoints/model_trained_on_dns3.tar", map_location='cpu', weights_only=False)
    converted_sd = convert_state_dict(ckpt['model'], types=[0, 2, 1, 2, 1])
    ulunas.load_state_dict(converted_sd, strict=False)
    ulunas.fuse_bn_()  # Fuse BatchNorm into Conv weights for optimized inference
    ulunas = ULUNAS_CUSTOM(ulunas.float(), custom_stft, custom_istft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, remove_dc_offset=REMOVE_DC_OFFSET, use_batch_fold=USE_BATCH_FOLD, fold_window=FOLD_WINDOW_LENGTH)
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    torch.onnx.export(
        ulunas,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del ulunas
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="UL_UNAS", task="denoise", model_family="ul_unas",
    max_dynamic_audio_seconds=30, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=1, feature_kind="stft_erb", center_pad=True, pad_mode=STFT_PAD_MODE,
    extra={"n_mels": N_MELS, "remove_dc_offset": REMOVE_DC_OFFSET},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
