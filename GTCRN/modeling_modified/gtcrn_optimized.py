"""
GTCRN Optimized: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params

Optimizations applied (reference: Export_SDAEC.py style):
  - Weight fusion: BatchNorm fused into Conv/ConvTranspose weights
  - Scale fusion: ERB weights pre-transposed as buffers (avoid runtime .T)
  - Reduce dim changes: minimize transpose, view, permute, reshape, squeeze, unsqueeze
  - Use split instead of chunk/slice/gather
  - Pre-compute: constants pre-computed and registered as buffers
  - Use -1 for unknown dims, avoid .shape queries where possible
  - Fuse concat+transpose into direct layout
"""
import torch
import numpy as np
import torch.nn as nn


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
        # Use -1 for unknown dims; avoid querying x.shape multiple times
        return self.unfold(x).view(1, -1, x.shape[2], x.shape[3])


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
        """x: (1, C, T, F)"""
        # Use split (not chunk) with known constant sizes
        x1, x2 = x.split(self.half_channels, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)

        # ShuffleNet channel shuffle: interleave h1 and x2 along channel dim
        # stack on dim=2 -> (1, C//2, 2, T, F), then reshape to (1, C, T, F)
        # This avoids: stack+transpose+contiguous+reshape (original uses 4 ops)
        return torch.stack([h1, x2], dim=2).view(1, self.full_channels, -1, x.shape[-1])


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
        """x: (1, C, T, F)"""
        ## Intra RNN
        # (1,C,T,F) -> (1,T,F,C) -> (T,F,C) -- squeeze batch, merge T into batch
        x_perm = x.permute(0, 2, 3, 1)                    # (1, T, F, C)
        intra_in = x_perm.view(-1, self.width, self.hidden_size)  # (T, F, C)
        intra_x = self.intra_fc(self.intra_rnn(intra_in)[0])      # (T, F, C)
        intra_x = self.intra_ln(intra_x.view(1, -1, self.width, self.hidden_size))  # (1, T, F, C)
        intra_out = x_perm + intra_x                       # (1, T, F, C)

        ## Inter RNN
        # (1,T,F,C) -> (1,F,T,C) -> (F,T,C)
        inter_in = intra_out.transpose(1, 2).view(-1, intra_out.shape[1], self.hidden_size)  # (F, T, C)
        inter_x = self.inter_fc(self.inter_rnn(inter_in)[0])      # (F, T, C)
        inter_x = self.inter_ln(inter_x.view(1, self.width, -1, self.hidden_size).transpose(1, 2))  # (1, T, F, C)
        inter_out = intra_out + inter_x                    # (1, T, F, C)

        return inter_out.permute(0, 3, 1, 2)              # (1, C, T, F)


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
        # (1, 2, T, 257) -> squeeze(0) -> (2, T, 257) -> transpose(-1,-2) -> (2, 257, T)
        m = self.erb.bs(m_feat).squeeze(0).transpose(-1, -2)  # (2, F, T)

        # Split on dim=0: each (1, F, T) — no extra squeeze needed
        m0, m1 = m.split(1, dim=0)

        # Complex Ratio Mask (fused, no separate Mask class)
        s_real = real_part * m0 - imag_part * m1
        s_imag = imag_part * m0 + real_part * m1

        return s_real, s_imag


if __name__ == "__main__":
    model = GTCRN().eval()
    model.fuse_bn_()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(flops, params)

    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    x2 = torch.stft(x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y1 = model(x1)[0]
    y2 = model(x2)[0]
    y1 = torch.istft(y1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y2 = torch.istft(y2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

    print((y1[:16000 - 256 * 2] - y2[:16000 - 256 * 2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
