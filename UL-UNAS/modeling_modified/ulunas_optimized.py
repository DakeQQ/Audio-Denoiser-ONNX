"""
UL-UNAS Optimized: ONNX-exportable speech enhancement model.

Optimizations applied:
  - Removed torch.stft / torch.istft (handled externally by STFT_Process)
  - Removed einops dependency (replaced with view/reshape/permute)
  - Weight fusion: BatchNorm fused into Conv/ConvTranspose weights
  - Use split instead of chunk
  - Pre-compute: constants pre-computed and registered as buffers
  - Modified forward: accepts magnitude (1, F, T), returns mask (1, F, T)
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
        x = torch.mean(x.square(), dim=1)  # (B, T, F)

        x = nn.functional.pad(x, (0, self.pad_len))
        x = x.view(1, -1, self.H, self.r)

        # Replace einops: 'b t h c -> (b t) h c'
        x = x.reshape(-1, self.H, self.r)
        x, _ = self.gru(x)  # (BT, H, 2r)
        x = self.fc(x)      # (BT, H, r)

        x = x.reshape(1, 1, -1, self.F_pad)  # (B, T, F)

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
        _, C2, _, F = x.shape
        C = C2 // 2
        x1, x2 = x.split(C, dim=1)
        # Replace einops: stack then interleave
        # x1,x2 each (B,C,T,F) -> stack on dim=2 -> (B,C,2,T,F) -> reshape to (B,2C,T,F)
        x = torch.stack([x1, x2], dim=2)  # (B, C, 2, T, F)
        x = x.reshape(1, C2, -1, F)
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
        ## Intra RNN
        x_perm = x.permute(0, 2, 3, 1)  # (B, T, F, C)
        intra_in = x_perm.reshape(-1, self.width, self.input_size)  # (B*T, F, C)
        intra_x = self.intra_fc(self.intra_rnn(intra_in)[0])       # (B*T, F, C)
        intra_x = intra_x.view(1, -1, self.width, self.input_size)  # (1, T, F, C)
        intra_x = self.intra_ln(intra_x)
        intra_out = x_perm + intra_x  # (1, T, F, C)

        ## Inter RNN
        inter_in = intra_out.transpose(1, 2).reshape(-1, intra_out.shape[1], self.input_size)  # (F, T, C)
        inter_x = self.inter_fc(self.inter_rnn(inter_in)[0])  # (F, T, C)
        inter_x = self.inter_ln(inter_x.view(1, self.width, -1, self.input_size).transpose(1, 2))  # (1, T, F, C)
        inter_out = intra_out + inter_x  # (1, T, F, C)

        return inter_out.permute(0, 3, 1, 2)  # (1, C, T, F)


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
