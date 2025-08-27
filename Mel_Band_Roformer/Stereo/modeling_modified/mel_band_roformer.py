from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from modeling_modified.attend import Attend

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from librosa import filters


# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.neg_mask = torch.tensor([-1.0, 1.0], dtype=torch.float32)

    def rotate_half(self, x):
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x_flipped = torch.flip(x_reshaped, dims=[-1])
        x_rotated = x_flipped * self.neg_mask
        return x_rotated.flatten(start_dim=-2)

    def forward(self, x, rotary_cos, rotary_sin):
        x = self.norm(x)

        b, n, _ = x.shape

        # More direct QKV projection and splitting
        qkv = self.to_qkv(x).reshape(b, n, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        q = q * rotary_cos + self.rotate_half(q) * rotary_sin
        k = k * rotary_cos + self.rotate_half(k) * rotary_sin

        out = self.attend(q, k, v)

        gates = self.to_gates(x)

        # Apply sigmoid gates
        gated_mask = gates.permute(0, 2, 1).unsqueeze(-1).sigmoid()
        out = out * gated_mask

        # Combine heads and project out, using reshape for clarity
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_embed=rotary_embed,
                          flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, rotary_cos, rotary_sin):
        for attn, ff in self.layers:
            x = attn(x, rotary_cos, rotary_sin) + x
            x = ff(x) + x
        return self.norm(x)


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class

class MelBandRoformer(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            dim_freqs_in=1025,
            sample_rate=44100,  # needed for mel filter bank from librosa
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            ]))

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]
        self.num_freqs = freqs

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        mel_filter_bank[0][0] = 1.
        mel_filter_bank[-1, -1] = 1.

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        repeated_freq_indices = torch.arange(freqs).expand(num_bands, -1)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = freq_indices.unsqueeze(1).expand(-1, self.audio_channels)
            freq_indices = freq_indices * 2 + torch.arange(self.audio_channels)
            freq_indices = freq_indices.flatten()

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = freqs_per_band.sum(dim=1)
        num_bands_per_freq = freqs_per_band.sum(dim=0)

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

        denom = self.num_bands_per_freq.repeat_interleave(self.audio_channels).view(1, 1, -1, 1, 1)
        self.register_buffer('denom', 1.0 / denom.clamp(min=1e-8), persistent=False)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth
            )
            self.mask_estimators.append(mask_estimator)

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length

        position_ids = torch.arange(1024, dtype=torch.float32).unsqueeze(-1)  # [L, 1]; 1024 is about 10 seconds audio
        inv_freq = 10000.0 ** -(torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head)
        freqs = position_ids * inv_freq
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1).unsqueeze(0).unsqueeze(0)  # [L, D]
        self.cos_rotary_pos_emb = torch.cos(freqs)  # [1, D, L]
        self.sin_rotary_pos_emb = torch.sin(freqs)  # [1, D, L]
        self.rotary_cos_freq = self.cos_rotary_pos_emb[:, :, :60]
        self.rotary_sin_freq = self.sin_rotary_pos_emb[:, :, :60]
        self.cos_rotary_pos_emb = self.cos_rotary_pos_emb.half()
        self.sin_rotary_pos_emb = self.sin_rotary_pos_emb.half()

    def forward(
            self,
            stft_repr,
            zeros,
            target=None,
            return_loss_breakdown=False
    ):
        b, s, f, t, c = stft_repr.shape

        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, -1, t, c)

        x = stft_repr[:, self.freq_indices]

        x = x.permute(0, 2, 1, 3).reshape(b, t, -1)

        x = self.band_split(x)

        rotary_cos = self.cos_rotary_pos_emb[:, :, :t].float()
        rotary_sin = self.sin_rotary_pos_emb[:, :, :t].float()

        for time_transformer, freq_transformer in self.layers:
            b_t, t_t, f_t, d_t = x.shape
            x = x.permute(0, 2, 1, 3).reshape(-1, t_t, d_t)
            x = time_transformer(x, rotary_cos, rotary_sin)
            x = x.reshape(b_t, f_t, t_t, d_t).permute(0, 2, 1, 3)

            b_f, t_f, f_f, d_f = x.shape
            x = x.reshape(-1, f_f, d_f)
            x = freq_transformer(x, self.rotary_cos_freq, self.rotary_sin_freq)
            x = x.reshape(b_f, t_f, f_f, d_f)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)

        b_m, n_m, t_m, _ = masks.shape
        masks = masks.view(b_m, n_m, t_m, -1, 2).transpose(3, 2)

        stft_repr = stft_repr.unsqueeze(1)

        time_dim = stft_repr.shape[-2]
        scatter_indices = self.freq_indices.view(1, 1, -1, 1, 1).expand(-1, -1, -1, time_dim, 2)

        masks_summed = zeros[:, :, :, :time_dim].to(stft_repr.dtype).scatter_add_(2, scatter_indices, masks)

        masks_averaged = masks_summed * self.denom
        masked_stft = stft_repr * masks_averaged

        _, _, _, t_o, c_o = masked_stft.shape
        output_stft = masked_stft.transpose(1, 2).reshape(-1, self.num_freqs, t_o, c_o)

        return output_stft[..., 0], output_stft[..., 1]
