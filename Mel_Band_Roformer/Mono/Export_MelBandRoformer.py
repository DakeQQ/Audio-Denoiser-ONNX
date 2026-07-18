import gc
import subprocess
import sys
import os
import tempfile
import yaml
from itertools import groupby
from pathlib import Path

from beartype import beartype
from beartype.typing import Tuple
from ml_collections import ConfigDict
import numpy as np
import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from STFT_Process import STFT_Process                                                             # The custom STFT/ISTFT can be exported in ONNX format.

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


project_path          = str(Path.home() / "Downloads" / "Mel-Band-Roformer-Vocal-Model-main")            # The Mel-Band-Roformer GitHub project path.
model_path            = str(Path(project_path) / "MelBandRoformer.ckpt")                                # The downloaded model.
config_path           = str(Path(project_path) / "configs" / "config_vocals_mel_band_roformer.yaml")    # The model configuration.
parent_path           = Path(__file__).resolve().parent                                                 # The folder that contains this script.
onnx_model_A          = str(parent_path / "MelBandRoformer_ONNX" / "MelBandRoformer.onnx")              # The exported onnx model path.
onnx_model_Metadata   = str(metadata_path_for_model(onnx_model_A))                                      # The metadata carrier onnx model path.

DYNAMIC_AXES          = False       # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET                 = 20          # Opset 20 exports exact GELU as one standard ONNX Gelu operator.
MODEL_SAMPLE_RATE     = 44100       # Mel-Band Roformer runs at 44.1kHz internally.
IN_SAMPLE_RATE        = 44100       # [8000, 16000, 22500, 24000, 44000, 48000]; input audio sample rate.
OUT_SAMPLE_RATE       = 44100       # [8000, 16000, 22500, 24000, 44000, 48000]; output audio sample rate.
INPUT_AUDIO_LENGTH    = 88200       # Maximum input audio length in IN_SAMPLE_RATE samples. Higher values yield better quality but time consume. It is better to set an integer multiple of the HOP_LENGTH value.
WINDOW_TYPE           = 'hann'      # Type of window function used in the STFT
NFFT                  = 2048        # Number of FFT components for the STFT process
WINDOW_LENGTH         = 2048        # Length of windowing, edit it carefully.
HOP_LENGTH            = 441         # Number of samples between successive frames in the STFT

USE_BATCH_FOLD        = True        # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
BATCH_WINDOW_SECONDS  = 1.5         # When the configured input length is >= this many seconds, fold into fixed-length windows and batch-process them together (each window is a separate clip -> per-window attention).
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length, rounded UP to a HOP multiple so the center-pad STFT -> ISTFT reconstructs W samples per window.
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop.
MAX_SIGNAL_LENGTH     = 2048 if DYNAMIC_AXES else (((FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH) // HOP_LENGTH) + 1)  # Max STFT frames (per-window count in fold mode). Sizes the scatter base and the ISTFT trim.

INPUT_TO_MODEL_SCALE  = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_AUDIO_LENGTH    = int(round(EXPORT_AUDIO_LENGTH * INPUT_TO_MODEL_SCALE))
OUTPUT_AUDIO_LENGTH   = int(round(EXPORT_AUDIO_LENGTH * INPUT_TO_OUTPUT_SCALE))

IN_AUDIO_DTYPE        = 'INT16'     # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE       = 'INT16'     # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16             = float(1.0 / 32768.0)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def l2_normalize(x):
    """Exact F.normalize lowering without exporter-generated Shape/Expand nodes."""
    return x / torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).clamp_min(1e-12)


def hz_to_mel(frequencies, htk=False):
    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def create_mel_filter_bank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney', dtype=np.float32):
    if fmax is None:
        fmax = float(sr) / 2

    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft // 2))
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 'slaney':
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"Unsupported mel filter norm: {norm}")

    return weights.astype(dtype, copy=False)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
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


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        dim_inner = heads * dim_head

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )


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
            norm_output=True
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()


class BandSplit(Module):
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
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
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)


def fold_stereo_to_mono(stereo, mono_layers, mono_band_split, mono_mask_estimators, num_freqs_per_band):
    """Fold a stereo checkpoint (loaded into `stereo`) into mono modules by averaging L/R.

    - Every channel-agnostic weight (all Transformer layers, the two uniform MaskEstimator
      MLP linears, and every matching-shape bias) is copied over verbatim.
    - The channel-dependent weights are averaged over the two channels per (real, imag) so a
      mono input behaves like the L/R average the stereo network was trained on: the BandSplit
      RMSNorm gamma / input Linear columns and the MaskEstimator pre-GLU output Linear rows.
    """
    def copy_matching(src, dst):
        # Copy every parameter/buffer whose shape is identical between src and dst.
        ssd, dsd = src.state_dict(), dst.state_dict()
        for k, v in dsd.items():
            if k in ssd and ssd[k].shape == v.shape:
                dsd[k] = ssd[k].clone()
        dst.load_state_dict(dsd, strict=False)

    with torch.no_grad():
        # 1) Copy all channel-agnostic weights directly (Transformer layers + the two uniform
        #    MaskEstimator MLP linears net[0]/net[2] + all matching-shape biases).
        copy_matching(stereo.layers, mono_layers)
        copy_matching(stereo.band_split, mono_band_split)
        for st_me, mo_me in zip(stereo.mask_estimators, mono_mask_estimators):
            copy_matching(st_me, mo_me)

        # 2) BandSplit: (4*fi) -> (2*fi) input features, grouped as [real_L, imag_L, real_R, imag_R].
        for b, (st_feat, mo_feat) in enumerate(zip(stereo.band_split.to_features, mono_band_split.to_features)):
            fi = int(num_freqs_per_band[b].item())

            gamma_st = st_feat[0].gamma.data.view(fi, 4)                          # [real_L, imag_L, real_R, imag_R]
            gamma_m = torch.stack([
                (gamma_st[:, 0] + gamma_st[:, 2]) * 0.5,                          # real
                (gamma_st[:, 1] + gamma_st[:, 3]) * 0.5,                          # imag
            ], dim=-1).reshape(-1)
            mo_feat[0].gamma.data.copy_(gamma_m)

            Wst = st_feat[1].weight.data                                         # (dim, 4*fi)
            dim_out = Wst.shape[0]
            Wst_v = Wst.view(dim_out, fi, 4)
            Wm = torch.empty((dim_out, fi, 2), dtype=Wst.dtype, device=Wst.device)
            Wm[:, :, 0] = (Wst_v[:, :, 0] + Wst_v[:, :, 2]) * 0.5                 # real
            Wm[:, :, 1] = (Wst_v[:, :, 1] + Wst_v[:, :, 3]) * 0.5                 # imag
            mo_feat[1].weight.data.copy_(Wm.reshape(dim_out, 2 * fi))
            mo_feat[1].bias.data.copy_(st_feat[1].bias.data)

        # MaskEstimator: pre-GLU Linear out_features (8*fi = 2 GLU halves x 4*fi) -> (4*fi).
        for st_me, mo_me in zip(stereo.mask_estimators, mono_mask_estimators):
            for b_idx, (st_mlp, mo_mlp) in enumerate(zip(st_me.to_freqs, mo_me.to_freqs)):
                st_last = [m for m in st_mlp[0] if isinstance(m, nn.Linear)][-1]
                mo_last = [m for m in mo_mlp[0] if isinstance(m, nn.Linear)][-1]

                fi = int(num_freqs_per_band[b_idx].item())
                Wst = st_last.weight.data                                         # (8*fi, hidden)
                bst = st_last.bias.data                                           # (8*fi,)
                hidden = Wst.shape[1]

                Wst_v = Wst.view(2, 4 * fi, hidden)                               # 2 GLU halves
                bst_v = bst.view(2, 4 * fi)

                def fold_rows(W_part):
                    Wp = W_part.view(fi, 4, hidden)                              # [real_L, imag_L, real_R, imag_R]
                    out = torch.empty(fi, 2, hidden, dtype=W_part.dtype, device=W_part.device)
                    out[:, 0] = (Wp[:, 0] + Wp[:, 2]) * 0.5                      # real
                    out[:, 1] = (Wp[:, 1] + Wp[:, 3]) * 0.5                      # imag
                    return out.view(2 * fi, hidden)

                def fold_bias(b_part):
                    bp = b_part.view(fi, 4)
                    out = torch.stack([(bp[:, 0] + bp[:, 2]) * 0.5, (bp[:, 1] + bp[:, 3]) * 0.5], dim=-1)
                    return out.view(2 * fi)

                W_new = torch.cat([fold_rows(Wst_v[0]), fold_rows(Wst_v[1])], dim=0)   # (4*fi, hidden)
                b_new = torch.cat([fold_bias(bst_v[0]), fold_bias(bst_v[1])], dim=0)   # (4*fi,)
                mo_last.weight.data.copy_(W_new)
                mo_last.bias.data.copy_(b_new)


class MelBandRoformer(torch.nn.Module):
    def __init__(
            self,
            stft_model,
            istft_model,
            max_signal_len,
            use_batch_fold=False,
            fold_window=0,
            *,
            dim,
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
            sample_rate=44100,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn=None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn=torch.hann_window,
            match_input_audio_length=False,
    ):
        super().__init__()
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.dynamic = DYNAMIC_AXES
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        if "int" in IN_AUDIO_DTYPE.lower():
            self.stft_model.stft_kernel.mul_(INV_INT16)

        # ---- This is the MONO export. The trained checkpoint is STEREO; the mono
        # weights are FOLDED from it by averaging the L/R channels of the BandSplit
        # and MaskEstimator branches (the channel-agnostic Transformer layers copy
        # over directly). Two checkpoint-shaped holders are built here: `st` (stereo)
        # receives the trained weights, and the mono holder `m` receives the averaged
        # ones. Both are throwaway holders — every fused buffer below is derived from
        # `m`, after which the originals go out of scope and are freed. The helper
        # classes (Attention/FeedForward/Transformer/BandSplit/MaskEstimator) exist
        # only to materialise parameters whose names match the checkpoint; none of
        # their forwards are used. ----
        audio_channels = 1  # mono output regardless of the config's `stereo` flag

        transformer_kwargs = dict(
            dim=dim, heads=heads, dim_head=dim_head,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )

        def build_layers():
            layers = ModuleList([])
            for _ in range(depth):
                layers.append(nn.ModuleList([
                    Transformer(depth=time_transformer_depth, **transformer_kwargs),
                    Transformer(depth=freq_transformer_depth, **transformer_kwargs)
                ]))
            return layers

        num_freqs = stft_n_fft // 2 + 1

        mel_filter_bank = torch.from_numpy(
            create_mel_filter_bank(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)
        )
        mel_filter_bank[0][0] = 1.
        mel_filter_bank[-1, -1] = 1.
        freqs_per_band = mel_filter_bank > 0

        # Mono frequency selection: no channel interleave (unlike the stereo variant).
        freq_indices = torch.arange(num_freqs).expand(num_bands, -1)[freqs_per_band]

        num_freqs_per_band = freqs_per_band.sum(dim=1)
        num_bands_per_freq = freqs_per_band.sum(dim=0)
        denom = num_bands_per_freq.repeat_interleave(audio_channels).view(1, -1, 1, 1)
        denom = 1.0 / denom.clamp(min=1e-8)
        dim_inputs = tuple(2 * f * audio_channels for f in num_freqs_per_band.tolist())        # mono band widths
        stereo_dim_inputs = tuple(2 * f * 2 for f in num_freqs_per_band.tolist())              # checkpoint (stereo) band widths

        # ---- Load the stereo checkpoint, then fold it into mono modules ----
        layers = build_layers()
        band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        mask_estimators = ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dim_inputs, depth=mask_estimator_depth)
            for _ in range(num_stems)
        ])

        st = nn.Module()
        st.layers = build_layers()
        st.band_split = BandSplit(dim=dim, dim_inputs=stereo_dim_inputs)
        st.mask_estimators = ModuleList([
            MaskEstimator(dim=dim, dim_inputs=stereo_dim_inputs, depth=mask_estimator_depth)
            for _ in range(num_stems)
        ])
        st.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True), strict=False)

        fold_stereo_to_mono(st, layers, band_split, mask_estimators, num_freqs_per_band)

        rotary_length = max(1024, int(max_signal_len), int(num_bands))
        position_ids = torch.arange(rotary_length, dtype=torch.float32).unsqueeze(-1)
        inv_freq = 10000.0 ** -(torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head)
        rotary = torch.repeat_interleave(position_ids * inv_freq, repeats=2, dim=-1).unsqueeze(0).unsqueeze(0)
        cos_rotary_pos_emb = torch.cos(rotary)
        sin_rotary_pos_emb = torch.sin(rotary)
        rotary_cos_freq = cos_rotary_pos_emb[:, :, :num_bands]
        rotary_sin_freq = sin_rotary_pos_emb[:, :, :num_bands]

        m = nn.Module()
        m.audio_channels = audio_channels
        m.num_freqs = int(num_freqs)
        m.freq_indices = freq_indices
        m.denom = denom
        m.layers = layers
        m.band_split = band_split
        m.mask_estimators = mask_estimators
        m.cos_rotary_pos_emb = cos_rotary_pos_emb.half()
        m.sin_rotary_pos_emb = sin_rotary_pos_emb.half()
        m.rotary_cos_freq = rotary_cos_freq
        m.rotary_sin_freq = rotary_sin_freq
        m.eval()

        # ---- Static dimensions (read once here, reused as python ints in the forward
        # pass so the exported graph carries no Shape / Gather / Range meta ops) ----
        attn0 = m.layers[0][0].layers[0][0]
        self.heads = int(attn0.heads)
        self.dim_head = int(attn0.dim_head)
        self.dim_inner = self.heads * self.dim_head
        self.dim = int(attn0.to_out[0].weight.shape[0])
        self.depth = len(m.layers)
        self.num_freqs = int(m.num_freqs)
        self.audio_channels = int(m.audio_channels)
        self.freq_complex = self.num_freqs * self.audio_channels
        self.num_selected = int(m.freq_indices.shape[0])
        self.dim_inputs = tuple(int(d) for d in m.band_split.dim_inputs)
        self.n_bands = len(self.dim_inputs)
        self.static_frames = int(max_signal_len)
        self.fold_batch = int(EXPORT_AUDIO_LENGTH // fold_window) if use_batch_fold else 1
        if use_batch_fold and EXPORT_AUDIO_LENGTH % fold_window:
            raise ValueError("EXPORT_AUDIO_LENGTH must be divisible by fold_window")
        if self.dynamic and use_batch_fold:
            raise ValueError("Dynamic axes and batch folding cannot be enabled together")

        # ---- Frequency gather / reconstruction buffers ----
        # The per-output-frequency averaging is folded into the MaskEstimator GLU value
        # branch. In the static graph each output bin receives at most two source masks,
        # so a tiny inverse Gather replaces ScatterElements and its multi-megabyte indices.
        self.register_buffer('freq_indices', m.freq_indices.to(torch.int32).contiguous())
        if self.dynamic:
            self.register_buffer('scatter_indices', m.freq_indices.view(1, -1, 1, 1).to(torch.int64).contiguous())
            self.register_buffer('scatter_base', torch.zeros((1, self.freq_complex, self.static_frames, 2), dtype=torch.float32))
        else:
            source_positions = [torch.nonzero(m.freq_indices == i, as_tuple=False).flatten() for i in range(self.freq_complex)]
            self.max_band_overlap = max(len(positions) for positions in source_positions)
            inverse_indices = torch.full(
                (self.max_band_overlap, self.freq_complex),
                self.num_selected,
                dtype=torch.int32,
            )
            for i, positions in enumerate(source_positions):
                inverse_indices[:len(positions), i] = positions.to(torch.int32)
            self.register_buffer('inverse_freq_indices', inverse_indices.flatten().contiguous())
            self.register_buffer('zero_mask', torch.zeros((self.fold_batch, 1, self.static_frames, 2), dtype=torch.float32))

        # ---- Rotary tables ----
        # The GPT-J alternating sign [-1, +1, ...] is folded into the sin tables so
        # rotate_half is one fixed int32 Gather rather than flip's large Slice subgraph.
        rot_sign = torch.ones(self.dim_head, dtype=torch.float32)
        rot_sign[0::2] = -1.0
        rotary_indices = torch.arange(self.dim_head, dtype=torch.int32).reshape(-1, 2).flip(1).reshape(-1)
        self.register_buffer('rotary_indices', rotary_indices.contiguous())
        self.register_buffer('time_cos', m.cos_rotary_pos_emb[:, :, :self.static_frames].float().contiguous())
        self.register_buffer('time_sin', (m.sin_rotary_pos_emb[:, :, :self.static_frames].float() * rot_sign).half().float().contiguous())
        self.register_buffer('freq_cos', m.rotary_cos_freq.float().contiguous())
        self.register_buffer('freq_sin', (m.rotary_sin_freq.float() * rot_sign).contiguous())

        # ---- BandSplit: fold each band RMSNorm (scale * gamma) into its Linear ----
        self._bs_w, self._bs_b = [], []
        for i, net in enumerate(m.band_split.to_features):
            rms, lin = net[0], net[1]
            g = rms.scale * rms.gamma.detach().double()
            self.register_buffer(f'bs_w_{i}', (lin.weight.detach().double() * g.unsqueeze(0)).float().contiguous())
            self.register_buffer(f'bs_b_{i}', lin.bias.detach().float().contiguous())
            self._bs_w.append(getattr(self, f'bs_w_{i}'))
            self._bs_b.append(getattr(self, f'bs_b_{i}'))

        # Consecutive equal-width bands share one batched MatMul. This preserves the
        # original band order without padding narrow bands to the 260-wide maximum.
        self._band_runs = []
        start = 0
        for run_index, (dim_in, grouped_dims) in enumerate(groupby(self.dim_inputs)):
            run_count = len(tuple(grouped_dims))
            self._band_runs.append((start, run_count, dim_in))
            if run_count > 1:
                run_w = torch.stack([self._bs_w[i].transpose(0, 1) for i in range(start, start + run_count)], dim=0).unsqueeze(1)
                run_b = torch.stack([self._bs_b[i] for i in range(start, start + run_count)], dim=0).unsqueeze(1).unsqueeze(1)
                self.register_buffer(f'bs_run_w_{run_index}', run_w.contiguous())
                self.register_buffer(f'bs_run_b_{run_index}', run_b.contiguous())
            start += run_count

        # ---- Transformer layers: fuse norm + qkv + gates, fold attention scale ----
        self._time_tf = [self._fuse_transformer(m.layers[i][0], f'time{i}') for i in range(self.depth)]
        self._freq_tf = [self._fuse_transformer(m.layers[i][1], f'freq{i}') for i in range(self.depth)]

        # ---- MaskEstimator: batch the two uniform MLP linears (384->1536->1536)
        # across the 60 bands with bmm; the per-band output linear (variable width)
        # stays a small loop feeding GLU. ----
        # The scatter-average denom is fused into each band's GLU value branch:
        # GLU(x) = a * sigmoid(b), so scaling the `a` rows (first half of net[4]) by the
        # destination-frequency denom makes scatter_add(masks) already averaged. The
        # per-source denom == denom[freq_indices[source]] (shared by the real & imag
        # halves), repeat_interleaved by 2 to match the (source, complex) cat layout.
        denom_val = m.denom.detach().double().reshape(-1)[m.freq_indices.long()].repeat_interleave(2)
        me = m.mask_estimators[0]
        w1, b1, w2, b2 = [], [], [], []
        self._me_w3, self._me_b3 = [], []
        offset = 0
        for i, mlp in enumerate(me.to_freqs):
            net = mlp[0]
            w1.append(net[0].weight.detach())
            b1.append(net[0].bias.detach())
            w2.append(net[2].weight.detach())
            b2.append(net[2].bias.detach())
            fi = self.dim_inputs[i]
            d_i = denom_val[offset:offset + fi]                      # (dim_in_i,) GLU value-branch scale
            offset += fi
            w3 = net[4].weight.detach().double()
            b3 = net[4].bias.detach().double()
            w3[:fi] *= d_i.unsqueeze(1)
            b3[:fi] *= d_i
            self.register_buffer(f'me_w3_{i}', w3.float().contiguous())
            self.register_buffer(f'me_b3_{i}', b3.float().contiguous())
            self._me_w3.append(getattr(self, f'me_w3_{i}'))
            self._me_b3.append(getattr(self, f'me_b3_{i}'))
        self.register_buffer('me_w1t', torch.stack(w1, 0).transpose(1, 2).float().contiguous())   # (60, 384, 1536)
        self.register_buffer('me_b1', torch.stack(b1, 0).unsqueeze(1).float().contiguous())        # (60, 1, 1536)
        self.register_buffer('me_w2t', torch.stack(w2, 0).transpose(1, 2).float().contiguous())   # (60, 1536, 1536)
        self.register_buffer('me_b2', torch.stack(b2, 0).unsqueeze(1).float().contiguous())        # (60, 1, 1536)

    def _fuse_transformer(self, tf, prefix):
        attn = tf.layers[0][0]
        ff = tf.layers[0][1]
        di = self.dim_inner
        scale = self.dim_head ** -0.5
        # RMSNorm (scale * gamma) folded into the projection that consumes it
        g_in = attn.norm.scale * attn.norm.gamma.detach().double()
        wqkv = attn.to_qkv.weight.detach().double()
        wq, wk, wv = wqkv[:di], wqkv[di:2 * di], wqkv[2 * di:3 * di]
        wg = attn.to_gates.weight.detach().double()
        # The attention scale (dim_head ** -0.5) folds into the query rows (rotary is
        # linear in q so pre-scaling is equivalent), so the runtime *scale disappears.
        in_w = (torch.cat([wq * scale, wk, wv, wg], dim=0) * g_in.unsqueeze(0)).float().contiguous()
        in_b = torch.cat([torch.zeros(3 * di, dtype=torch.float64), attn.to_gates.bias.detach().double()]).float().contiguous()
        out_w = attn.to_out[0].weight.detach().float().contiguous()
        g_ff = ff.net[0].scale * ff.net[0].gamma.detach().double()
        ff1_w = (ff.net[1].weight.detach().double() * g_ff.unsqueeze(0)).float().contiguous()
        ff1_b = ff.net[1].bias.detach().float().contiguous()
        ff2_w = ff.net[4].weight.detach().float().contiguous()
        ff2_b = ff.net[4].bias.detach().float().contiguous()
        out_g = (tf.norm.scale * tf.norm.gamma.detach().double()).float().contiguous()
        params = dict(in_w=in_w, in_b=in_b, out_w=out_w, ff1_w=ff1_w, ff1_b=ff1_b, ff2_w=ff2_w, ff2_b=ff2_b, out_g=out_g)
        refs = {}
        for key, val in params.items():
            name = f'{prefix}_{key}'
            self.register_buffer(name, val)
            refs[key] = getattr(self, name)
        return refs

    def _attention(self, x, p, rcos, rsin, b):
        normed = l2_normalize(x)
        qkvg = F.linear(normed, p['in_w'], p['in_b'])
        di = self.dim_inner
        # One Split separates the fused projection into the qkv block and the gates.
        qkv_flat, gates = qkvg.split([3 * di, self.heads], dim=-1)
        # (3, b, heads, n, dim_head); split q/k (one batched rotary) from v.
        qkv = qkv_flat.reshape(b, -1, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        qk, v = qkv.split([2, 1], dim=0)                                # (2, b, heads, n, dim_head), (1, ...)
        qk = qk * rcos + torch.index_select(qk, -1, self.rotary_indices) * rsin # one batched rotary for q and k
        q, k = qk.split(1, dim=0)                                       # each (1, b, heads, n, dim_head)
        attn = torch.matmul(q, k.transpose(-1, -2)).softmax(dim=-1)
        # Transpose heads out first so the gate (b, n, heads, 1) needs no transpose,
        # then merge heads with a single contiguous reshape.
        out = torch.matmul(attn, v).transpose(2, 3)                     # (b, n, heads, dim_head)
        gates = gates.unsqueeze(0).unsqueeze(-1).sigmoid()                          # (b, n, heads, 1)
        out = (out * gates).reshape(b, -1, di)                          # (b, n, dim_inner)
        return F.linear(out, p['out_w'])

    def _feedforward(self, x, p):
        h = F.gelu(F.linear(l2_normalize(x), p['ff1_w'], p['ff1_b']))
        return F.linear(h, p['ff2_w'], p['ff2_b'])

    def _transformer(self, x, p, rcos, rsin, b):
        x = x + self._attention(x, p, rcos, rsin, b)
        x = x + self._feedforward(x, p)
        return l2_normalize(x) * p['out_g']

    def _band_split(self, x):
        run_sizes = [run_count * dim_in for _, run_count, dim_in in self._band_runs]
        parts = x.split(run_sizes, dim=-1)
        outs = []
        for run_index, (part, (start, run_count, dim_in)) in enumerate(zip(parts, self._band_runs)):
            if run_count == 1:
                outs.append(F.linear(l2_normalize(part), self._bs_w[start], self._bs_b[start]).unsqueeze(0))
                continue
            if self.dynamic:
                run_x = part.reshape(part.shape[0], part.shape[1], run_count, dim_in).permute(2, 0, 1, 3)
            else:
                run_x = part.reshape(self.fold_batch, self.static_frames, run_count, dim_in).permute(2, 0, 1, 3)
            run_out = torch.matmul(l2_normalize(run_x), getattr(self, f'bs_run_w_{run_index}'))
            outs.append(run_out + getattr(self, f'bs_run_b_{run_index}'))
        return torch.cat(outs, dim=0)                                    # (n_bands, B, t, dim)

    def _mask_estimator(self, x):
        # x arrives as (n_bands, t, dim) so the band axis is already the bmm batch (no permute).
        h = torch.tanh(torch.baddbmm(self.me_b1, x, self.me_w1t))        # (60, t, 1536)
        h = torch.tanh(torch.baddbmm(self.me_b2, h, self.me_w2t))        # (60, t, 1536)
        outs = [F.glu(F.linear(h[i], self._me_w3[i], self._me_b3[i]), dim=-1) for i in range(self.n_bands)]
        return torch.cat(outs, dim=-1)                                   # (t, num_selected * 2)

    def _core(self, stft_repr):
        # stft_repr arrives as (B*chan, num_freqs, t, 2). B = number of folded windows (1 in the
        # non-fold path). Each window is an independent clip: the audio-channel axis is interleaved
        # into freq_complex, and the window batch B rides alongside the band / time batch dims.
        t = stft_repr.shape[-2] if self.dynamic else self.static_frames
        B = self.fold_batch if self.use_batch_fold else 1
        # Mono already arrives in the required (window, frequency, frame, complex)
        # layout; the generic channel-interleave reshape/transpose round trip is dead.
        x = torch.index_select(stft_repr, 1, self.freq_indices)
        x = x.transpose(1, 2).reshape(B, t, self.num_selected * 2)
        x = self._band_split(x)                                          # (n_bands, B, t, dim)

        tcos = self.time_cos[:, :, :t] if self.dynamic else self.time_cos
        tsin = self.time_sin[:, :, :t] if self.dynamic else self.time_sin
        # Axial attention: time attends over t (batch = n_bands*B), freq attends over bands
        # (batch = t*B). Folding B into each batch keeps every window independent.
        nb = self.n_bands
        for i in range(self.depth):
            x = x.reshape(nb * B, t, self.dim)
            x = self._transformer(x, self._time_tf[i], tcos, tsin, nb * B)
            x = x.reshape(nb, B, t, self.dim).permute(2, 1, 0, 3).reshape(t * B, nb, self.dim)
            x = self._transformer(x, self._freq_tf[i], self.freq_cos, self.freq_sin, t * B)
            x = x.reshape(t, B, nb, self.dim).permute(2, 1, 0, 3)

        masks = self._mask_estimator(x.reshape(nb, B * t, self.dim)).view(B, t, self.num_selected, 2).transpose(1, 2)
        if self.dynamic:
            scatter_indices = self.scatter_indices.expand(B, -1, t, 2)
            masks_averaged = self.scatter_base[:, :, :t].scatter_add(1, scatter_indices, masks)
        else:
            masks_averaged = torch.index_select(
                torch.cat((masks, self.zero_mask), dim=1),
                1,
                self.inverse_freq_indices,
            ).reshape(B, self.max_band_overlap, self.freq_complex, t, 2).sum(dim=1)

        real_in, imag_in = stft_repr[..., 0], stft_repr[..., 1]
        mask_real, mask_imag = masks_averaged[..., 0], masks_averaged[..., 1]
        masked_real = real_in * mask_real - imag_in * mask_imag
        masked_imag = real_in * mask_imag + imag_in * mask_real
        return masked_real, masked_imag

    def forward(self, audio):
        audio = audio.float()
        if INPUT_TO_MODEL_SCALE < 1.0:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        if INPUT_TO_MODEL_SCALE > 1.0:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            channel_batch = audio.reshape(self.fold_batch, 1, self.fold_window)
        else:
            channel_batch = audio
        real_stft, imag_stft = self.stft_model(channel_batch)
        stft_repr = torch.stack((real_stft, imag_stft), dim=-1)         # (B*chan, freq, t, complex)
        real_part, imag_part = self._core(stft_repr)
        if self.use_batch_fold:
            audio = self.istft_model(real_part, imag_part).reshape(1, 1, EXPORT_AUDIO_LENGTH)
        else:
            audio = self.istft_model(real_part, imag_part)
        if MODEL_TO_OUTPUT_SCALE < 1.0:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=MODEL_TO_OUTPUT_SCALE,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            audio *= 32767.0
        if MODEL_TO_OUTPUT_SCALE > 1.0:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=MODEL_TO_OUTPUT_SCALE,
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
    inference_script = Path(__file__).resolve().with_name('Inference_MelBandRoformer_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    static_model_input_length = FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else MODEL_AUDIO_LENGTH
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0 if DYNAMIC_AXES else MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect', precompute_static=not DYNAMIC_AXES, static_input_length=static_model_input_length).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect', precompute_static=not DYNAMIC_AXES).eval()

    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Model configuration not found: {config_path}")
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading configuration from: {config_path}")

    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    mel_band_roformer = MelBandRoformer(custom_stft, custom_istft, MAX_SIGNAL_LENGTH, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH, **dict(config.model))
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    final_onnx_path = Path(onnx_model_A).expanduser().resolve()
    temporary_export = tempfile.TemporaryDirectory(prefix=f'.{final_onnx_path.stem}-', dir=final_onnx_path.parent)
    temporary_onnx_path = Path(temporary_export.name) / final_onnx_path.name
    try:
        torch.onnx.export(
            mel_band_roformer,
            (audio,),
            str(temporary_onnx_path),
            input_names=['noisy_audio'],
            output_names=['denoised_audio'],
            dynamic_axes={
                'noisy_audio': {2: 'audio_len'},
                'denoised_audio': {2: 'out_audio_len'}
            } if DYNAMIC_AXES else None,
            opset_version=OPSET,
            dynamo=False
        )
        os.replace(temporary_onnx_path, final_onnx_path)
    finally:
        temporary_export.cleanup()
    del mel_band_roformer
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
    
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="MelBandRoformer_Mono", task="denoise", model_family="mel_band_roformer",
    max_dynamic_audio_seconds=6, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=1, feature_kind="stft_mel_band", center_pad=True, pad_mode="reflect",
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
# Remove the persistent raw artifact created by older exporter versions.
Path(onnx_model_A).with_name(f'{Path(onnx_model_A).stem}.raw{Path(onnx_model_A).suffix}').unlink(missing_ok=True)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
