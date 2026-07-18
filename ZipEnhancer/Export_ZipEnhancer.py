import gc
import json
import shutil
import subprocess
import sys
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from modelscope.models.base import Model
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata
from Rewrite_ONNX_Asymmetric_Padding import rewrite_asymmetric_causal_convs


# --- File paths -------------------------------------------------------------
model_path          = str(Path.home() / "Downloads" / "speech_zipenhancer_ans_multiloss_16k_base")  # The ZipEnhancer download path.
parent_path         = Path(__file__).resolve().parent                                    # The folder that contains this script.
onnx_model_A        = str(parent_path / "ZipEnhancer_ONNX" / "ZipEnhancer.onnx")        # The exported onnx model path.
onnx_model_Metadata = str(metadata_path_for_model(onnx_model_A))                         # The metadata carrier onnx model path.
onnx_rewrite_report = str(parent_path / "ZipEnhancer_ONNX" / "ZipEnhancer.rewrite_report.json")

# --- Export settings --------------------------------------------------------
DYNAMIC_AXES = False              # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET        = 20                 # The ONNX opset version to export.

# --- Audio I/O dtype --------------------------------------------------------
IN_AUDIO_DTYPE  = 'INT16'         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE = 'INT16'         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.

# --- Sample rates -----------------------------------------------------------
MODEL_SAMPLE_RATE = 16000         # ZipEnhancer runs at 16kHz internally.
IN_SAMPLE_RATE    = 16000         # [8000, 16000, 22500, 24000, 44100, 48000]; input audio sample rate.
OUT_SAMPLE_RATE   = 16000         # [8000, 16000, 22500, 24000, 44100, 48000]; output audio sample rate.

# --- STFT / audio parameters ------------------------------------------------
INPUT_AUDIO_LENGTH = 32000        # Maximum input audio length in IN_SAMPLE_RATE samples. Recommended to be greater than 4800 and less than 48000. Higher values yield better quality but time consume. It is better to set an integer multiple of the NFFT value.
WINDOW_TYPE        = 'hann'       # Type of window function used in the STFT. ZipEnhancer's mag_pha_stft/mag_pha_istft use torch.hann_window, so this must be 'hann' (not 'hamming').
N_MELS             = 100          # Number of Mel bands to generate in the Mel-spectrogram
NFFT               = 400          # Number of FFT components for the STFT process
WINDOW_LENGTH      = 400          # Length of windowing, edit it carefully.
HOP_LENGTH         = 100          # Number of samples between successive frames in the STFT

# --- Derived constants ------------------------------------------------------
INPUT_TO_MODEL_SCALE  = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_AUDIO_LENGTH    = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * INPUT_TO_MODEL_SCALE))
OUTPUT_AUDIO_LENGTH   = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * INPUT_TO_OUTPUT_SCALE))
BATCH_WINDOW_SECONDS  = 1.5        # When the configured input audio length is >= this many seconds, fold into fixed-length windows and batch-process them together to accelerate inference. Dual-path attention then runs per window.
USE_BATCH_FOLD        = True       # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length, rounded UP to a HOP multiple so STFT (center_pad) -> ISTFT reconstructs exactly W samples per window.
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop.
MAX_SIGNAL_LENGTH     = 1024 if DYNAMIC_AXES else ((FOLD_WINDOW_LENGTH // HOP_LENGTH + 1) if USE_BATCH_FOLD else (MODEL_AUDIO_LENGTH // HOP_LENGTH + 1))  # Max STFT frames (per-window count in fold mode). Sizes the precomputed dual-path attention position tables AND the ISTFT COLA trim; for static export it must equal the real per-window frame count.
INV_INT16             = float(1.0 / 32768.0)
STATIC_SHAPE          = not DYNAMIC_AXES
USE_RECTANGULAR_ISTFT = True       # CUDA EP 1.27 places Atan on CPU; rectangular ISTFT removes the decoder-side Atan fallback.


def _validate_export_configuration():
    supported_dtypes = {'F16', 'F32', 'INT16'}
    if IN_AUDIO_DTYPE.upper() not in supported_dtypes:
        raise ValueError(f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}.")
    if OUT_AUDIO_DTYPE.upper() not in supported_dtypes:
        raise ValueError(f"Unsupported OUT_AUDIO_DTYPE={OUT_AUDIO_DTYPE!r}.")
    if not (0 < HOP_LENGTH <= WINDOW_LENGTH <= NFFT):
        raise ValueError("Expected 0 < HOP_LENGTH <= WINDOW_LENGTH <= NFFT.")
    if min(IN_SAMPLE_RATE, MODEL_SAMPLE_RATE, OUT_SAMPLE_RATE) <= 0:
        raise ValueError("All sample rates must be positive.")
    if USE_BATCH_FOLD:
        if DYNAMIC_AXES:
            raise ValueError("Batch folding requires a static export.")
        if not (IN_SAMPLE_RATE == MODEL_SAMPLE_RATE == OUT_SAMPLE_RATE):
            raise ValueError("Batch folding requires equal input, model, and output sample rates.")
        if INPUT_AUDIO_LENGTH < int(BATCH_WINDOW_SECONDS * IN_SAMPLE_RATE):
            raise ValueError("INPUT_AUDIO_LENGTH is too short for the configured fold window.")
        if FOLD_WINDOW_LENGTH % HOP_LENGTH != 0:
            raise ValueError("FOLD_WINDOW_LENGTH must be divisible by HOP_LENGTH.")
        if EXPORT_AUDIO_LENGTH % FOLD_WINDOW_LENGTH != 0:
            raise ValueError("EXPORT_AUDIO_LENGTH must contain whole fold windows.")
    expected_frames = (
        FOLD_WINDOW_LENGTH // HOP_LENGTH + 1
        if USE_BATCH_FOLD else MODEL_AUDIO_LENGTH // HOP_LENGTH + 1
    )
    if STATIC_SHAPE and MAX_SIGNAL_LENGTH != expected_frames:
        raise ValueError(
            f"MAX_SIGNAL_LENGTH={MAX_SIGNAL_LENGTH} does not match the static "
            f"STFT frame count {expected_frames}."
        )


def _sz(t, dim):
    """Dimension `dim` of `t` as a constant int for static export, else the symbolic size."""
    return int(t.shape[dim]) if STATIC_SHAPE else t.shape[dim]


def _ceil_div(x, y):
    return (x + y - 1) // y


# ---------------------------------------------------------------------------
# Export-friendly forward overrides (previously provided by ./modeling_modified).
# Each function below replaces a single ModelScope ZipEnhancer submodule forward
# with a simplified, ONNX-export-friendly version. In eval/tracing mode they are
# mathematically identical to the originals: training-only dropout, per-sequence
# layer-skipping randomness and the diagnostic Balancer/Whiten (Identity) ops are
# removed. The data-dependent positional encoding is handled directly in the
# wrapper (ZipEnhancer._pos_enc), so CompactRelPositionalEncoding is never called.
# All other submodules are used unchanged from the installed ModelScope package.
# ---------------------------------------------------------------------------
def _biasnorm_forward(self, x):
    # BiasNorm: cheaper LayerNorm replacement; channel_dim == -1 (== 2 for the
    # (seq, batch, channel) tensors used throughout this model).
    if hasattr(self, 'onnx_l2_scale'):
        channel_dim = self.channel_dim if self.channel_dim >= 0 else self.channel_dim + x.ndim
        bias = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        norm = torch.linalg.vector_norm(x - bias, ord=2, dim=channel_dim, keepdim=True)
        return (x / norm) * self.onnx_l2_scale
    return self.log_scale.exp() * (x / torch.sqrt((x - self.bias).pow(2).mean(dim=self.channel_dim, keepdim=True)))


def _activation_dropout_and_linear_forward(self, x):
    # SwooshL/SwooshR activation (dropout is a no-op in eval) -> linear. The
    # activation's constant offset is folded into onnx_bias during export
    # preparation, leaving only the input-dependent Softplus - 0.08*x path.
    if self.activation == 'SwooshL':
        x = torch.nn.functional.softplus(x - 4.0) - 0.08 * x
    else:
        x = torch.nn.functional.softplus(x - 1.0) - 0.08 * x
    bias = self.onnx_bias if hasattr(self, 'onnx_bias') else self.bias
    return torch.nn.functional.linear(x, self.weight, bias)


def _zipformer2_encoder_layer_forward(self, src, pos_emb, chunk_size=-1, attn_mask=None, src_key_padding_mask=None):
    # Export layout is (batch, sequence, channel). It keeps NonlinAttention's
    # value path batch-major and removes two full-tensor transposes per layer.
    src_orig = src
    if hasattr(self, 'onnx_attn_ff1_weight'):
        projected = torch.nn.functional.linear(
            src, self.onnx_attn_ff1_weight, self.onnx_attn_ff1_bias)
        attn_projected, ff1_projected = torch.split(
            projected,
            (self.onnx_attn_projection_size, self.onnx_ff1_projection_size),
            dim=-1)
        attn_weights = self.self_attn_weights(
            attn_projected,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            projected=True)
        src = src + self.feed_forward1.out_proj(ff1_projected)
    else:
        attn_weights = self.self_attn_weights(src, pos_emb=pos_emb, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.feed_forward1(src)
    # NonlinAttention uses only head zero. Select it as a rank-3 tensor here so
    # its value path stays (batch, sequence, channel) and avoids a singleton-head
    # Slice plus two rank-4 reshape/transpose round-trips in every encoder layer.
    src = src + self.nonlin_attention(src, attn_weights[:, 0])
    src = src + self.self_attn1(src, attn_weights)
    src = src + self.conv_module1(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)
    src = src + self.feed_forward2(src)
    src = self.bypass_mid(src_orig, src)
    src = src + self.self_attn2(src, attn_weights)
    src = src + self.conv_module2(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)
    src = src + self.feed_forward3(src)
    if hasattr(self, 'onnx_final_norm_scale'):
        # Fuse BiasNorm, the layer's final bypass, and the enclosing dual-path
        # bypass.  Both bypasses share src_orig, so their scales multiply.
        norm = torch.linalg.vector_norm(
            src - self.norm.bias, ord=2, dim=-1, keepdim=True)
        src = (
            (src / norm) * self.onnx_final_norm_scale
            + src_orig * self.onnx_final_residual_scale
        )
    else:
        src = self.norm(src)
        src = self.bypass(src_orig, src)
    return src


def _bypass_forward(self, src_orig, src):
    return src_orig + (src - src_orig) * self.bypass_scale


def _simple_downsample_forward(self, src):
    batch_size, seq_len, in_channels = _sz(src, 0), _sz(src, 1), _sz(src, 2)
    ds = self.downsample
    d_seq_len = (seq_len + ds - 1) // ds
    pad_len = d_seq_len * ds - seq_len
    if STATIC_SHAPE and pad_len == 0:
        # Frame count divisible by the factor -> the tail pad is empty, so the
        # expand + concat is a static no-op and is dropped from the graph.
        src = src.reshape(batch_size, d_seq_len, ds, in_channels)
    elif STATIC_SHAPE and pad_len == 1:
        # The ZipEnhancer static shapes all pad by at most one frame for the
        # downsampled encoders. The slice already has shape (B, 1, C), so an
        # Expand subgraph would be a no-op.
        src = torch.cat((src, src[:, seq_len - 1:]), dim=1).reshape(
            batch_size, d_seq_len, ds, in_channels)
    else:
        src_extra = src[:, seq_len - 1:].expand(batch_size, pad_len, in_channels)
        src = torch.cat((src, src_extra), dim=1).reshape(
            batch_size, d_seq_len, ds, in_channels)
    weights = (
        self.onnx_downsample_weights
        if hasattr(self, 'onnx_downsample_weights')
        else self.bias.softmax(dim=0).reshape(1, 1, ds, 1)
    )
    return (src * weights).sum(dim=2)


def _simple_upsample_forward(self, src):
    batch_size, seq_len, num_channels = _sz(src, 0), _sz(src, 1), _sz(src, 2)
    if STATIC_SHAPE:
        src = src.unsqueeze(2)
        return torch.cat((src,) * self.upsample, dim=2).reshape(
            batch_size, -1, num_channels)
    return src.unsqueeze(2).expand(
        batch_size, seq_len, self.upsample, num_channels).reshape(
            batch_size, -1, num_channels)


def _rel_pos_mha_weights_forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None, projected=False):
    # Export-only RelPositionMultiheadAttentionWeights.forward: keeps just the
    # torch.jit.is_tracing() relative-shift branch (the eager as_strided path is not
    # ONNX-exportable) and drops the training-only diagnostics. Concretizing seq_len /
    # batch_size for static export turns the q/k/p/pos reshapes into constant-target
    # Reshapes (no Shape/Gather/Concat), while the relative-shift indices are left
    # dynamic (see below) to avoid baking a multi-MB constant.
    query_head_dim = self.query_head_dim
    pos_head_dim = self.pos_head_dim
    num_heads = self.num_heads
    query_dim = query_head_dim * num_heads
    pos_dim = pos_head_dim * num_heads
    batch_size, seq_len = _sz(x, 0), _sz(x, 1)
    if projected or hasattr(self, 'onnx_in_proj_weight'):
        # The static export stores in_proj rows as per-head [q, k, p] blocks, so the
        # common head layout can be reshaped/transposed once and then split cheaply.
        if not projected:
            x = torch.nn.functional.linear(x, self.onnx_in_proj_weight, self.onnx_in_proj_bias)
        x = x.reshape(batch_size, -1, num_heads, 2 * query_head_dim + pos_head_dim).transpose(1, 2)
        q, k, p = x.split((query_head_dim, query_head_dim, pos_head_dim), dim=-1)
        k = k.transpose(2, 3)
    else:
        x = self.in_proj(x)
        q, k, p = x.split((query_dim, query_dim, pos_dim), dim=-1)
        q = q.reshape(batch_size, -1, num_heads, query_head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, num_heads, query_head_dim).permute(0, 2, 3, 1)
        p = p.reshape(batch_size, -1, num_heads, pos_head_dim).transpose(1, 2)
    attn_scores = torch.matmul(q, k)
    if hasattr(self, 'onnx_linear_pos'):
        # Already packed as (1, head, pos_dim, relative_position) in __init__.
        pos_emb = self.onnx_linear_pos
        seq_len2 = self.onnx_pos_len
    else:
        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1
        pos_emb = pos_emb.reshape(
            _sz(pos_emb, 0), seq_len2, num_heads, pos_head_dim).permute(0, 2, 3, 1)
    pos_scores = torch.matmul(p, pos_emb)
    # Relative -> absolute position shift via the "skew" trick: prepend one column, then
    # reinterpret each (seq_len, 2*seq_len-1) block as (2*seq_len, seq_len) and slice out
    # the needed part. This is mathematically identical to the original as_strided/gather
    # (out[..., i, j] = pos_scores[..., i, seq_len-1-i+j]) but uses only Slice/Concat/
    # Reshape, so the expensive GatherElements and its runtime index tensor (Range/Tile/
    # Add) are gone and nothing large is baked into the model file. The prepended column is
    # always discarded by the final slice (the output only references original columns
    # seq_len-1-i+j in [0, 2*seq_len-2]), so its value is irrelevant: we reuse a slice of
    # the first column rather than allocating a zero pad whose 'batch' (the spatial cross
    # axis, up to MAX_SIGNAL_LENGTH) would make it large.
    pos_scores = torch.cat((pos_scores[..., [0]], pos_scores), dim=-1)
    pos_scores = pos_scores.reshape(batch_size, num_heads, seq_len2 + 1, -1)
    pos_scores = pos_scores[:, :, 1:, :]
    pos_scores = pos_scores.reshape(batch_size, num_heads, -1, seq_len2)
    pos_scores = pos_scores[:, :, :, :seq_len]
    attn_scores = attn_scores + pos_scores
    # The ModelScope training helper manually expands max/sub/exp/sum/div to
    # reduce backward memory. Export is inference-only, so emit the standard
    # ONNX Softmax operator directly.
    return torch.softmax(attn_scores, dim=-1)


def _self_attention_forward(self, x, attn_weights):
    batch_size = _sz(x, 0)
    num_heads = _sz(attn_weights, 1)
    value_head_dim = self.in_proj.out_features // num_heads
    x = self.in_proj(x)
    x = x.reshape(batch_size, -1, num_heads, value_head_dim).transpose(1, 2)
    x = torch.matmul(attn_weights, x)
    x = x.transpose(1, 2).contiguous().reshape(
        batch_size, -1, num_heads * value_head_dim)
    return self.whiten(self.out_proj(x))


def _nonlin_attention_forward(self, x, attn_weights):
    x = self.in_proj(x)
    hidden_channels = self.hidden_channels

    # The projection is exactly three equal branches. One Split avoids three
    # independent Slice operators and their duplicated boundary constants.
    s, x_mid, y = x.split(hidden_channels, dim=-1)

    s = self.tanh(self.balancer(s))
    # identity1/2/3 are diagnostic-only scaling.Identity() no-ops -> dropped.
    x_mid = self.whiten1(x_mid) * s
    x_mid = torch.matmul(attn_weights, x_mid)
    x_mid = x_mid * y
    return self.whiten2(self.out_proj(x_mid))


def _convolution_module_forward(self, x, src_key_padding_mask=None, chunk_size=-1):
    x = self.in_proj(x)
    channels = self.in_proj.out_features // 2
    # Value and gate partition the complete projection, so emit one Split rather
    # than two repeated Slices with separate static controls.
    x_mid, gate = x.split(channels, dim=-1)
    gate = self.sigmoid(self.balancer1(gate))
    x_mid = self.activation2(self.activation1(x_mid) * gate)
    x_mid = x_mid.transpose(1, 2)

    if src_key_padding_mask is not None:
        x_mid = x_mid.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x_mid), 0.0)

    if (not torch.jit.is_scripting() and not torch.jit.is_tracing() and chunk_size >= 0):
        x_mid = self.depthwise_conv(x_mid, chunk_size=chunk_size)
    else:
        x_mid = self.depthwise_conv(x_mid)

    x_mid = self.whiten(self.balancer2(x_mid).transpose(1, 2))
    return self.out_proj(x_mid)


def apply_onnx_export_patches():
    """Override the ModelScope ZipEnhancer submodule forwards in place so the model
    builds an ONNX-export-friendly graph without any ./modeling_modified shims."""
    from modelscope.models.audio.ans.zipenhancer_layers import scaling, zipformer
    scaling.BiasNorm.forward = _biasnorm_forward
    scaling.ActivationDropoutAndLinear.forward = _activation_dropout_and_linear_forward
    zipformer.Zipformer2EncoderLayer.forward = _zipformer2_encoder_layer_forward
    zipformer.BypassModule.forward = _bypass_forward
    zipformer.SimpleDownsample.forward = _simple_downsample_forward
    zipformer.SimpleUpsample.forward = _simple_upsample_forward
    zipformer.RelPositionMultiheadAttentionWeights.forward = _rel_pos_mha_weights_forward
    zipformer.SelfAttention.forward = _self_attention_forward
    zipformer.NonlinAttention.forward = _nonlin_attention_forward
    zipformer.ConvolutionModule.forward = _convolution_module_forward

class ZipEnhancer(torch.nn.Module):
    def __init__(self, zip_enhancer, stft_model, istft_model, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0, use_rectangular_istft=False):
        super(ZipEnhancer, self).__init__()
        self.zip_enhancer = zip_enhancer
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.compress_factor = float(0.3)
        self.compress_factor_inv = float(1.0 / self.compress_factor)
        self.compress_factor_sqrt = float(self.compress_factor * 0.5)
        if "int" not in OUT_AUDIO_DTYPE.lower():
            self.register_buffer(
                'inv_int16',
                torch.tensor(
                    [INV_INT16],
                    dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32),
                persistent=False)
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.use_rectangular_istft = use_rectangular_istft
        if self.use_rectangular_istft:
            # atan2(0, 0) represents zero phase, i.e. the rectangular unit vector
            # (real=1, imag=0). Keep it broadcast-ready for the fused phase path.
            self.register_buffer(
                'zero_phase_unit',
                torch.tensor([1.0, 0.0], dtype=torch.float32).reshape(1, 2, 1, 1),
                persistent=False)
        self._prepare_pos_tables()
        self._prepare_invariant_export_buffers()
        if STATIC_SHAPE:
            self._prepare_static_export_buffers()

    def _prepare_pos_tables(self):
        # Keep positional tables in their native float32 precision. Quantizing this
        # immutable state before projecting it does not reduce the static ONNX model
        # (only the projected float32 slices survive) and needlessly changes outputs.
        # Cache the half-length and leading dimension once for the dynamic fallback.
        from modelscope.models.audio.ans.zipenhancer_layers.zipformer import CompactRelPositionalEncoding
        for module in self.zip_enhancer.modules():
            if isinstance(module, CompactRelPositionalEncoding):
                module.onnx_pe_half = module.pe.size(0) // 2
                module.pe = module.pe.unsqueeze(0)

    def _register_export_buffer(self, module, name, value):
        value = value.detach().clone()
        if hasattr(module, name):
            setattr(module, name, value)
        else:
            module.register_buffer(name, value, persistent=False)

    def _validate_causal_dense_block(self, block, name):
        if len(block.dense_block) != 4:
            raise ValueError(f"{name} must contain exactly four dense layers.")
        for index, layer in enumerate(block.dense_block):
            conv, norm, prelu = layer[1], layer[2], layer[3]
            expected_dilation = (1 << index, 1)
            if (
                conv.kernel_size != (2, 3)
                or conv.stride != (1, 1)
                or conv.padding != (0, 0)
                or conv.dilation != expected_dilation
                or conv.groups != 1
                or conv.bias is None
            ):
                raise ValueError(
                    f"{name} layer {index} is not the expected causal 2x3 convolution."
                )
            if (
                not norm.affine
                or norm.track_running_stats
                or norm.weight is None
                or norm.bias is None
                or norm.eps != 1e-5
                or prelu.weight.numel() != conv.out_channels
            ):
                raise ValueError(
                    f"{name} layer {index} normalization/PReLU is not export-compatible."
                )

    def _prepare_invariant_export_buffers(self):
        from modelscope.models.audio.ans.zipenhancer_layers.scaling import (
            ActivationDropoutAndLinear, BiasNorm)
        from modelscope.models.audio.ans.zipenhancer_layers.zipformer import SimpleDownsample

        for module in self.zip_enhancer.modules():
            if isinstance(module, BiasNorm):
                scale = module.log_scale.exp() * math.sqrt(module.num_channels)
                self._register_export_buffer(module, 'onnx_l2_scale', scale)
            elif isinstance(module, ActivationDropoutAndLinear):
                if module.bias is None or module.activation not in ('SwooshL', 'SwooshR'):
                    raise ValueError(
                        "ActivationDropoutAndLinear must use a biased SwooshL/SwooshR projection.")
                offset = 0.035 if module.activation == 'SwooshL' else 0.313261687
                folded_bias = (
                    module.bias.double()
                    - offset * module.weight.double().sum(dim=1)
                ).to(module.bias.dtype)
                self._register_export_buffer(module, 'onnx_bias', folded_bias)
            elif isinstance(module, SimpleDownsample):
                self._register_export_buffer(
                    module,
                    'onnx_downsample_weights',
                    module.bias.softmax(dim=0).reshape(1, 1, module.downsample, 1))

        encoder_block = self.zip_enhancer.dense_encoder.dense_block
        mask_block = self.zip_enhancer.mask_decoder.dense_block
        phase_block = self.zip_enhancer.phase_decoder.dense_block
        self._validate_causal_dense_block(encoder_block, "DenseEncoder")
        self._validate_causal_dense_block(mask_block, "MaskDecoder")
        self._validate_causal_dense_block(phase_block, "PhaseDecoder")
        mask_layers = mask_block.dense_block
        phase_layers = phase_block.dense_block
        if len(mask_layers) != len(phase_layers):
            raise ValueError("Mask and phase decoder dense blocks must have equal depth.")
        self.decoder_dense_depth = len(mask_layers)
        for index, (mask_layer, phase_layer) in enumerate(zip(mask_layers, phase_layers)):
            mask_conv, phase_conv = mask_layer[1], phase_layer[1]
            mask_norm, phase_norm = mask_layer[2], phase_layer[2]
            mask_prelu, phase_prelu = mask_layer[3], phase_layer[3]
            if (
                mask_conv.weight.shape != phase_conv.weight.shape
                or mask_conv.dilation != phase_conv.dilation
                or mask_conv.stride != phase_conv.stride
                or mask_norm.weight.shape != phase_norm.weight.shape
                or mask_norm.bias.shape != phase_norm.bias.shape
                or mask_prelu.weight.shape != phase_prelu.weight.shape
            ):
                raise ValueError("Mask and phase decoder dense layers are not group-fusible.")
            self._register_export_buffer(
                self, f'decoder_dense_conv_weight_{index}',
                torch.cat((mask_conv.weight, phase_conv.weight), dim=0))
            self._register_export_buffer(
                self, f'decoder_dense_conv_bias_{index}',
                torch.cat((mask_conv.bias, phase_conv.bias), dim=0))
            self._register_export_buffer(
                self, f'decoder_dense_norm_weight_{index}',
                torch.cat((mask_norm.weight, phase_norm.weight), dim=0))
            self._register_export_buffer(
                self, f'decoder_dense_norm_bias_{index}',
                torch.cat((mask_norm.bias, phase_norm.bias), dim=0))
            self._register_export_buffer(
                self, f'decoder_dense_prelu_weight_{index}',
                torch.cat((mask_prelu.weight, phase_prelu.weight), dim=0))

        mask_decoder = self.zip_enhancer.mask_decoder
        phase_decoder = self.zip_enhancer.phase_decoder
        mask_up = mask_decoder.mask_conv[0]
        phase_up = phase_decoder.phase_conv[0]
        if (
            mask_up.conv1.weight.shape != phase_up.conv1.weight.shape
            or mask_up.conv1.kernel_size != (1, 3)
            or mask_up.conv1.stride != phase_up.conv1.stride
            or mask_up.conv1.padding != phase_up.conv1.padding
            or mask_up.conv1.dilation != phase_up.conv1.dilation
            or mask_up.conv1.groups != 1
            or phase_up.conv1.groups != 1
            or mask_up.conv1.bias is None
            or phase_up.conv1.bias is None
            or mask_up.upscale_width_factor != phase_up.upscale_width_factor
        ):
            raise ValueError("Mask and phase decoder upsamplers are not group-fusible.")
        mask_up_norm = mask_decoder.mask_conv[1]
        phase_up_norm = phase_decoder.phase_conv[1]
        mask_up_prelu = mask_decoder.mask_conv[2]
        phase_up_prelu = phase_decoder.phase_conv[2]
        if (
            not mask_up_norm.affine
            or not phase_up_norm.affine
            or mask_up_norm.track_running_stats
            or phase_up_norm.track_running_stats
            or mask_up_norm.eps != 1e-5
            or phase_up_norm.eps != 1e-5
            or mask_up_norm.weight.shape != phase_up_norm.weight.shape
            or mask_up_norm.bias.shape != phase_up_norm.bias.shape
            or mask_up_prelu.weight.shape != phase_up_prelu.weight.shape
        ):
            raise ValueError("Mask and phase decoder upsampler norms are not group-fusible.")
        self.decoder_upscale_width_factor = mask_up.upscale_width_factor
        self._register_export_buffer(
            self, 'decoder_up_weight',
            torch.cat((mask_up.conv1.weight, phase_up.conv1.weight), dim=0))
        self._register_export_buffer(
            self, 'decoder_up_bias',
            torch.cat((mask_up.conv1.bias, phase_up.conv1.bias), dim=0))
        self._register_export_buffer(
            self, 'decoder_up_norm_weight',
            torch.cat((mask_up_norm.weight, phase_up_norm.weight), dim=0))
        self._register_export_buffer(
            self, 'decoder_up_norm_bias',
            torch.cat((mask_up_norm.bias, phase_up_norm.bias), dim=0))
        self._register_export_buffer(
            self, 'decoder_up_prelu_weight',
            torch.cat((mask_up_prelu.weight, phase_up_prelu.weight), dim=0))

        phase_r = phase_decoder.phase_conv_r
        phase_i = phase_decoder.phase_conv_i
        if (
            phase_r.weight.shape != phase_i.weight.shape
            or phase_r.kernel_size != (1, 2)
            or phase_r.stride != (1, 1)
            or phase_r.padding != (0, 0)
            or phase_r.dilation != (1, 1)
            or phase_r.groups != 1
            or phase_i.kernel_size != phase_r.kernel_size
            or phase_i.stride != phase_r.stride
            or phase_i.padding != phase_r.padding
            or phase_i.dilation != phase_r.dilation
            or phase_i.groups != phase_r.groups
            or phase_r.bias is None
            or phase_i.bias is None
        ):
            raise ValueError("Phase real/imaginary output heads are not fusible.")
        self._register_export_buffer(
            self, 'phase_output_weight',
            torch.cat((phase_r.weight, phase_i.weight), dim=0))
        self._register_export_buffer(
            self, 'phase_output_bias',
            torch.cat((phase_r.bias, phase_i.bias), dim=0))

        for encoder in self.zip_enhancer.TSConformer.encoders:
            dualpath = encoder.encoder if hasattr(encoder, 'encoder') else encoder
            if len(dualpath.bypass_layers) != 2 * len(dualpath.f_layers):
                raise ValueError("Unexpected dual-path outer-bypass topology.")
            for index, layer in enumerate(dualpath.f_layers):
                self._register_attention_layer_projection(
                    layer, dualpath.bypass_layers[2 * index])
            for index, layer in enumerate(dualpath.t_layers):
                self._register_attention_layer_projection(
                    layer, dualpath.bypass_layers[2 * index + 1])
            if hasattr(encoder, 'out_combiner'):
                self._register_export_buffer(
                    encoder.out_combiner,
                    'onnx_residual_scale',
                    (1.0 - encoder.out_combiner.bypass_scale.double()).to(
                        encoder.out_combiner.bypass_scale.dtype))

    def _conv2d_out_dim(self, size, kernel, stride, padding, dilation):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    def _register_attention_pos_table(self, attn_weights, encoder_pos, length):
        pos = self._pos_enc(encoder_pos, length)
        pos = torch.nn.functional.linear(pos, attn_weights.linear_pos.weight, None)
        pos = pos.reshape(
            pos.shape[0], pos.shape[1], attn_weights.num_heads,
            attn_weights.pos_head_dim).permute(0, 2, 3, 1).contiguous()
        attn_weights.onnx_pos_len = int(pos.shape[-1])
        self._register_export_buffer(attn_weights, 'onnx_linear_pos', pos)

    def _register_attention_in_proj(self, attn_weights):
        query_head_dim = attn_weights.query_head_dim
        pos_head_dim = attn_weights.pos_head_dim
        num_heads = attn_weights.num_heads
        query_dim = query_head_dim * num_heads
        pos_dim = pos_head_dim * num_heads
        q_w, k_w, p_w = attn_weights.in_proj.weight.split((query_dim, query_dim, pos_dim), dim=0)
        q_b, k_b, p_b = attn_weights.in_proj.bias.split((query_dim, query_dim, pos_dim), dim=0)

        in_features = attn_weights.in_proj.in_features
        q_w = q_w.reshape(num_heads, query_head_dim, in_features)
        k_w = k_w.reshape(num_heads, query_head_dim, in_features)
        p_w = p_w.reshape(num_heads, pos_head_dim, in_features)
        q_b = q_b.reshape(num_heads, query_head_dim)
        k_b = k_b.reshape(num_heads, query_head_dim)
        p_b = p_b.reshape(num_heads, pos_head_dim)

        weight = torch.cat((q_w, k_w, p_w), dim=1).reshape(-1, in_features)
        bias = torch.cat((q_b, k_b, p_b), dim=1).reshape(-1)
        self._register_export_buffer(attn_weights, 'onnx_in_proj_weight', weight)
        self._register_export_buffer(attn_weights, 'onnx_in_proj_bias', bias)

    def _register_attention_layer_projection(self, layer, outer_bypass):
        attn_weights = layer.self_attn_weights
        self._register_attention_in_proj(attn_weights)
        layer.onnx_attn_projection_size = attn_weights.onnx_in_proj_weight.shape[0]
        layer.onnx_ff1_projection_size = layer.feed_forward1.in_proj.weight.shape[0]
        self._register_export_buffer(
            layer,
            'onnx_attn_ff1_weight',
            torch.cat((
                attn_weights.onnx_in_proj_weight,
                layer.feed_forward1.in_proj.weight), dim=0))
        self._register_export_buffer(
            layer,
            'onnx_attn_ff1_bias',
            torch.cat((
                attn_weights.onnx_in_proj_bias,
                layer.feed_forward1.in_proj.bias), dim=0))

        from modelscope.models.audio.ans.zipenhancer_layers.scaling import BiasNorm
        if not isinstance(layer.norm, BiasNorm):
            raise ValueError("The final Zipformer normalization must be BiasNorm.")
        combined_scale = (
            layer.bypass.bypass_scale.double()
            * outer_bypass.bypass_scale.double()
        )
        l2_scale = (
            layer.norm.log_scale.double().exp()
            * math.sqrt(layer.norm.num_channels)
        )
        self._register_export_buffer(
            layer,
            'onnx_final_norm_scale',
            (combined_scale * l2_scale).to(layer.bypass.bypass_scale.dtype))
        self._register_export_buffer(
            layer,
            'onnx_final_residual_scale',
            (1.0 - combined_scale).to(layer.bypass.bypass_scale.dtype))

    def _register_dualpath_pos_tables(self, encoder, freq_len, time_len):
        for layer in encoder.f_layers:
            self._register_attention_pos_table(layer.self_attn_weights, encoder.encoder_pos, freq_len)
        for layer in encoder.t_layers:
            self._register_attention_pos_table(layer.self_attn_weights, encoder.encoder_pos, time_len)

    def _prepare_static_export_buffers(self):
        dense_stride = self.zip_enhancer.dense_encoder.dense_conv_2[0]
        freq_len = self._conv2d_out_dim(
            NFFT // 2 + 1,
            dense_stride.kernel_size[1],
            dense_stride.stride[1],
            dense_stride.padding[1],
            dense_stride.dilation[1])
        time_len = MAX_SIGNAL_LENGTH

        for encoder in self.zip_enhancer.TSConformer.encoders:
            if hasattr(encoder, 'encoder'):
                down_time_len = _ceil_div(time_len, encoder.t_downsample_factor)
                down_freq_len = _ceil_div(freq_len, encoder.f_downsample_factor)
                self._register_dualpath_pos_tables(encoder.encoder, down_freq_len, down_time_len)
            else:
                self._register_dualpath_pos_tables(encoder, freq_len, time_len)

    def _pos_enc(self, encoder_pos, length):
        # CompactRelPositionalEncoding depends only on the sequence length, and its pe
        # table is precomputed in __init__ for max_len=1000 and already
        # unsqueezed to (1, 2*max_len-1, embed_dim) with its half-length index cached (see
        # _quantize_pos_tables). Slice the needed rows directly with the known (static)
        # length, instead of materialising a permuted/
        # contiguous view of the feature map just to read its first dimension. The slice
        # folds to a constant for a static export.
        half = encoder_pos.onnx_pe_half
        return encoder_pos.pe[:, half - length + 1:half + length, :]

    def _dense_block(self, block, x):
        # DenseBlockV2 is causal along time. PyTorch Conv2d cannot encode its
        # asymmetric (top=dilation, bottom=0) padding, while ONNX Conv can. Use
        # an exact symmetric Conv + tail Slice in the temporary raw export; the
        # checked post-export rewrite folds only these eight patterns into native
        # asymmetric Conv pads in the separate deployment model.
        skip = x
        for layer in block.dense_block:
            conv = layer[1]
            dilation = conv.dilation[0]
            x = torch.nn.functional.conv2d(
                skip,
                conv.weight,
                conv.bias,
                stride=conv.stride,
                padding=(dilation, 1),
                dilation=conv.dilation,
                groups=conv.groups)
            x = x[:, :, :-dilation, :]
            x = layer[2](x)
            x = layer[3](x)
            skip = torch.cat((x, skip), dim=1)
        return x

    def _decoder_dense_pair(self, x, b, c, t, f):
        # MaskDecoder and PhaseDecoder apply the same DenseBlock topology to the
        # same encoder output with independent parameters. Pack those parameters
        # as two convolution groups, keep both streams together for all four
        # layers, and split only the final result.
        skip = torch.cat((x, x), dim=1)
        for index in range(self.decoder_dense_depth):
            dilation = 1 << index
            x = torch.nn.functional.conv2d(
                skip,
                getattr(self, f'decoder_dense_conv_weight_{index}'),
                getattr(self, f'decoder_dense_conv_bias_{index}'),
                padding=(dilation, 1),
                dilation=(dilation, 1),
                groups=2)
            x = x[:, :, :-dilation, :]
            x = torch.nn.functional.instance_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=getattr(self, f'decoder_dense_norm_weight_{index}'),
                bias=getattr(self, f'decoder_dense_norm_bias_{index}'),
                use_input_stats=True,
                momentum=0.1,
                eps=1e-5)
            x = torch.nn.functional.prelu(
                x, getattr(self, f'decoder_dense_prelu_weight_{index}'))
            if index + 1 < self.decoder_dense_depth:
                current = x.reshape(b, 2, c, t, f)
                history = skip.reshape(b, 2, (index + 1) * c, t, f)
                skip = torch.cat((current, history), dim=2).reshape(
                    b, 2 * (index + 2) * c, t, f)
        return x

    def _decoder_upsample_pair(self, x, b, c, t, f):
        upscale = self.decoder_upscale_width_factor
        x = torch.nn.functional.conv2d(
            x,
            self.decoder_up_weight,
            self.decoder_up_bias,
            padding=(0, 1),
            groups=2)
        x = x.reshape(b, 2, c, upscale, t, f)
        x = x.permute(0, 1, 2, 4, 5, 3).contiguous().reshape(
            b, 2 * c, t, f * upscale)
        x = torch.nn.functional.instance_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=self.decoder_up_norm_weight,
            bias=self.decoder_up_norm_bias,
            use_input_stats=True,
            momentum=0.1,
            eps=1e-5)
        x = torch.nn.functional.prelu(x, self.decoder_up_prelu_weight)
        return torch.split(x, c, dim=1)

    def _dualpath_encoder(self, e, x, b, c, t, f):
        # DualPathZipformer2Encoder: (b, c, t, f) -> (b, c, t, f). One frequency-path layer
        # then one time-path layer. Keep tensors batch-major (batch, seq, channel) throughout;
        # the dual-path batch is the spatial cross-axis times the folded-window batch.
        pe_f = None if STATIC_SHAPE else self._pos_enc(e.encoder_pos, f)
        pe_t = None if STATIC_SHAPE else self._pos_enc(e.encoder_pos, t)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(b * t, f, c)                            # (b*t, f, c)
        x = e.f_layers[0](x, pe_f, src_key_padding_mask=None)
        x = x.reshape(b, t, f, c).permute(0, 2, 1, 3).contiguous().reshape(b * f, t, c)         # (b*f, t, c)
        x = e.t_layers[0](x, pe_t, src_key_padding_mask=None)
        return x.reshape(b, f, t, c).permute(0, 3, 2, 1).contiguous()                           # (b, c, t, f)

    def _downsampled_encoder(self, e, x, b, c, t, f):
        # DualPathDownsampledZipformer2Encoder: downsample t & f, run the inner dual-path
        # encoder, upsample back and combine. All 3-D tensors stay batch-major.
        ie = e.encoder
        src_orig = x                                                                            # (b, c, t, f)
        x = e.downsample_t(x.permute(0, 3, 2, 1).contiguous().reshape(b * f, t, c))             # (b*f, dt, c)
        dt = _sz(x, 1)
        x = e.downsample_f(x.reshape(b, f, dt, c).permute(0, 2, 1, 3).contiguous().reshape(b * dt, f, c))   # (b*dt, df, c)
        df = _sz(x, 1)
        ipe_f = None if STATIC_SHAPE else self._pos_enc(ie.encoder_pos, df)
        ipe_t = None if STATIC_SHAPE else self._pos_enc(ie.encoder_pos, dt)
        x = ie.f_layers[0](x, ipe_f, src_key_padding_mask=None)                                 # (b*dt, df, c)
        x = x.reshape(b, dt, df, c).permute(0, 2, 1, 3).contiguous().reshape(b * df, dt, c)     # (b*df, dt, c)
        x = ie.t_layers[0](x, ipe_t, src_key_padding_mask=None)                                 # (b*df, dt, c)

        # The channel-wise out-combiner scale commutes with both nearest-neighbor
        # upsamplers. Apply it while the tensor is still downsampled, then add the
        # pre-scaled full-resolution residual only once at the end.
        x = x * e.out_combiner.bypass_scale
        x = e.upsample_f(x.reshape(b, df, dt, c).permute(0, 2, 1, 3).contiguous().reshape(b * dt, df, c))   # (b*dt, f_up, c)
        x = e.upsample_t(x[:, :f].reshape(b, dt, f, c).permute(0, 2, 1, 3).contiguous().reshape(b * f, dt, c)) # (b*f, t_up, c)
        x = x[:, :t].reshape(b, f, t, c).permute(0, 3, 2, 1).contiguous()                       # (b, c, t, f)
        return src_orig * e.out_combiner.onnx_residual_scale.reshape(1, c, 1, 1) + x

    def forward(self, audio):
        audio = audio.float()
        if "int" not in IN_AUDIO_DTYPE.lower():
            audio = audio * 32768.0      # F16/F32 inputs arrive in [-1, 1]; lift them to int16 amplitude so the per-window RMS renorm returns the output at int16 scale.
        # Resample the input from IN_SAMPLE_RATE to the model's internal MODEL_SAMPLE_RATE
        # (16 kHz). For a static export the target length is fixed (MODEL_AUDIO_LENGTH);
        # for a dynamic export the scale factor is used instead.
        if self.in_sample_rate != MODEL_SAMPLE_RATE:
            audio = torch.nn.functional.interpolate(
                audio,
                size=MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            # Input length is already a whole number of windows (tail padded OUTSIDE the model
            # in numpy), so fold (1, 1, num_window*W) -> (num_window, 1, W) and batch the whole
            # dual-path graph. The per-window RMS norm below then normalizes each window.
            audio = audio.reshape(-1, 1, self.fold_window)
        model_input_len = audio.shape[-1] if not STATIC_SHAPE else None
        norm_factor = torch.sqrt(torch.mean(audio * audio, dim=-1, keepdim=True) + 1e-6)
        audio /= norm_factor
        real_part, imag_part = self.stft_model(audio)
        # Match modelscope mag_pha_stft: mag = (real^2 + imag^2 + 1e-9)^(compress_factor/2); pha = atan2(imag, real + 1e-5).
        magnitude = torch.pow(real_part * real_part + imag_part * imag_part + 1e-9, self.compress_factor_sqrt)
        phase = torch.atan2(imag_part, real_part + 1e-5)
        # --- ZipEnhancer.forward inlined ---
        # ================= DenseEncoder =================
        de = self.zip_enhancer.dense_encoder
        # [B,F,T] magnitude/phase -> [B,2,T,F] feature map: a single stack+transpose
        # replaces 2x(unsqueeze+permute)+cat (fewer layout ops, identical result).
        x = torch.stack((magnitude, phase), dim=1).transpose(2, 3)
        x = de.dense_conv_1(x)          # Conv2d -> InstanceNorm2d -> PReLU
        x = self._dense_block(de.dense_block, x)  # DenseBlockV2 (depth=4)
        x = de.dense_conv_2(x)          # Conv2d -> InstanceNorm2d -> PReLU
        # ================= Zipformer2DualPathEncoder (4 encoders) =================
        # c (channels), t (frames) and f (sub-bands) stay constant across the 4 outer
        # encoders, so read them once and reuse instead of re-querying every block.
        c, t, f = _sz(x, 1), _sz(x, 2), _sz(x, 3)
        b = _sz(x, 0)
        encs = self.zip_enhancer.TSConformer.encoders
        x = self._dualpath_encoder(encs[0], x, b, c, t, f)
        x = self._downsampled_encoder(encs[1], x, b, c, t, f)
        x = self._downsampled_encoder(encs[2], x, b, c, t, f)
        x = self._dualpath_encoder(encs[3], x, b, c, t, f)
        decoder_features = self._decoder_dense_pair(x, b, c, t, f)
        mx, px = self._decoder_upsample_pair(decoder_features, b, c, t, f)
        # ================= MappingDecoder =================
        md = self.zip_enhancer.mask_decoder
        mx = md.mask_conv[3](mx)
        # ================= PhaseDecoder =================
        # Fuse the real & imag 1x2 output convs (they share px) into one 2-channel conv.
        # The old polar path did atan2(phase_i, phase_r) and then ISTFT recomputed cos/sin
        # from that angle. CUDA EP does not place Atan, so keep the phase vector in
        # rectangular form: cos/sin(atan2(i, r)) == (r, i) / hypot(r, i).
        phase_ri = torch.nn.functional.conv2d(
            px,
            self.phase_output_weight,
            self.phase_output_bias)
        if self.use_rectangular_istft:
            # Keep real/imag together as (B, 2, F, T): normalize and apply the
            # uncompressed magnitude once. Feed the packed tensor straight to
            # ISTFT so no Split -> Squeeze -> Concat round-trip is exported.
            magnitude = torch.pow(
                md.relu(mx), self.compress_factor_inv).transpose(2, 3)
            phase_ri = phase_ri.transpose(2, 3)
            phase_norm = torch.linalg.vector_norm(phase_ri, ord=2, dim=1, keepdim=True)
            has_phase = phase_norm > 0.0
            phase_ri = torch.where(has_phase, phase_ri, self.zero_phase_unit)
            phase_norm = torch.where(has_phase, phase_norm, 1.0)
            # Divide the one-channel gain before broadcasting it over real and
            # imaginary channels; this halves decoder-side Div traffic.
            phase_ri = phase_ri * (magnitude / phase_norm)
            phase_ri = phase_ri.reshape(b, 2 * (NFFT // 2 + 1), t)
            audio = self.istft_model.inverse_packed(phase_ri)
        else:
            magnitude = md.relu(mx).squeeze(1).transpose(1, 2)
            phase = torch.atan2(phase_ri[:, 1:2], phase_ri[:, 0:1]).squeeze(1).transpose(1, 2)
            audio = self.istft_model(torch.pow(magnitude, self.compress_factor_inv), phase)
        if not STATIC_SHAPE:
            audio = audio[..., :model_input_len]
        audio *= norm_factor
        if self.use_batch_fold:
            audio = audio.reshape(1, 1, -1)                             # stitch windows back
        # Resample the denoised audio from MODEL_SAMPLE_RATE to OUT_SAMPLE_RATE.
        if self.out_sample_rate != MODEL_SAMPLE_RATE:
            audio = torch.nn.functional.interpolate(
                audio,
                size=OUTPUT_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else MODEL_TO_OUTPUT_SCALE,
                mode='linear',
                align_corners=False
            )

        if "int" in OUT_AUDIO_DTYPE.lower():
            # Clip already maps +/-Inf to the required int16 endpoints. Handle
            # NaN explicitly, but avoid torch.nan_to_num's redundant IsInf,
            # comparison, boolean-cast and Where chains for this integer path.
            audio = torch.where(torch.isnan(audio), 0.0, audio)
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)

        audio = torch.nan_to_num(audio, nan=0.0, posinf=32767.0, neginf=-32768.0)  # The ZipEnhancer output wav not [-1.0, 1.0]
        
        audio *= self.inv_int16
        
        if "32" in OUT_AUDIO_DTYPE:
            return audio
        
        return audio.to(torch.float16)



def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_ZipEnhancer_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


def main():
    _validate_export_configuration()
    print('Export start ...')
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    export_dir.mkdir(parents=True, exist_ok=True)
    apply_onnx_export_patches()
    with TemporaryDirectory(prefix="zipenhancer_raw_") as temporary_dir:
        raw_model_path = Path(temporary_dir) / "ZipEnhancer.raw.onnx"
        with torch.inference_mode():
            custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect').eval()
            custom_istft = STFT_Process(model_type='istft_B' if USE_RECTANGULAR_ISTFT else 'istft_A', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect', static_norm=STATIC_SHAPE).eval()
            zip_enhancer = Model.from_pretrained(model_name_or_path=model_path, device='cpu').model.eval()
            zip_enhancer = ZipEnhancer(zip_enhancer, custom_stft, custom_istft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, use_batch_fold=USE_BATCH_FOLD, fold_window=FOLD_WINDOW_LENGTH, use_rectangular_istft=USE_RECTANGULAR_ISTFT)
            if "32" in IN_AUDIO_DTYPE:
                IN_TORCH_DTYPE = torch.float32
            elif "int" in IN_AUDIO_DTYPE.lower():
                IN_TORCH_DTYPE = torch.int16
            else:
                IN_TORCH_DTYPE = torch.float16
            audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
            torch.onnx.export(
                zip_enhancer,
                (audio,),
                str(raw_model_path),
                input_names=['noisy_audio'],
                output_names=['denoised_audio'],
                dynamic_axes={
                    'noisy_audio': {2: 'audio_len'},
                    'denoised_audio': {2: 'audio_len'}
                } if DYNAMIC_AXES else None,
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False
            )
            del zip_enhancer
            del audio
            del custom_stft
            del custom_istft
            gc.collect()
        model_metadata = build_audio_metadata_from_globals(
            globals(), producer=Path(__file__).name, model_name="ZipEnhancer", task="denoise", model_family="zipenhancer",
            max_dynamic_audio_seconds=2, normalize_audio_default=False, input_channels=1, output_channels=1,
            num_audio_inputs=1, feature_kind="stft_zipformer", center_pad=True, pad_mode="reflect", extra={"n_mels": N_MELS},
        )
        # Stamp the temporary source before surgery because the checked matcher
        # validates this metadata. The temporary directory is removed on exit.
        raw_metadata_path = stamp_export_metadata(raw_model_path, model_metadata, OPSET)
        rewrite_report = rewrite_asymmetric_causal_convs(raw_model_path, onnx_model_A)
        # Runtime metadata is loaded from a sidecar carrier. The final graph
        # already inherited identical embedded metadata during surgery.
        shutil.copy2(raw_metadata_path, onnx_model_Metadata)
        # Do not persist or report the temporary source path. Its checksum
        # remains in the audit report so the finalized graph can be identified.
        rewrite_report.pop("source_model", None)
        rewrite_report["source_model_retained"] = False
        rewrite_report["source_model_deleted_after_process"] = True
        Path(onnx_rewrite_report).write_text(
            json.dumps(rewrite_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"Metadata saved to: {onnx_model_Metadata}")
        print(f"Rewrite report saved to: {onnx_rewrite_report}")
        print('\nExport done!')
        _run_inference_demo()
    print("Temporary raw export deleted.")


if __name__ == '__main__':
    main()
