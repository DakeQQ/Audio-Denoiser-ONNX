import gc
import subprocess
import sys
from pathlib import Path

import torch
from modelscope.models.base import Model
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


# --- File paths -------------------------------------------------------------
model_path          = "/home/DakeQQ/Downloads/speech_zipenhancer_ans_multiloss_16k_base"  # The ZipEnhancer download path.
parent_path         = Path(__file__).resolve().parent                                    # The folder that contains this script.
onnx_model_A        = str(parent_path / "ZipEnhancer_ONNX" / "ZipEnhancer.onnx")        # The exported onnx model path.
onnx_model_Metadata = str(metadata_path_for_model(onnx_model_A))                         # The metadata carrier onnx model path.

# --- Export settings --------------------------------------------------------
DYNAMIC_AXES = False              # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET        = 18                 # The ONNX opset version to export.

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
    scale = self.onnx_scale if hasattr(self, 'onnx_scale') else self.log_scale.exp()
    return scale * (x / torch.sqrt((x - self.bias).pow(2).mean(dim=self.channel_dim, keepdim=True)))


def _activation_dropout_and_linear_forward(self, x):
    # SwooshL/SwooshR activation (inlined, dropout is a no-op in eval) -> linear.
    if self.activation == 'SwooshL':
        x = (1.0 + (x - 4.0).exp()).log() - 0.08 * x - 0.035
    else:
        x = (1.0 + (x - 1.0).exp()).log() - 0.08 * x - 0.313261687
    return torch.nn.functional.linear(x, self.weight, self.bias)


def _zipformer2_encoder_layer_forward(self, src, pos_emb, chunk_size=-1, attn_mask=None, src_key_padding_mask=None):
    src_orig = src
    attn_weights = self.self_attn_weights(src, pos_emb=pos_emb, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask)
    src = src + self.feed_forward1(src)
    src = src + self.nonlin_attention(src, attn_weights[:1])
    src = src + self.self_attn1(src, attn_weights)
    src = src + self.conv_module1(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)
    src = src + self.feed_forward2(src)
    src = self.bypass_mid(src_orig, src)
    src = src + self.self_attn2(src, attn_weights)
    src = src + self.conv_module2(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)
    src = src + self.feed_forward3(src)
    src = self.norm(src)
    src = self.bypass(src_orig, src)
    return src


def _bypass_forward(self, src_orig, src):
    return src_orig + (src - src_orig) * self.bypass_scale


def _simple_downsample_forward(self, src):
    seq_len, batch_size, in_channels = _sz(src, 0), _sz(src, 1), _sz(src, 2)
    ds = self.downsample
    d_seq_len = (seq_len + ds - 1) // ds
    pad_len = d_seq_len * ds - seq_len
    if STATIC_SHAPE and pad_len == 0:
        # Frame count divisible by the factor -> the tail pad is empty, so the
        # expand + concat is a static no-op and is dropped from the graph.
        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
    elif STATIC_SHAPE and pad_len == 1:
        # The ZipEnhancer static shapes all pad by at most one frame for the
        # downsampled encoders. The slice already has shape (1, B, C), so an
        # Expand subgraph would be a no-op.
        src = torch.cat((src, src[seq_len - 1:]), dim=0).reshape(d_seq_len, ds, batch_size, in_channels)
    else:
        src_extra = src[seq_len - 1:].expand(pad_len, batch_size, in_channels)
        src = torch.cat((src, src_extra), dim=0).reshape(d_seq_len, ds, batch_size, in_channels)
    return (src * self.bias.softmax(dim=0).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


def _simple_upsample_forward(self, src):
    seq_len, batch_size, num_channels = _sz(src, 0), _sz(src, 1), _sz(src, 2)
    if STATIC_SHAPE:
        src = src.unsqueeze(1)
        return torch.cat((src,) * self.upsample, dim=1).reshape(-1, batch_size, num_channels)
    return src.unsqueeze(1).expand(seq_len, self.upsample, batch_size, num_channels).reshape(-1, batch_size, num_channels)


def _rel_pos_mha_weights_forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
    # Export-only RelPositionMultiheadAttentionWeights.forward: keeps just the
    # torch.jit.is_tracing() relative-shift branch (the eager as_strided path is not
    # ONNX-exportable) and drops the training-only diagnostics. Concretizing seq_len /
    # batch_size for static export turns the q/k/p/pos reshapes into constant-target
    # Reshapes (no Shape/Gather/Concat), while the relative-shift indices are left
    # dynamic (see below) to avoid baking a multi-MB constant.
    from modelscope.models.audio.ans.zipenhancer_layers.scaling import softmax
    query_head_dim = self.query_head_dim
    pos_head_dim = self.pos_head_dim
    num_heads = self.num_heads
    query_dim = query_head_dim * num_heads
    pos_dim = pos_head_dim * num_heads
    seq_len, batch_size = _sz(x, 0), _sz(x, 1)
    if hasattr(self, 'onnx_in_proj_weight'):
        # The static export stores in_proj rows as per-head [q, k, p] blocks, so the
        # common head layout can be reshaped/transposed once and then split cheaply.
        x = torch.nn.functional.linear(x, self.onnx_in_proj_weight, self.onnx_in_proj_bias)
        x = x.reshape(-1, batch_size, num_heads, 2 * query_head_dim + pos_head_dim).transpose(0, 2)
        q, k, p = x.split((query_head_dim, query_head_dim, pos_head_dim), dim=-1)
        k = k.transpose(2, 3)
    else:
        x = self.in_proj(x)
        q, k, p = x.split((query_dim, query_dim, pos_dim), dim=-1)
        q = q.reshape(-1, batch_size, num_heads, query_head_dim).permute(2, 1, 0, 3)
        k = k.reshape(-1, batch_size, num_heads, query_head_dim).permute(2, 1, 3, 0)
        p = p.reshape(-1, batch_size, num_heads, pos_head_dim).permute(2, 1, 0, 3)
    attn_scores = torch.matmul(q, k)
    pos_emb = self.onnx_linear_pos if hasattr(self, 'onnx_linear_pos') else self.linear_pos(pos_emb)
    seq_len2 = _sz(pos_emb, 1) if STATIC_SHAPE else 2 * seq_len - 1
    pos_emb_batch = _sz(pos_emb, 0) if STATIC_SHAPE else -1
    pos_emb = pos_emb.reshape(pos_emb_batch, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
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
    pos_scores = pos_scores.reshape(num_heads, batch_size, seq_len2 + 1, -1)
    pos_scores = pos_scores[:, :, 1:, :]
    pos_scores = pos_scores.reshape(num_heads, batch_size, -1, seq_len2)
    pos_scores = pos_scores[:, :, :, :seq_len]
    attn_scores = attn_scores + pos_scores
    return softmax(attn_scores, dim=-1)


def _self_attention_forward(self, x, attn_weights):
    batch_size = _sz(x, 1)
    num_heads = _sz(attn_weights, 0)
    value_head_dim = self.in_proj.out_features // num_heads
    x = self.in_proj(x)
    x = x.reshape(-1, batch_size, num_heads, value_head_dim).permute(2, 1, 0, 3)
    x = torch.matmul(attn_weights, x)
    x = x.permute(2, 1, 0, 3).contiguous().reshape(-1, batch_size, num_heads * value_head_dim)
    return self.whiten(self.out_proj(x))


def _nonlin_attention_forward(self, x, attn_weights):
    x = self.in_proj(x)
    batch_size = _sz(x, 1)
    hidden_channels = self.hidden_channels
    num_heads = _sz(attn_weights, 0)
    head_dim = hidden_channels // num_heads

    s = x[..., :hidden_channels]
    x_mid = x[..., hidden_channels:2 * hidden_channels]
    y = x[..., 2 * hidden_channels:3 * hidden_channels]

    s = self.tanh(self.balancer(s))
    # identity1/2/3 are diagnostic-only scaling.Identity() no-ops -> dropped.
    x_mid = self.whiten1(x_mid) * s
    x_mid = x_mid.reshape(-1, batch_size, num_heads, head_dim).permute(2, 1, 0, 3)
    x_mid = torch.matmul(attn_weights, x_mid)
    x_mid = x_mid.permute(2, 1, 0, 3).reshape(-1, batch_size, hidden_channels)
    x_mid = x_mid * y
    return self.whiten2(self.out_proj(x_mid))


def _convolution_module_forward(self, x, src_key_padding_mask=None, chunk_size=-1):
    x = self.in_proj(x)
    channels = self.in_proj.out_features // 2
    x_mid = x[..., :channels]
    gate = x[..., channels:2 * channels]
    gate = self.sigmoid(self.balancer1(gate))
    x_mid = self.activation2(self.activation1(x_mid) * gate)
    x_mid = x_mid.permute(1, 2, 0)

    if src_key_padding_mask is not None:
        x_mid = x_mid.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x_mid), 0.0)

    if (not torch.jit.is_scripting() and not torch.jit.is_tracing() and chunk_size >= 0):
        x_mid = self.depthwise_conv(x_mid, chunk_size=chunk_size)
    else:
        x_mid = self.depthwise_conv(x_mid)

    x_mid = self.whiten(self.balancer2(x_mid).permute(2, 0, 1))
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
    def __init__(self, zip_enhancer, stft_model, istft_model, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(ZipEnhancer, self).__init__()
        self.zip_enhancer = zip_enhancer
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.compress_factor = float(0.3)
        self.compress_factor_inv = float(1.0 / self.compress_factor)
        self.compress_factor_sqrt = float(self.compress_factor * 0.5)
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self._quantize_pos_tables()
        if STATIC_SHAPE:
            self._prepare_static_export_buffers()

    def _quantize_pos_tables(self):
        # Store each CompactRelPositionalEncoding `pe` table in float16 to halve its
        # in-memory footprint (and, for a dynamic-axes export, the surviving initializer);
        # _pos_enc slices the needed rows and casts them back to float32 on use. Done before
        # the static buffers are built so the precomputed onnx_linear_pos tables and the
        # dynamic forward path both reflect the same fp16 positional encoding. The table's
        # half-length index (pe.size(0) // 2, the boundary between the negative/positive
        # relative-position halves) is cached on the module, and the table is pre-unsqueezed
        # to (1, 2*max_len-1, embed_dim) so _pos_enc neither recomputes the index nor adds a
        # leading batch dim on every call.
        from modelscope.models.audio.ans.zipenhancer_layers.zipformer import CompactRelPositionalEncoding
        for module in self.zip_enhancer.modules():
            if isinstance(module, CompactRelPositionalEncoding):
                module.onnx_pe_half = module.pe.size(0) // 2
                module.pe = module.pe.half().unsqueeze(0)

    def _register_export_buffer(self, module, name, value):
        value = value.detach().clone()
        if hasattr(module, name):
            setattr(module, name, value)
        else:
            module.register_buffer(name, value, persistent=False)

    def _conv2d_out_dim(self, size, kernel, stride, padding, dilation):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    def _register_attention_pos_table(self, attn_weights, encoder_pos, length):
        pos = self._pos_enc(encoder_pos, length)
        pos = torch.nn.functional.linear(pos, attn_weights.linear_pos.weight, None)
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

    def _register_dualpath_pos_tables(self, encoder, freq_len, time_len):
        for layer in encoder.f_layers:
            self._register_attention_in_proj(layer.self_attn_weights)
            self._register_attention_pos_table(layer.self_attn_weights, encoder.encoder_pos, freq_len)
        for layer in encoder.t_layers:
            self._register_attention_in_proj(layer.self_attn_weights)
            self._register_attention_pos_table(layer.self_attn_weights, encoder.encoder_pos, time_len)

    def _prepare_static_export_buffers(self):
        from modelscope.models.audio.ans.zipenhancer_layers.scaling import BiasNorm

        for module in self.zip_enhancer.modules():
            if isinstance(module, BiasNorm):
                self._register_export_buffer(module, 'onnx_scale', module.log_scale.exp())

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
        # table is precomputed in __init__ for max_len=1000, stored in float16 and already
        # unsqueezed to (1, 2*max_len-1, embed_dim) with its half-length index cached (see
        # _quantize_pos_tables). Slice the needed rows directly with the known (static)
        # length and cast them back to float32, instead of materialising a permuted/
        # contiguous view of the feature map just to read its first dimension. The slice
        # folds to a constant for a static export.
        half = encoder_pos.onnx_pe_half
        return encoder_pos.pe[:, half - length + 1:half + length, :].float()

    def _dualpath_encoder(self, e, x, b, c, t, f):
        # DualPathZipformer2Encoder: (b, c, t, f) -> (b, c, t, f). One frequency-path layer
        # then one time-path layer. The tensor is kept 3-D (seq, batch, channel) across the
        # bypass. The dual-path "batch" is the spatial cross-axis (t for the freq layer, f for
        # the time layer); under batch-fold the window batch b multiplies it (batch = b*t / b*f).
        # For b=1 these reshapes reduce to the original single-window view/permute (bit-identical).
        pe_f = self._pos_enc(e.encoder_pos, f)
        pe_t = self._pos_enc(e.encoder_pos, t)
        x = x.permute(3, 0, 2, 1).contiguous().reshape(f, b * t, c)                            # (f, b*t, c)
        x = e.bypass_layers[0](x, e.f_layers[0](x, pe_f, src_key_padding_mask=None))
        x = x.reshape(f, b, t, c).permute(2, 1, 0, 3).contiguous().reshape(t, b * f, c)         # (t, b*f, c)
        x = e.bypass_layers[1](x, e.t_layers[0](x, pe_t, src_key_padding_mask=None))
        return x.reshape(t, b, f, c).permute(1, 3, 0, 2).contiguous()                           # (b, c, t, f)

    def _downsampled_encoder(self, e, x, b, c, t, f):
        # DualPathDownsampledZipformer2Encoder: downsample t & f, run the inner dual-path
        # encoder, upsample back and combine. The window batch b rides along each spatial
        # cross-axis (b*f / b*dt / b*df / b*t). For b=1 every reshape reduces to the original
        # single-window view/permute (bit-identical).
        ie = e.encoder
        src_orig = x.permute(2, 3, 0, 1)                                                        # (t, f, b, c)
        x = e.downsample_t(x.permute(2, 0, 3, 1).contiguous().reshape(t, b * f, c))             # (dt, b*f, c)
        dt = _sz(x, 0)
        x = e.downsample_f(x.reshape(dt, b, f, c).permute(2, 1, 0, 3).contiguous().reshape(f, b * dt, c))   # (df, b*dt, c)
        df = _sz(x, 0)
        ipe_f = self._pos_enc(ie.encoder_pos, df)
        ipe_t = self._pos_enc(ie.encoder_pos, dt)
        x = ie.bypass_layers[0](x, ie.f_layers[0](x, ipe_f, src_key_padding_mask=None))         # (df, b*dt, c)
        x = x.reshape(df, b, dt, c).permute(2, 1, 0, 3).contiguous().reshape(dt, b * df, c)     # (dt, b*df, c)
        x = ie.bypass_layers[1](x, ie.t_layers[0](x, ipe_t, src_key_padding_mask=None))         # (dt, b*df, c)
        x = e.upsample_f(x.reshape(dt, b, df, c).permute(2, 1, 0, 3).contiguous().reshape(df, b * dt, c))   # (f_up, b*dt, c)
        x = e.upsample_t(x[:f].reshape(f, b, dt, c).permute(2, 1, 0, 3).contiguous().reshape(dt, b * f, c)) # (t_up, b*f, c)
        x = e.out_combiner(src_orig, x[:t].reshape(t, b, f, c).permute(0, 2, 1, 3))             # (t, f, b, c)
        return x.permute(2, 3, 0, 1).contiguous()                                               # (b, c, t, f)

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
        model_input_len = _sz(audio, -1)
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
        x = de.dense_block(x)           # DenseBlockV2 (depth=4)
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
        enc_out = x
        # ================= MappingDecoder =================
        md = self.zip_enhancer.mask_decoder
        mx = md.dense_block(enc_out)    # DenseBlockV2 (depth=4)
        # mask_conv: SubPixelConvTranspose2d (inlined, static reshape) -> InstanceNorm2d -> PReLU -> Conv2d
        sp = md.mask_conv[0]
        b_, c_, t_, f_ = _sz(mx, 0), _sz(mx, 1), _sz(mx, 2), _sz(mx, 3)
        mx = sp.conv1(mx)
        mx = mx.view(b_, c_, sp.upscale_width_factor, t_, f_).permute(0, 1, 3, 4, 2).contiguous().view(b_, c_, t_, f_ * sp.upscale_width_factor)
        mx = md.mask_conv[1](mx)
        mx = md.mask_conv[2](mx)
        mx = md.mask_conv[3](mx)
        magnitude = md.relu(mx).squeeze(1).transpose(1, 2)
        # ================= PhaseDecoder =================
        pd = self.zip_enhancer.phase_decoder
        px = pd.dense_block(enc_out)    # DenseBlockV2 (depth=4)
        # phase_conv: SubPixelConvTranspose2d (inlined, static reshape) -> InstanceNorm2d -> PReLU
        sp = pd.phase_conv[0]
        b_, c_, t_, f_ = _sz(px, 0), _sz(px, 1), _sz(px, 2), _sz(px, 3)
        px = sp.conv1(px)
        px = px.view(b_, c_, sp.upscale_width_factor, t_, f_).permute(0, 1, 3, 4, 2).contiguous().view(b_, c_, t_, f_ * sp.upscale_width_factor)
        px = pd.phase_conv[1](px)
        px = pd.phase_conv[2](px)
        # Fuse the real & imag 1x2 output convs (they share px) into one 2-channel conv,
        # then split: phase = atan2(imag, real). The concat of constant weights folds away.
        phase_ri = torch.nn.functional.conv2d(
            px,
            torch.cat((pd.phase_conv_r.weight, pd.phase_conv_i.weight), dim=0),
            torch.cat((pd.phase_conv_r.bias, pd.phase_conv_i.bias), dim=0))
        phase = torch.atan2(phase_ri[:, 1:2], phase_ri[:, 0:1]).squeeze(1).transpose(1, 2)
        audio = self.istft_model(torch.pow(magnitude, self.compress_factor_inv), phase)[..., :model_input_len]
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

        audio = torch.nan_to_num(audio, nan=0.0, posinf=32767.0, neginf=-32768.0)  # The ZipEnhancer output wav not [-1.0, 1.0] 

        if "int" in OUT_AUDIO_DTYPE.lower():
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        
        audio *= self.inv_int16
        
        if "32" in OUT_AUDIO_DTYPE:
            return audio
        
        return audio.to(torch.float16)



def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_ZipEnhancer_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
apply_onnx_export_patches()
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect').eval()
    custom_istft = STFT_Process(model_type='istft_A', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect').eval()
    zip_enhancer = Model.from_pretrained(model_name_or_path=model_path, device='cpu').model.eval()
    zip_enhancer = ZipEnhancer(zip_enhancer, custom_stft, custom_istft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, use_batch_fold=USE_BATCH_FOLD, fold_window=FOLD_WINDOW_LENGTH)
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
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
