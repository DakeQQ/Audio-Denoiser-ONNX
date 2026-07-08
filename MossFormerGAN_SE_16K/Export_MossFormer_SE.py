import argparse
import gc
import subprocess
import sys
import os
from pathlib import Path
from clearvoice.models.mossformer_gan_se.generator import MossFormerGAN_SE_16K as MossFormerGANModel
import torch
import torch.nn.functional as F
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


model_path           = "/home/DakeQQ/Downloads/MossFormerGAN_SE_16K"                           # The MossFormerGAN_SE_16K download folder
parent_path          = Path(__file__).resolve().parent                                      # The folder that contains this script.
onnx_model_A         = str(parent_path / "MossFormer_ONNX" / "MossFormerGAN_SE_16K.onnx")  # The exported onnx model path.
onnx_model_Metadata  = str(metadata_path_for_model(onnx_model_A))                           # The metadata carrier onnx model path.
SCRIPT_DIR           = Path(__file__).resolve().parent


DYNAMIC_AXES         = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET                = 18


MODEL_SAMPLE_RATE    = 16000               # MossFormerGAN_SE runs at 16kHz internally.
IN_SAMPLE_RATE       = 16000               # [8000, 16000, 22500, 24000, 44100, 48000]; input audio sample rate.
OUT_SAMPLE_RATE      = 16000               # [8000, 16000, 22500, 24000, 44100, 48000]; output audio sample rate.
INPUT_AUDIO_LENGTH   = 32000               # Maximum input audio length in IN_SAMPLE_RATE samples. Higher values yield better quality but time consume.
MODEL_AUDIO_LENGTH   = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
OUTPUT_AUDIO_LENGTH  = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE))


WINDOW_TYPE          = 'hamming'           # Type of window function used in the STFT
N_MELS               = 60                  # Number of Mel bands to generate in the Mel-spectrogram
NFFT                 = 400                 # Number of FFT components for the STFT process
WINDOW_LENGTH        = 400                 # Length of windowing, edit it carefully.
HOP_LENGTH           = 100                 # Number of samples between successive frames in the STFT


BATCH_WINDOW_SECONDS = 1.5                 # When the configured input length is >= this many seconds, fold into fixed-length windows and batch-process them together (each window runs the full SyncANet independently -> per-window attention).
USE_BATCH_FOLD       = True                # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
FOLD_WINDOW_LENGTH   = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length, rounded UP to a HOP multiple so the center-pad STFT -> ISTFT reconstructs W samples per window.
EXPORT_AUDIO_LENGTH  = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop.
MAX_SIGNAL_LENGTH    = 2048 if DYNAMIC_AXES else (((FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else MODEL_AUDIO_LENGTH) + HOP_LENGTH - 1) // HOP_LENGTH + 1)  # STFT frame count (per-window in fold mode). Sizes the precomputed rotary / diagonal-mask tables and the ISTFT trim.


IN_AUDIO_DTYPE       = 'INT16'             # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'INT16'             # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16            = float(1.0 / 32768.0)


INPUT_TO_MODEL_SCALE  = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)


def load_mossformergan_model(checkpoint_dir):
    model = MossFormerGANModel(argparse.Namespace(mode='inference', fft_len=NFFT)).model
    with open(os.path.join(checkpoint_dir, 'last_best_checkpoint'), 'r') as file:
        checkpoint_name = file.readline().strip()
    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), map_location=lambda storage, loc: storage)
    pretrained_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state = model.state_dict()
    for key in state.keys():
        if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
            state[key] = pretrained_model[key]
        elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
            state[key] = pretrained_model[key.replace('module.', '')]
        elif 'module.' + key in pretrained_model and state[key].shape == pretrained_model['module.' + key].shape:
            state[key] = pretrained_model['module.' + key]
    model.load_state_dict(state)
    return model


def _fold_ln_linear(ln, lin):
    # Fold ``LayerNorm(weight=g, bias=beta)`` followed by ``Linear(W, b)`` into an
    # affine-free LayerNorm + ``Linear(W * g, W @ beta + b)``. Pushing the scale/shift
    # into the projection lets every FFConvM that reads the same tensor share one
    # (affine-free) normalization. Done in float64 for fidelity, returned as float32.
    Wd = lin.weight.double()
    g = ln.weight.double()
    beta = ln.bias.double()
    bd = lin.bias.double() if lin.bias is not None else torch.zeros(Wd.shape[0], dtype=torch.float64)
    return (Wd * g.unsqueeze(0)).float(), (Wd @ beta + bd).float()


def _fold_norm4d_conv2d(norm, conv):
    # Fold ``LayerNormalization4D``'s per-channel affine into a following grouped Conv2d.
    # This is used for SyncANet's intra path, where each Fconv group sees one input channel.
    Wd = conv.weight.detach().double()
    groups = int(conv.groups)
    gamma = norm.gamma.detach().reshape(-1).double()
    beta = norm.beta.detach().reshape(-1).double()
    out_channels, in_per_group = Wd.shape[:2]
    out_per_group = out_channels // groups
    Wg = Wd.view(groups, out_per_group, in_per_group, *Wd.shape[2:])
    scale = gamma.view(groups, 1, in_per_group, 1, 1)
    shift = beta.view(groups, 1, in_per_group, 1, 1)
    bias = conv.bias.detach().double() if conv.bias is not None else torch.zeros(out_channels, dtype=torch.float64)
    bias = bias.view(groups, out_per_group) + (Wg * shift).sum(dim=(2, 3, 4))
    return (Wg * scale).reshape_as(Wd).float(), bias.reshape(-1).float()


def _fold_norm4d_unfold1d(norm, kernel_size):
    # Fold ``LayerNormalization4D``'s per-channel affine into the inter-path window extractor.
    # ``F.unfold`` on [Q, C, T, 1] with kernel (K, 1) returns channels ordered as
    # [c0_t0, c0_t1, ..., c1_t0, c1_t1, ...]. A grouped Conv1d with one input channel per
    # group and K sparse output filters per group produces the same layout while baking in
    # y * gamma + beta as the Conv weights / bias.
    gamma = norm.gamma.detach().reshape(-1).float()
    beta = norm.beta.detach().reshape(-1).float()
    channels = int(gamma.numel())
    weight = torch.zeros(channels * kernel_size, 1, kernel_size, dtype=torch.float32)
    bias = torch.empty(channels * kernel_size, dtype=torch.float32)
    for channel in range(channels):
        for offset in range(kernel_size):
            out_channel = channel * kernel_size + offset
            weight[out_channel, 0, offset] = gamma[channel]
            bias[out_channel] = beta[channel]
    return weight, bias


def _ffconvm_parts(ff):
    # FFConvM = mdl[LayerNorm, Linear, SiLU, ConvModule, Dropout]; the ConvModule's
    # depthwise Conv1d (k31, pad15, residual) is at mdl[3].sequential[1].conv.
    return ff.mdl[0], ff.mdl[1], ff.mdl[3].sequential[1].conv


def _mossformer_block(p, x0, Q, BT, rotary_cos, rotary_sin, pos_ids, b=1):
    # Inlined ``MossFormer`` (GatedFormer) forward + cal_attention, expanded to leaf ops.
    # ``b`` is the window-fold batch: x0 has leading dim ``b * BT`` and the cross-token (``_c``)
    # attention is reshaped to (b, BT, ...) so it attends only WITHIN each window's BT tokens
    # (no cross-window mixing). ``b == 1`` reproduces the original single-window graph exactly.
    # ``p`` carries the precomputed fused weights (see ``MOSSFORMER_SE._mossformer_params``):
    # ``to_hidden`` + ``to_qk`` are fused into ONE Linear + ONE grouped depthwise conv (then
    # split into the value/gate pair v, u and the query/key qk), and ``to_out`` has its
    # LayerNorm folded into its Linear. The linear-attention ``1 / Q`` scale is folded
    # into OffsetScale head 3 (lin_k) when Q is static. The triple attention keeps a single group
    # (``padding_to_multiple_of(n, n) == 0`` always) so ``group_size`` never participates.
    # The rotary embedding (first ``rot`` query/key channels, GPT-J interleaved, position along
    # the Q axis) is applied to all four offset-scaled heads at once.
    # ``x0`` is (BT, Q, C); ``Q`` / ``BT`` are static ints when known, else None (read from x0).
    C = p['C']
    hidden = p['hidden']
    qk_dim = p['qk']
    vdim = p['vdim']
    rot = p['rot']
    half = rot // 2
    if Q is None:
        Q = x0.shape[1]
    if BT is None:
        BT = x0.shape[0]
    # token shift
    x_shift, x_pass = x0.chunk(2, dim=-1)
    x_shift = F.pad(x_shift, (0, 0, 1, -1))
    normed_x = torch.cat((x_shift, x_pass), dim=-1)
    # fused to_hidden + to_qk: affine-free LayerNorm -> one Linear -> SiLU -> one grouped conv (+residual)
    base = F.layer_norm(normed_x, (C,), None, None, 1e-5)
    cwi = p['in_cw']
    huv = F.silu(F.linear(base, p['in_w'], p['in_b']))
    huv = huv + F.conv1d(huv.transpose(1, 2), cwi, None, 1, 15, 1, cwi.shape[0]).transpose(1, 2)
    v, u, qk = huv.split((vdim, hidden - vdim, qk_dim), dim=-1)
    # OffsetScale (4 heads) is a broadcast multiply-add; the rotary rotation is then applied to
    # every head at once (the cos/sin tables broadcast over the head axis), avoiding a 4x loop.
    scaled = qk.unsqueeze(-2) * p['gamma'] + p['beta']                 # (BT, Q, 4, qk_dim)
    if rotary_cos is not None:
        cos = rotary_cos[:Q].float().reshape(1, Q, 1, half)
        sin = rotary_sin[:Q].float().reshape(1, Q, 1, half)
    else:
        inv_freq = p['inv_freq']
        ang = torch.arange(Q, dtype=inv_freq.dtype, device=x0.device).unsqueeze(-1) * inv_freq
        cos = ang.cos().reshape(1, Q, 1, half)
        sin = ang.sin().reshape(1, Q, 1, half)
    tm = scaled[..., :rot]
    rest = scaled[..., rot:]
    even = tm[..., 0::2]
    odd = tm[..., 1::2]
    rotp = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(start_dim=-2)
    quad_q, lin_q, quad_k, lin_k = torch.cat((rotp, rest), dim=-1).unbind(dim=-2)
    # single-group triple attention
    qq = quad_q.unsqueeze(1)
    kk = quad_k.unsqueeze(1)
    lq = lin_q.unsqueeze(1)
    lk = lin_k.unsqueeze(1)
    vg = v.unsqueeze(1)
    ug = u.unsqueeze(1)
    qq_c = quad_q.reshape(b, BT, Q, qk_dim).transpose(2, 1)
    kk_c = quad_k.reshape(b, BT, Q, qk_dim).transpose(2, 1)
    vg_c = v.reshape(b, BT, Q, vdim).transpose(2, 1)
    ug_c = u.reshape(b, BT, Q, vdim).transpose(2, 1)
    sim = torch.matmul(qq, kk.transpose(-1, -2)) / Q
    sim_c = torch.matmul(qq_c, kk_c.transpose(-1, -2)) / BT
    attn = F.relu(sim).pow(2)
    attn_c = F.relu(sim_c).pow(2)
    # Zero the diagonal (self-time) of the cross-group attention. The identity mask is built from
    # an ``arange`` equality (not ``torch.eye``) to avoid the boolean EyeLike op, which the ORT
    # CPU provider does not implement. Position ids come from a precomputed buffer when static.
    if pos_ids is not None:
        idx = pos_ids[:BT]
    else:
        idx = torch.arange(BT, device=x0.device, dtype=torch.int32)
    eye = idx.unsqueeze(-1) == idx
    attn_c = attn_c.masked_fill(eye, 0.0)
    quad_v = torch.matmul(attn, vg) + torch.matmul(attn_c, vg_c).transpose(2, 1).reshape(-1, 1, Q, vdim)
    quad_u = torch.matmul(attn, ug) + torch.matmul(attn_c, ug_c).transpose(2, 1).reshape(-1, 1, Q, vdim)
    lin_kv = torch.matmul(lk.transpose(-1, -2), vg).squeeze(1)
    lin_ku = torch.matmul(lk.transpose(-1, -2), ug).squeeze(1)
    if not p['lin_k_scaled']:
        inv_q = torch.reciprocal(torch.as_tensor(Q, dtype=lin_kv.dtype, device=lin_kv.device))
        lin_kv = lin_kv * inv_q
        lin_ku = lin_ku * inv_q
    lin_v = torch.matmul(lq, lin_kv.unsqueeze(1))
    lin_u = torch.matmul(lq, lin_ku.unsqueeze(1))
    att_v = (quad_v + lin_v).reshape(-1, Q, vdim)
    att_u = (quad_u + lin_u).reshape(-1, Q, vdim)
    out = (att_u * v) * torch.sigmoid(att_v * u)
    # folded to_out: affine-free LayerNorm -> Linear -> SiLU -> grouped conv (+residual)
    base_o = F.layer_norm(out, (vdim,), None, None, 1e-5)
    cwo = p['out_cw']
    ho = F.silu(F.linear(base_o, p['out_w'], p['out_b']))
    ho = ho + F.conv1d(ho.transpose(1, 2), cwo, None, 1, 15, 1, cwo.shape[0]).transpose(1, 2)
    return x0 + ho


class MOSSFORMER_SE(torch.nn.Module):
    def __init__(self, mossformer_se, stft_model, istft_model, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(MOSSFORMER_SE, self).__init__()
        self.mossformer_se = mossformer_se
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.compress_factor = float(0.3)
        self.compress_factor_inv = float(0.5 / self.compress_factor) - 0.5
        self.compress_factor_sqrt = float(self.compress_factor * 0.5)
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate

        # ── Precompute rotary (cos/sin) tables and diagonal-mask position ids ──
        # The inlined MossFormer attention applies a rotary embedding along the
        # token axis and zeroes the diagonal of the cross-group attention. Both
        # are deterministic functions of the (static) sequence length, so for a
        # static export they are baked into registered buffers here instead of
        # being recomputed (arange / cos / sin / eye) inside every one of the 12
        # MossFormer calls. This removes 12 Cos + 12 Sin (plus Range / Mul) ops
        # from the exported graph. The tables are stored as compact float16
        # buffers (sliced + cast to float32 at use) and the mask uses int32 ids,
        # so the initializers are half the size of the float32 / int64 originals.
        #
        # For DYNAMIC_AXES the token (time-frame) axis is unbounded at runtime,
        # so a fixed-size table could silently truncate long inputs; in that case
        # the tables are left as None and the original runtime construction is
        # used (correct for any length).
        self.precompute_rotary = not DYNAMIC_AXES
        if self.precompute_rotary:
            blocks = mossformer_se.blocks
            inv_freq = blocks[0].intra_mossformer.rotary_pos_emb.freqs.detach()
            # All MossFormer instances share the same (non-learned) rotary freqs.
            for blk in blocks:
                for mf in (blk.intra_mossformer, blk.inter_mossformer):
                    assert torch.equal(mf.rotary_pos_emb.freqs.detach(), inv_freq), \
                        "Rotary freqs differ across MossFormer layers; cannot share one table."
            # Longest token axis over both paths: frequency sub-bands (intra rotary
            # axis / inter diagonal axis) and time frames (inter rotary axis /
            # intra diagonal axis), plus a small margin for the emb_ks padding.
            n_freqs = int(blocks[0].intra_mossformer.group_size)
            max_seq = max(n_freqs, MAX_SIGNAL_LENGTH) + 2
            pos = torch.arange(max_seq, dtype=inv_freq.dtype)
            ang = pos.unsqueeze(-1) * inv_freq                  # (max_seq, 16)
            # float16 storage halves the ONNX initializer footprint; the buffer
            # is an initializer, so the per-call Slice + Cast(float32) stay as
            # runtime ops (they are not constant-folded) and the compact float16
            # constant is what ships in the graph.
            self.register_buffer('rotary_cos', ang.cos().to(torch.float16))
            self.register_buffer('rotary_sin', ang.sin().to(torch.float16))
            # int32 (not int64) position ids for the diagonal-mask equality: the
            # values are all < MAX_SIGNAL_LENGTH so they fit int32, ONNX Equal
            # supports int32, and int32 halves the buffer size. int64 is reserved
            # for tensors that feed a Slice / Gather as indices (this one does not).
            self.register_buffer('pos_ids', torch.arange(max_seq, dtype=torch.int32))
        else:
            self.rotary_cos = None
            self.rotary_sin = None
            self.pos_ids = None

        # ── Precompute fused FFConvM / qkv weights ──
        # Every FFConvM is LayerNorm -> Linear -> SiLU -> ConvModule(depthwise conv k31 + residual).
        # The LayerNorm affine folds into the following Linear (``_fold_ln_linear``), so the
        # normalization becomes affine-free and is shared by every FFConvM that reads the same
        # tensor. Two FFConvMs consuming the same input are then fused into ONE Linear (rows
        # concatenated) and ONE depthwise conv (channels concatenated -- the groups stay
        # independent, so the fused conv is exact), and split afterwards:
        #   * intra_to_u + intra_to_v  and  inter_to_u + inter_to_v   (per block)
        #   * to_hidden + to_qk        (per MossFormer; to_hidden's half is later split into v, u)
        # ``to_out`` keeps its own input but still folds its LayerNorm into its Linear.
        # The fused tensors are registered as buffers (so they ship as ONNX initializers) and
        # referenced through ``self.blk_params`` in the forward pass.
        blocks = mossformer_se.blocks
        mf0 = blocks[0].intra_mossformer
        self.in_channels = int(blocks[0].intra_to_u.mdl[1].weight.shape[1])   # emb_dim * emb_ks
        self.n_freqs = int(mf0.group_size)                                    # static frequency axis
        # The time-frame count depends only on the input length, so it is a static int for the
        # default (16 kHz, non-dynamic) export and None otherwise (then read from the tensor).
        self.frames_static = MAX_SIGNAL_LENGTH if not DYNAMIC_AXES else None
        self.n_features = int(NFFT // 2 + 1)
        self.decoder_channels = int(mossformer_se.mask_decoder.sub_pixel.conv.weight.shape[0] // mossformer_se.mask_decoder.sub_pixel.r)
        self.subpixel_width = self.n_freqs * int(mossformer_se.mask_decoder.sub_pixel.r)
        mf_C = int(mf0.to_qk.mdl[1].weight.shape[1])                          # 64
        mf_hidden = int(mf0.to_hidden.mdl[1].weight.shape[0])                 # 256
        mf_qk = int(mf0.to_qk.mdl[1].weight.shape[0])                         # 128
        mf_rot = int(2 * mf0.rotary_pos_emb.freqs.shape[0])                   # 32
        self._buf_id = 0
        self.blk_params = []
        for blk in blocks:
            p = {}
            p['intra_fconv_w'], p['intra_fconv_b'] = self._fold_norm4d_fconv(blk.intra_norm, blk.Fconv)
            p['inter_unfold_w'], p['inter_unfold_b'] = self._fold_norm4d_unfold(blk.inter_norm, blk.emb_ks)
            p['intra_uv_w'], p['intra_uv_b'], p['intra_uv_cw'] = self._fuse_pair(blk.intra_to_u, blk.intra_to_v)
            p['inter_uv_w'], p['inter_uv_b'], p['inter_uv_cw'] = self._fuse_pair(blk.inter_to_u, blk.inter_to_v)
            p['intra_mf'] = self._mossformer_params(blk.intra_mossformer, mf_C, mf_hidden, mf_qk, mf_hidden // 2, mf_rot, self.n_freqs)
            p['inter_mf'] = self._mossformer_params(blk.inter_mossformer, mf_C, mf_hidden, mf_qk, mf_hidden // 2, mf_rot, self.frames_static)
            p['attn'] = self._attention_params(blk)
            self.blk_params.append(p)

    def _rb(self, t):
        # Register ``t`` as a uniquely named buffer and return the live reference, so the fused
        # weight ships as an ONNX initializer while the forward pass can index it by name.
        name = "_pb%d" % self._buf_id
        self._buf_id += 1
        self.register_buffer(name, t.contiguous())
        return getattr(self, name)

    def _fold_norm4d_fconv(self, norm, conv):
        w, b = _fold_norm4d_conv2d(norm, conv)
        return self._rb(w), self._rb(b)

    def _fold_norm4d_unfold(self, norm, kernel_size):
        w, b = _fold_norm4d_unfold1d(norm, int(kernel_size))
        return self._rb(w), self._rb(b)

    def _fuse_pair(self, ff0, ff1):
        # Fuse two FFConvMs that read the same input: affine-free LayerNorm (shared) + one Linear
        # (folded LayerNorm affines, rows concatenated) + one depthwise conv (channels concatenated).
        ln0, lin0, cv0 = _ffconvm_parts(ff0)
        ln1, lin1, cv1 = _ffconvm_parts(ff1)
        w0, b0 = _fold_ln_linear(ln0, lin0)
        w1, b1 = _fold_ln_linear(ln1, lin1)
        return (self._rb(torch.cat([w0, w1], 0)),
                self._rb(torch.cat([b0, b1], 0)),
                self._rb(torch.cat([cv0.weight, cv1.weight], 0)))

    def _mossformer_params(self, mf, C, hidden, qk, vdim, rot, q_len):
        # Precompute the fused weights for one MossFormer: to_hidden + to_qk fused into one
        # Linear/conv, to_out's LayerNorm folded into its Linear, plus the (static) dims and
        # the rotary / offset-scale tensors. If the token axis is static, fold the linear
        # attention ``1 / Q`` scale into OffsetScale head 3 (lin_k): rotary is linear, so
        # scaling gamma and beta before the rotation is equivalent to scaling lin_k after it.
        in_w, in_b, in_cw = self._fuse_pair(mf.to_hidden, mf.to_qk)
        ln_o, lin_o, cv_o = _ffconvm_parts(mf.to_out)
        out_w, out_b = _fold_ln_linear(ln_o, lin_o)
        gamma = mf.qk_offset_scale.gamma.detach().clone()
        beta = mf.qk_offset_scale.beta.detach().clone()
        lin_k_scaled = q_len is not None
        if lin_k_scaled:
            inv_q = 1.0 / float(q_len)
            gamma[3].mul_(inv_q)
            beta[3].mul_(inv_q)
        return {
            'C': C, 'hidden': hidden, 'qk': qk, 'vdim': vdim, 'rot': rot,
            'in_w': in_w, 'in_b': in_b, 'in_cw': in_cw,
            'out_w': self._rb(out_w), 'out_b': self._rb(out_b), 'out_cw': self._rb(cv_o.weight.clone()),
            'gamma': self._rb(gamma), 'beta': self._rb(beta), 'lin_k_scaled': lin_k_scaled,
            'inv_freq': mf.rotary_pos_emb.freqs,
        }

    def _attention_params(self, blk):
        # Fuse the per-head triple-attention Q/K/V 1x1 projections into one conv/PReLU. The
        # LayerNormalization4DCF statistics still stay per logical head by reshaping to
        # (Q-or-K, head, channel, time, freq) before the reductions. The attention
        # 1/sqrt(D) scale is folded into the Q/K norm affines as D**-0.25 on both sides.
        heads = int(blk.n_head)
        q_mods = [getattr(blk, "attn_conv_Q_%d" % jj) for jj in range(heads)]
        k_mods = [getattr(blk, "attn_conv_K_%d" % jj) for jj in range(heads)]
        v_mods = [getattr(blk, "attn_conv_V_%d" % jj) for jj in range(heads)]
        q_ch = int(q_mods[0][0].weight.shape[0])
        v_ch = int(v_mods[0][0].weight.shape[0])
        qk_flat = q_ch * self.n_freqs
        qk_input_scale = float(qk_flat ** -0.25)
        conv_w = torch.cat([m[0].weight.detach() for m in (q_mods + k_mods + v_mods)], dim=0)
        conv_b = torch.cat([m[0].bias.detach() for m in (q_mods + k_mods + v_mods)], dim=0)
        prelu_w = torch.cat([m[1].weight.detach().expand(m[0].weight.shape[0]) for m in (q_mods + k_mods + v_mods)], dim=0)
        q_gamma = torch.stack([m[2].gamma.detach().squeeze(0).squeeze(1) for m in q_mods], dim=0) * qk_input_scale
        k_gamma = torch.stack([m[2].gamma.detach().squeeze(0).squeeze(1) for m in k_mods], dim=0) * qk_input_scale
        q_beta = torch.stack([m[2].beta.detach().squeeze(0).squeeze(1) for m in q_mods], dim=0) * qk_input_scale
        k_beta = torch.stack([m[2].beta.detach().squeeze(0).squeeze(1) for m in k_mods], dim=0) * qk_input_scale
        v_gamma = torch.stack([m[2].gamma.detach().squeeze(0).squeeze(1) for m in v_mods], dim=0)
        v_beta = torch.stack([m[2].beta.detach().squeeze(0).squeeze(1) for m in v_mods], dim=0)
        return {
            'w': self._rb(conv_w), 'b': self._rb(conv_b), 'prelu': self._rb(prelu_w),
            'qk_gamma': self._rb(torch.stack([q_gamma, k_gamma], dim=0).unsqueeze(0).unsqueeze(4)),
            'qk_beta': self._rb(torch.stack([q_beta, k_beta], dim=0).unsqueeze(0).unsqueeze(4)),
            'v_gamma': self._rb(v_gamma.unsqueeze(0).unsqueeze(3)),
            'v_beta': self._rb(v_beta.unsqueeze(0).unsqueeze(3)),
            'heads': heads, 'q_ch': q_ch, 'v_ch': v_ch,
            'qk_total': 2 * heads * q_ch,
            'qk_flat': qk_flat, 'v_flat': v_ch * self.n_freqs,
            'eps': float(q_mods[0][2].eps),
        }

    def forward(self, audio):
        # ``self.mossformer_se`` is the trained ``SyncANet`` network. Every sub-module
        # ``forward`` it used to call is expanded below into its raw functional ops
        # (F.conv2d / F.linear / F.instance_norm / ... with the original .weight/.bias)
        # so the full ONNX export graph is visible in one place. The structural loops
        # (6 SyncANet blocks, 4 dilated-dense layers, 4 attention heads) are kept, and
        # the custom STFT / ISTFT remain plain calls.
        M = self.mossformer_se
        audio = audio.float()
        if "int" not in IN_AUDIO_DTYPE.lower():
            audio = audio * 32768.0      # F16/F32 inputs arrive in [-1, 1]; lift them to int16 amplitude so the per-window RMS renorm returns the output at int16 scale.
        if self.in_sample_rate != MODEL_SAMPLE_RATE:
            audio = torch.nn.functional.interpolate(
                audio,
                size=MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        model_input_len = audio.shape[-1]
        if self.use_batch_fold:
            # Input length is already a whole number of windows (padded OUTSIDE the model in
            # numpy), so fold (1, 1, num_window*W) -> (num_window, 1, W). norm_factor (dim=-1
            # keepdim) is already per-window, and the whole SyncANet then runs batched.
            audio = audio.reshape(-1, 1, self.fold_window)
            model_input_len = self.fold_window
        norm_factor = torch.sqrt(torch.mean(audio * audio, dim=-1, keepdim=True) + 1e-6)
        audio /= norm_factor
        padding_len = (HOP_LENGTH - model_input_len % HOP_LENGTH) % HOP_LENGTH
        audio = torch.cat([audio, audio[..., :padding_len]], dim=-1)
        real_part, imag_part = self.stft_model(audio)
        power = real_part * real_part + imag_part * imag_part
        magnitude_compress = torch.pow(power, self.compress_factor_sqrt)
        safe_power = power.clamp_min(torch.finfo(power.dtype).tiny)
        phase_scale = torch.pow(safe_power, self.compress_factor_sqrt - 0.5)
        real_compress = real_part * phase_scale
        imag_compress = imag_part * phase_scale
        # Stack the magnitude / compressed-real / compressed-imag channels straight into the
        # (1, 3, F, T) layout (one op instead of cat-along-batch + unsqueeze), then transpose to
        # (1, 3, T, F) for the encoder.
        x = torch.stack([magnitude_compress, real_compress, imag_compress], dim=1).transpose(-1, -2)

        # ============================================================
        # DenseEncoder (inlined)
        # ============================================================
        enc = M.dense_encoder
        # conv_1: Conv2d(3 -> 64, 1x1) -> InstanceNorm2d -> PReLU
        x = F.conv2d(x, enc.conv_1[0].weight, enc.conv_1[0].bias)
        x = F.instance_norm(x, None, None, enc.conv_1[1].weight, enc.conv_1[1].bias, True, 0.1, 1e-5)
        x = F.prelu(x, enc.conv_1[2].weight)
        # DilatedDenseNet (depth 4) -> dense connectivity; each layer ends with a UniDeepFsmn
        dd = enc.dilated_dense
        skip = x
        for i in range(dd.depth):
            conv = getattr(dd, "conv%d" % (i + 1))
            norm = getattr(dd, "norm%d" % (i + 1))
            prelu = getattr(dd, "prelu%d" % (i + 1))
            uf = getattr(dd, "fsmn%d" % (i + 1)).fsmn
            dil = 2 ** i
            out = F.pad(skip, (1, 1, dil, 0))
            out = F.conv2d(out, conv.weight, conv.bias, dilation=(dil, 1))
            out = F.instance_norm(out, None, None, norm.weight, norm.bias, True, 0.1, 1e-5)
            out = F.prelu(out, prelu.weight)
            # FSMN_Wrap -> UniDeepFsmn (lorder 5)
            fin = out.permute(0, 2, 3, 1)
            f1 = F.relu(F.linear(fin, uf.linear.weight, uf.linear.bias))
            p1 = F.linear(f1, uf.project.weight)
            x_per = p1.permute(1, 3, 2, 0)
            y = F.pad(x_per, [0, 0, uf.lorder - 1, uf.lorder - 1])
            o = x_per + F.conv2d(y, uf.conv1.weight, groups=uf.conv1.weight.shape[0])
            out = (fin + o.permute(3, 0, 2, 1)).permute(0, 3, 1, 2)
            skip = torch.cat([out, skip], dim=1)
        x = out
        # conv_2: Conv2d(64 -> 64, 1x3, stride (1,2), pad (0,1)) -> InstanceNorm2d -> PReLU
        x = F.conv2d(x, enc.conv_2[0].weight, enc.conv_2[0].bias, stride=(1, 2), padding=(0, 1))
        x = F.instance_norm(x, None, None, enc.conv_2[1].weight, enc.conv_2[1].bias, True, 0.1, 1e-5)
        x = F.prelu(x, enc.conv_2[2].weight)

        # ============================================================
        # SyncANet blocks (inlined). The loop over the 6 blocks is kept.
        # ============================================================
        bsz = x.shape[0]                                 # window-fold batch (num_window; 1 otherwise)
        for ii in range(M.n_layers):
            b = M.blocks[ii]
            pb = self.blk_params[ii]

            # ------------------------- intra path -------------------------
            # intra_norm: LayerNormalization4D (statistics over the channel dim)
            mu = x.mean(dim=1, keepdim=True)
            sd = torch.sqrt(x.var(dim=1, unbiased=False, keepdim=True) + b.intra_norm.eps)
            t = (x - mu) / sd
            # Fconv with intra_norm gamma / beta folded into the grouped Conv2d weights / bias.
            t = F.conv2d(t, pb['intra_fconv_w'], pb['intra_fconv_b'], groups=b.emb_dim)
            # (B, C, T, F) -> (B*T, F, C): fold the window batch into the time-frame batch so the
            # frequency-axis attention runs per (window, time-frame). For B=1 this equals the
            # original squeeze(0).permute(1, 2, 0).
            t = t.permute(0, 2, 3, 1).contiguous()
            t = t.reshape(-1, t.shape[-2], t.shape[-1])
            # intra_to_u + intra_to_v (fused FFConvM pair): one affine-free LayerNorm, one Linear,
            # one grouped depthwise conv (+residual), then split into the u / v branches.
            huv = F.layer_norm(t, (self.in_channels,), None, None, 1e-5)
            cw = pb['intra_uv_cw']
            huv = F.silu(F.linear(huv, pb['intra_uv_w'], pb['intra_uv_b']))
            huv = huv + F.conv1d(huv.transpose(1, 2), cw, None, 1, 15, 1, cw.shape[0]).transpose(1, 2)
            iu, iv = huv.chunk(2, dim=-1)
            # intra_rnn: UniDeepFsmn (lorder 20) on the gated "u" branch
            uf = b.intra_rnn[0]
            fz = iu.unsqueeze(0)
            f1 = F.relu(F.linear(fz, uf.linear.weight, uf.linear.bias))
            p1 = F.linear(f1, uf.project.weight)
            x_per = p1.permute(1, 3, 2, 0)
            y = F.pad(x_per, [0, 0, uf.lorder - 1, uf.lorder - 1])
            o = x_per + F.conv2d(y, uf.conv1.weight, groups=uf.conv1.weight.shape[0])
            iu = (fz + o.permute(3, 0, 2, 1)).squeeze(0)
            t = iv * iu
            t = t.transpose(1, 2)
            # intra_linear: ConvTranspose1d
            t = F.conv_transpose1d(t, b.intra_linear.weight, b.intra_linear.bias, stride=b.emb_hs)
            t = t.transpose(1, 2)
            # intra_mossformer: inlined fused MossFormer path (rotary along the frequency axis).
            t = _mossformer_block(pb['intra_mf'], t, self.n_freqs, self.frames_static, self.rotary_cos, self.rotary_sin, self.pos_ids, bsz)
            # (B*T, F, C) -> (B, C, T, F)
            t = t.reshape(bsz, -1, self.n_freqs, t.shape[-1]).permute(0, 3, 1, 2).contiguous()
            # intra_se: SELayer (avg + max channel attention)
            se = b.intra_se
            sa = F.adaptive_avg_pool2d(t, 1).reshape(bsz, -1)
            sa = torch.sigmoid(F.linear(F.relu(F.linear(sa, se.avg_pool_layer[0].weight, se.avg_pool_layer[0].bias)), se.avg_pool_layer[2].weight, se.avg_pool_layer[2].bias))
            sm = F.adaptive_max_pool2d(t, 1).reshape(bsz, -1)
            sm = torch.sigmoid(F.linear(F.relu(F.linear(sm, se.max_pool_layer[0].weight, se.max_pool_layer[0].bias)), se.max_pool_layer[2].weight, se.max_pool_layer[2].bias))
            t = (sa + sm).reshape(bsz, -1, 1, 1) * t
            t = t + x

            # ------------------------- inter path -------------------------
            inp = t
            mu = inp.mean(dim=1, keepdim=True)
            sd = torch.sqrt(inp.var(dim=1, unbiased=False, keepdim=True) + b.inter_norm.eps)
            t = (inp - mu) / sd
            # (B, C, T, F) -> (B*F, C, T): fold the window batch into the frequency batch so the
            # time-axis attention runs per (window, freq). For B=1 this equals squeeze(0).permute(2, 0, 1).
            t = t.permute(0, 3, 1, 2).reshape(-1, inp.shape[1], inp.shape[2]).contiguous()
            # Grouped Conv1d is the inter-path unfold, with inter_norm gamma / beta folded in.
            t = F.conv1d(t, pb['inter_unfold_w'], pb['inter_unfold_b'], stride=b.emb_hs, groups=b.emb_dim)
            t = t.transpose(1, 2)
            # inter_to_u + inter_to_v (fused FFConvM pair): one affine-free LayerNorm, one Linear,
            # one grouped depthwise conv (+residual), then split into the u / v branches.
            huv = F.layer_norm(t, (self.in_channels,), None, None, 1e-5)
            cw = pb['inter_uv_cw']
            huv = F.silu(F.linear(huv, pb['inter_uv_w'], pb['inter_uv_b']))
            huv = huv + F.conv1d(huv.transpose(1, 2), cw, None, 1, 15, 1, cw.shape[0]).transpose(1, 2)
            iu, iv = huv.chunk(2, dim=-1)
            # inter_rnn: UniDeepFsmn (lorder 20)
            uf = b.inter_rnn[0]
            fz = iu.unsqueeze(0)
            f1 = F.relu(F.linear(fz, uf.linear.weight, uf.linear.bias))
            p1 = F.linear(f1, uf.project.weight)
            x_per = p1.permute(1, 3, 2, 0)
            y = F.pad(x_per, [0, 0, uf.lorder - 1, uf.lorder - 1])
            o = x_per + F.conv2d(y, uf.conv1.weight, groups=uf.conv1.weight.shape[0])
            iu = (fz + o.permute(3, 0, 2, 1)).squeeze(0)
            t = iv * iu
            t = t.transpose(1, 2)
            t = F.conv_transpose1d(t, b.inter_linear.weight, b.inter_linear.bias, stride=b.emb_hs)
            t = t.transpose(1, 2)
            # inter_mossformer: inlined fused MossFormer path (rotary along the time axis).
            t = _mossformer_block(pb['inter_mf'], t, self.frames_static, self.n_freqs, self.rotary_cos, self.rotary_sin, self.pos_ids, bsz)
            # (B*F, T, C) -> (B, C, F, T) (SELayer runs in this layout, transposed to (B,C,T,F) after)
            t = t.reshape(bsz, self.n_freqs, -1, t.shape[-1]).permute(0, 3, 1, 2).contiguous()
            # inter_se: SELayer
            se = b.inter_se
            sa = F.adaptive_avg_pool2d(t, 1).reshape(bsz, -1)
            sa = torch.sigmoid(F.linear(F.relu(F.linear(sa, se.avg_pool_layer[0].weight, se.avg_pool_layer[0].bias)), se.avg_pool_layer[2].weight, se.avg_pool_layer[2].bias))
            sm = F.adaptive_max_pool2d(t, 1).reshape(bsz, -1)
            sm = torch.sigmoid(F.linear(F.relu(F.linear(sm, se.max_pool_layer[0].weight, se.max_pool_layer[0].bias)), se.max_pool_layer[2].weight, se.max_pool_layer[2].bias))
            t = (sa + sm).reshape(bsz, -1, 1, 1) * t
            inter = t.transpose(-1, 2) + inp

            # ------------------------- triple attention -------------------------
            ap = pb['attn']
            old_T = self.frames_static if self.frames_static is not None else inter.shape[2]
            qkv = F.prelu(F.conv2d(inter, ap['w'], ap['b']), ap['prelu'])
            qk = qkv[:, :ap['qk_total']].view(bsz, 2, ap['heads'], ap['q_ch'], old_T, self.n_freqs)
            qk_mu = qk.mean(dim=(3, 5), keepdim=True)
            qk_sd = torch.sqrt(qk.var(dim=(3, 5), unbiased=False, keepdim=True) + ap['eps'])
            qk = ((qk - qk_mu) / qk_sd) * ap['qk_gamma'] + ap['qk_beta']
            vv = qkv[:, ap['qk_total']:].view(bsz, ap['heads'], ap['v_ch'], old_T, self.n_freqs)
            v_mu = vv.mean(dim=(2, 4), keepdim=True)
            v_sd = torch.sqrt(vv.var(dim=(2, 4), unbiased=False, keepdim=True) + ap['eps'])
            vv = ((vv - v_mu) / v_sd) * ap['v_gamma'] + ap['v_beta']
            # Keep the window batch (B) as an outer batch: attention is per (window, head).
            Q = qk[:, 0].permute(0, 1, 3, 2, 4).reshape(bsz, ap['heads'], old_T, ap['qk_flat'])
            K = qk[:, 1].permute(0, 1, 3, 2, 4).reshape(bsz, ap['heads'], old_T, ap['qk_flat'])
            V = vv.permute(0, 1, 3, 2, 4).reshape(bsz, ap['heads'], old_T, ap['v_flat'])
            attn_mat = F.softmax(torch.matmul(Q, K.transpose(-1, -2)), dim=-1)
            V = torch.matmul(attn_mat, V).reshape(bsz, ap['heads'], old_T, ap['v_ch'], self.n_freqs).permute(0, 1, 3, 2, 4)
            V = V.reshape(bsz, ap['heads'] * ap['v_ch'], old_T, self.n_freqs)
            pm = b.attn_concat_proj
            V = F.prelu(F.conv2d(V, pm[0].weight, pm[0].bias), pm[1].weight)
            mu = V.mean(dim=(1, 3), keepdim=True)
            sd = torch.sqrt(V.var(dim=(1, 3), unbiased=False, keepdim=True) + pm[2].eps)
            V = ((V - mu) / sd) * pm[2].gamma + pm[2].beta
            x = V + inter

        # ============================================================
        # MaskDecoder (inlined) -> mask
        # ============================================================
        md = M.mask_decoder
        dd = md.dense_block
        skip = x
        for i in range(dd.depth):
            conv = getattr(dd, "conv%d" % (i + 1))
            norm = getattr(dd, "norm%d" % (i + 1))
            prelu = getattr(dd, "prelu%d" % (i + 1))
            uf = getattr(dd, "fsmn%d" % (i + 1)).fsmn
            dil = 2 ** i
            out = F.pad(skip, (1, 1, dil, 0))
            out = F.conv2d(out, conv.weight, conv.bias, dilation=(dil, 1))
            out = F.instance_norm(out, None, None, norm.weight, norm.bias, True, 0.1, 1e-5)
            out = F.prelu(out, prelu.weight)
            fin = out.permute(0, 2, 3, 1)
            f1 = F.relu(F.linear(fin, uf.linear.weight, uf.linear.bias))
            p1 = F.linear(f1, uf.project.weight)
            x_per = p1.permute(1, 3, 2, 0)
            y = F.pad(x_per, [0, 0, uf.lorder - 1, uf.lorder - 1])
            o = x_per + F.conv2d(y, uf.conv1.weight, groups=uf.conv1.weight.shape[0])
            out = (fin + o.permute(3, 0, 2, 1)).permute(0, 3, 1, 2)
            skip = torch.cat([out, skip], dim=1)
        xm = out
        # sub_pixel: SPConvTranspose2d
        sp = md.sub_pixel
        xm = F.pad(xm, (1, 1, 0, 0))
        xm = F.conv2d(xm, sp.conv.weight, sp.conv.bias)
        decoder_frames = self.frames_static if self.frames_static is not None else xm.shape[2]
        xm = xm.view((bsz, sp.r, self.decoder_channels, decoder_frames, self.n_freqs)).permute(0, 2, 3, 4, 1).contiguous().view((bsz, self.decoder_channels, decoder_frames, self.subpixel_width))
        # conv_1 -> InstanceNorm2d -> PReLU
        xm = F.conv2d(xm, md.conv_1.weight, md.conv_1.bias)
        xm = F.instance_norm(xm, None, None, md.norm.weight, md.norm.bias, True, 0.1, 1e-5)
        xm = F.prelu(xm, md.prelu.weight)
        # final_conv -> rearrange -> prelu_out
        xm = F.conv2d(xm, md.final_conv.weight, md.final_conv.bias).permute(0, 3, 2, 1).squeeze(-1)
        mask = F.prelu(xm, md.prelu_out.weight).permute(0, 2, 1).transpose(1, 2)

        # ============================================================
        # ComplexDecoder (inlined) -> complex_out
        # ============================================================
        cd = M.complex_decoder
        dd = cd.dense_block
        skip = x
        for i in range(dd.depth):
            conv = getattr(dd, "conv%d" % (i + 1))
            norm = getattr(dd, "norm%d" % (i + 1))
            prelu = getattr(dd, "prelu%d" % (i + 1))
            uf = getattr(dd, "fsmn%d" % (i + 1)).fsmn
            dil = 2 ** i
            out = F.pad(skip, (1, 1, dil, 0))
            out = F.conv2d(out, conv.weight, conv.bias, dilation=(dil, 1))
            out = F.instance_norm(out, None, None, norm.weight, norm.bias, True, 0.1, 1e-5)
            out = F.prelu(out, prelu.weight)
            fin = out.permute(0, 2, 3, 1)
            f1 = F.relu(F.linear(fin, uf.linear.weight, uf.linear.bias))
            p1 = F.linear(f1, uf.project.weight)
            x_per = p1.permute(1, 3, 2, 0)
            y = F.pad(x_per, [0, 0, uf.lorder - 1, uf.lorder - 1])
            o = x_per + F.conv2d(y, uf.conv1.weight, groups=uf.conv1.weight.shape[0])
            out = (fin + o.permute(3, 0, 2, 1)).permute(0, 3, 1, 2)
            skip = torch.cat([out, skip], dim=1)
        xc = out
        sp = cd.sub_pixel
        xc = F.pad(xc, (1, 1, 0, 0))
        xc = F.conv2d(xc, sp.conv.weight, sp.conv.bias)
        decoder_frames = self.frames_static if self.frames_static is not None else xc.shape[2]
        xc = xc.view((bsz, sp.r, self.decoder_channels, decoder_frames, self.n_freqs)).permute(0, 2, 3, 4, 1).contiguous().view((bsz, self.decoder_channels, decoder_frames, self.subpixel_width))
        # InstanceNorm2d -> PReLU -> Conv2d(64 -> 2)
        xc = F.instance_norm(xc, None, None, cd.norm.weight, cd.norm.bias, True, 0.1, 1e-5)
        xc = F.prelu(xc, cd.prelu.weight)
        xc = F.conv2d(xc, cd.conv.weight, cd.conv.bias)
        complex_out = xc.transpose(-1, -2)

        mag_real = mask * real_compress
        mag_imag = mask * imag_compress
        complex_real, complex_imag = complex_out.split(1, dim=1)
        final_real = mag_real + complex_real.squeeze(1)
        final_imag = mag_imag + complex_imag.squeeze(1)
        factor = torch.pow(final_real * final_real + final_imag * final_imag, self.compress_factor_inv)
        audio = self.istft_model(final_real * factor, final_imag * factor)[..., :model_input_len]
        audio *= norm_factor
        if self.use_batch_fold:
            audio = audio.reshape(1, 1, -1)                             # stitch windows back
        if self.out_sample_rate != MODEL_SAMPLE_RATE:
            audio = torch.nn.functional.interpolate(
                audio,
                size=OUTPUT_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else MODEL_TO_OUTPUT_SCALE,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        audio = audio * self.inv_int16
        if "32" in OUT_AUDIO_DTYPE:
            return audio
        return audio.to(torch.float16)




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_MossFormer_SE_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    # The original ClearVoice pipeline (clearvoice/utils/misc.py -> torch.stft(..., center=True))
    # uses the torch default pad_mode='reflect'. The custom STFT must mirror that, otherwise the
    # first/last ~NFFT//HOP frames are computed from zero padding instead of a reflected signal,
    # producing large edge errors in the spectrogram fed to the network.
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect').eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode='reflect').eval()

    mossformer = load_mossformergan_model(model_path).eval().float().to("cpu")
    mossformer = MOSSFORMER_SE(mossformer, custom_stft, custom_istft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH)
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)

    torch.onnx.export(
        mossformer,
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
    del mossformer
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="MossFormerGAN_SE_16K", task="denoise", model_family="mossformergan_se",
    max_dynamic_audio_seconds=6, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=1, feature_kind="stft_power_compress", center_pad=True, pad_mode="reflect", extra={"n_mels": N_MELS},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
