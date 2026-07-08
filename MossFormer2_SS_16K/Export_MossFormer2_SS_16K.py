import gc
import subprocess
import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata

parent_path             = Path(__file__).resolve().parent                                    # The folder that contains this script.
model_path              = "/home/DakeQQ/Downloads/MossFormer2_SS_16K"                         # The MossFormer2_SS_16K download folder.
onnx_model_A            = str(parent_path / "MossFormer_ONNX" / "MossFormer2_SS_16K.onnx")  # The exported onnx model path.
onnx_model_Metadata     = str(metadata_path_for_model(onnx_model_A))                         # The metadata carrier onnx model path.


DYNAMIC_AXES            = False        # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET                   = 18
MODEL_SAMPLE_RATE       = 16000        # MossFormer2_SS_16K runs at 16kHz internally.
IN_SAMPLE_RATE          = 16000        # [8000, 16000, 22500, 24000, 44000, 48000]; input audio sample rate.
OUT_SAMPLE_RATE         = 16000        # [8000, 16000, 22500, 24000, 44000, 48000]; output audio sample rate.
INPUT_AUDIO_LENGTH      = 32000        # Maximum input audio length in IN_SAMPLE_RATE samples. Higher values yield better quality but time consume.
PAD_HEAD                = 8000         # ~0.5 Seconds
IN_AUDIO_DTYPE          = 'INT16'      # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE         = 'INT16'      # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16               = float(1.0 / 32768.0)


MODEL_AUDIO_LENGTH      = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
OUTPUT_AUDIO_LENGTH     = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
INPUT_TO_MODEL_SCALE    = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE   = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE   = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
BATCH_WINDOW_SECONDS    = 1.5          # When the configured input length is >= this many seconds, fold into fixed-length windows and batch-process them together (each window runs the full network independently, i.e. per-window attention + per-window RMS normalization).
FOLD_WINDOW_LENGTH      = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + 8 - 1) // 8) * 8  # Per-window model-rate length, rounded UP to the encoder stride (8) so (W - enc_kernel16)//8 frames reconstruct exactly via the ConvTranspose decoder.
USE_BATCH_FOLD          = True         # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
EXPORT_AUDIO_LENGTH     = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop.

# Standalone model construction: import the pristine MossFormer2_SS_16K network
# directly from the installed clearvoice package (no site-package patching).
from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K


def load_mossformer2_ss_model(checkpoint_dir):
    """Builds the MossFormer2_SS_16K network and loads its checkpoint.

    This inlines ClearVoice's model construction and ``SpeechModel._load_model``
    checkpoint-loading logic so the export script does not depend on the
    ``modeling_modified`` shim files or the ClearVoice framework.
    """
    args = argparse.Namespace(
        num_spks=2,                    # Number of speakers to separate.
        encoder_kernel_size=16,        # Kernel size for the Conv1D encoder.
        encoder_embedding_dim=512,     # Embedding dimension from the encoder.
        mossformer_sequence_dim=512,   # Sequence dimension for the MossFormer blocks.
        num_mossformer_layer=24,       # Number of MossFormer layers.
    )
    model = MossFormer2_SS_16K(args).model
    best_name = os.path.join(checkpoint_dir, 'last_best_checkpoint')
    with open(best_name, 'r') as f:
        checkpoint_name = f.readline().strip()
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
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

class MOSSFORMER_SS(torch.nn.Module):
    def __init__(self, mossformer_ss, input_audio_len, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(MOSSFORMER_SS, self).__init__()
        self.mossformer_ss = mossformer_ss
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.num_spks = mossformer_ss.num_spks
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.norm_factor = float(10.0 ** (-25.0 / 20.0))
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding

        model_audio_len = fold_window if use_batch_fold else (MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else int(round(input_audio_len * MODEL_SAMPLE_RATE / in_sample_rate)))
        enc_kernel = mossformer_ss.enc.conv1d.kernel_size[0]
        enc_stride = mossformer_ss.enc.conv1d.stride[0]
        self.static_frames = (model_audio_len - enc_kernel) // enc_stride + 1

        mask_net = mossformer_ss.mask_net
        gfsmn = mask_net.mdl.intra_mdl.mossformerM
        flash_layers = gfsmn.layers
        fsmn_blocks = gfsmn.fsmn
        flash0 = flash_layers[0]

        # Safe upper bound for the precomputed positional / rotary tables. The encoder
        # produces n = (model_len - kernel) // (kernel // 2) + 1 frames, always below this.
        max_seq = max(int(2000.0 * input_audio_len / in_sample_rate) + 16, self.static_frames + 16)

        # ---- ScaledSinuEmbedding positional table (kept fp32: a 24-layer squared-ReLU
        # stack amplifies fp16 rounding, and fp16 saves nothing under static export) ----
        t = torch.arange(max_seq, dtype=torch.float32)
        sinu = t.unsqueeze(-1) * mask_net.pos_enc.inv_freq
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        emb_pos = (emb * mask_net.pos_enc.scale).transpose(0, -1).unsqueeze(0)
        self.register_buffer('emb_pos', emb_pos.contiguous())

        # ---- FLASH layer geometry (read once from the live modules) ----
        group_size = flash0.group_size
        qk_dim = flash0.qk_offset_scale.gamma.shape[-1]                 # 128
        vu_dim = flash0.to_hidden.mdl[1].weight.shape[0] // 2           # 1024
        model_dim = flash0.to_hidden.mdl[1].weight.shape[1]            # 512
        dw_kernel = flash0.to_hidden.mdl[3].sequential[1].conv.weight.shape[-1]   # 17
        self.model_dim = model_dim
        self.flash_group_size = group_size
        self.fl_vu = vu_dim
        self.fl_vu2 = vu_dim * 2
        self.fl_qk = qk_dim
        self.fl_in_groups = self.fl_vu2 + qk_dim                       # 2176
        self.dw_pad = (dw_kernel - 1) // 2                            # 8
        self.fl_inv_g = float(1.0 / group_size)
        self.static_padding = (group_size - (self.static_frames % group_size)) % group_size
        self.static_padded_len = self.static_frames + self.static_padding
        self.static_num_groups = self.static_padded_len // group_size
        self.static_inv_n = float(1.0 / self.static_frames)
        self.fold_lin_inv_n = not DYNAMIC_AXES

        # Shared ScaleNorm scalars for to_hidden || to_qk (identical dim / eps)
        self.fl_scale = float(flash0.to_hidden.mdl[0].scale)
        self.fl_eps = float(flash0.to_hidden.mdl[0].eps)
        self.fl_norm_eps = float(self.fl_eps / self.fl_scale)
        self.fl_in_scale_fold = float(1.0 / self.fl_scale)
        self.fl_out_scale = float(flash0.to_out.mdl[0].scale)
        self.fl_out_eps = float(flash0.to_out.mdl[0].eps)
        self.fl_out_norm_eps = float(self.fl_out_eps / self.fl_out_scale)
        self.fl_out_scale_fold = float(1.0 / self.fl_out_scale)

        # ---- Rotary cos/sin tables, shared by every FLASH layer (depend only on the
        # frame position). Precomputed once and sliced per call so the Range / Mul /
        # Stack / Cos / Sin become constant initializers. Stored fp32 (see note above). ----
        rot_freqs = flash0.rotary_pos_emb.freqs
        self.rot_dim = int(2 * rot_freqs.shape[0])                     # literal rotated channels (32)
        rot_ang = torch.arange(max_seq, dtype=rot_freqs.dtype).unsqueeze(-1) * rot_freqs
        rot_ang = torch.stack((rot_ang, rot_ang), dim=-1).flatten(-2)  # interleave-double -> [max_seq, rot_dim]
        self.register_buffer('rot_cos', rot_ang.cos().unsqueeze(0).unsqueeze(2).contiguous())  # [1, max_seq, 1, rot_dim]
        self.register_buffer('rot_sin', rot_ang.sin().unsqueeze(0).unsqueeze(2).contiguous())

        # ---- Zero-pad templates for the concat-based group padding (faster than F.pad) ----
        self.register_buffer('shift_pad', torch.zeros((1, 1, model_dim // 2), dtype=torch.float32))
        self.register_buffer('pad_A4', torch.zeros((1, group_size, 4, qk_dim), dtype=torch.float32))
        self.register_buffer('pad_VU', torch.zeros((1, group_size, vu_dim * 2), dtype=torch.float32))

        # ---- Per-layer fused FLASH projection weights (built in float64, cast to float32).
        # to_hidden (-> v, u) and to_qk (-> q, k) share one ScaleNorm denominator, so both
        # projections collapse into ONE Linear + ONE depthwise Conv with the scalar ScaleNorm
        # gains folded into the weights. The to_out gain is folded too. ----
        self._fl_in_w, self._fl_in_b, self._fl_in_c = [], [], []
        self._fl_out_w, self._fl_out_b, self._fl_out_c = [], [], []
        self._qkos_gamma, self._qkos_beta = [], []
        for i, fl in enumerate(flash_layers):
            gh = fl.to_hidden.mdl[0].g.detach().double()
            gqk = fl.to_qk.mdl[0].g.detach().double()
            w_in = torch.cat((fl.to_hidden.mdl[1].weight.detach().double() * gh * self.fl_in_scale_fold,
                              fl.to_qk.mdl[1].weight.detach().double() * gqk * self.fl_in_scale_fold), dim=0).float().contiguous()
            b_in = torch.cat((fl.to_hidden.mdl[1].bias.detach(),
                              fl.to_qk.mdl[1].bias.detach()), dim=0).float().contiguous()
            c_in = torch.cat((fl.to_hidden.mdl[3].sequential[1].conv.weight.detach(),
                              fl.to_qk.mdl[3].sequential[1].conv.weight.detach()), dim=0).contiguous()
            w_out = (fl.to_out.mdl[1].weight.detach().double() * fl.to_out.mdl[0].g.detach().double() * self.fl_out_scale_fold).float().contiguous()
            qk_scale = torch.ones((4, 1), dtype=torch.float64)
            qk_scale[0, 0] = self.fl_inv_g                            # fold 1/group_size into quad_q
            if self.fold_lin_inv_n:
                qk_scale[3, 0] = self.static_inv_n                     # fold 1/n into lin_k for static export
            qkos_gamma = (fl.qk_offset_scale.gamma.detach().double() * qk_scale).float().contiguous()
            qkos_beta = (fl.qk_offset_scale.beta.detach().double() * qk_scale).float().contiguous()
            self.register_buffer(f'fl_in_w_{i}', w_in)
            self.register_buffer(f'fl_in_b_{i}', b_in)
            self.register_buffer(f'fl_in_c_{i}', c_in)
            self.register_buffer(f'fl_out_w_{i}', w_out)
            self.register_buffer(f'fl_out_b_{i}', fl.to_out.mdl[1].bias.detach().float().contiguous())
            self.register_buffer(f'fl_out_c_{i}', fl.to_out.mdl[3].sequential[1].conv.weight.detach().contiguous())
            self.register_buffer(f'qkos_gamma_{i}', qkos_gamma)
            self.register_buffer(f'qkos_beta_{i}', qkos_beta)
            self._fl_in_w.append(getattr(self, f'fl_in_w_{i}'))
            self._fl_in_b.append(getattr(self, f'fl_in_b_{i}'))
            self._fl_in_c.append(getattr(self, f'fl_in_c_{i}'))
            self._fl_out_w.append(getattr(self, f'fl_out_w_{i}'))
            self._fl_out_b.append(getattr(self, f'fl_out_b_{i}'))
            self._fl_out_c.append(getattr(self, f'fl_out_c_{i}'))
            self._qkos_gamma.append(getattr(self, f'qkos_gamma_{i}'))
            self._qkos_beta.append(getattr(self, f'qkos_beta_{i}'))

        # ---- Per-layer fused FSMN gate weights: to_u and to_v share one (affine-free)
        # LayerNorm; the LayerNorm affine is folded into each Linear so both run as one
        # fused Linear + depthwise Conv. The dilated-dense memory stays a leaf module call. ----
        gf0 = fsmn_blocks[0].gated_fsmn
        self.fs_inner = gf0.to_u.mdl[1].weight.shape[0]               # 256
        self.fs_uv_groups = self.fs_inner * 2                         # 512
        self.fs_ln_shape = tuple(gf0.to_u.mdl[0].normalized_shape)
        self.fs_ln_eps = float(gf0.to_u.mdl[0].eps)
        self._fs_uv_w, self._fs_uv_b, self._fs_uv_c = [], [], []
        for i, fb in enumerate(fsmn_blocks):
            gf = fb.gated_fsmn
            w_parts, b_parts, c_parts = [], [], []
            for branch in (gf.to_u, gf.to_v):
                ln, lin = branch.mdl[0], branch.mdl[1]
                w_parts.append(lin.weight.detach().double() * ln.weight.detach().double().unsqueeze(0))
                b_parts.append(lin.weight.detach().double() @ ln.bias.detach().double() + lin.bias.detach().double())
                c_parts.append(branch.mdl[3].sequential[1].conv.weight.detach())
            self.register_buffer(f'fs_uv_w_{i}', torch.cat(w_parts, dim=0).float().contiguous())
            self.register_buffer(f'fs_uv_b_{i}', torch.cat(b_parts, dim=0).float().contiguous())
            self.register_buffer(f'fs_uv_c_{i}', torch.cat(c_parts, dim=0).contiguous())
            self._fs_uv_w.append(getattr(self, f'fs_uv_w_{i}'))
            self._fs_uv_b.append(getattr(self, f'fs_uv_b_{i}'))
            self._fs_uv_c.append(getattr(self, f'fs_uv_c_{i}'))

        # ---- Speaker tail: fold conv1d_out into output || output_gate for each speaker.
        # This turns conv1d_out + two shared gate projections into one speaker-stacked 1x1
        # convolution: [1, 512, n] -> [1, spks * 1024, n] -> view [spks, 1024, n].
        self.tail_channels = mask_net.conv1_decoder.in_channels
        gate_w = torch.cat((mask_net.output[0].weight.detach(),
                            mask_net.output_gate[0].weight.detach()), dim=0).squeeze(-1).double()
        gate_b = torch.cat((mask_net.output[0].bias.detach(),
                            mask_net.output_gate[0].bias.detach()), dim=0).double()
        tail_w_parts, tail_b_parts = [], []
        conv_out_w = mask_net.conv1d_out.weight.detach()
        conv_out_b = mask_net.conv1d_out.bias.detach()
        for spk in range(self.num_spks):
            start = spk * self.tail_channels
            end = start + self.tail_channels
            spk_w = conv_out_w[start:end, :, 0].double()
            spk_b = conv_out_b[start:end].double()
            tail_w_parts.append((gate_w @ spk_w).float().unsqueeze(-1))
            tail_b_parts.append((gate_w @ spk_b + gate_b).float())
        self.register_buffer('tail_gate_w', torch.cat(tail_w_parts, dim=0).contiguous())
        self.register_buffer('tail_gate_b', torch.cat(tail_b_parts, dim=0).contiguous())

    def norm_audio(self, x, EPS=1e-6):
        # Per-window normalization: reduce over (channel, time) with keepdim so each folded
        # window (batch element) is normalized independently. For batch=1 this equals the
        # original global normalization (mean over the whole clip), bit-identical, and the
        # masked mean pow_x[pow_x>avg].mean() is written as sum(pow_x*mask)/count(mask).
        rms = torch.sqrt((x ** 2).mean(dim=(1, 2), keepdim=True))
        scalar = self.norm_factor / (rms + EPS)
        x = x * scalar
        pow_x = x ** 2
        avg_pow_x = pow_x.mean(dim=(1, 2), keepdim=True)
        mask = (pow_x > avg_pow_x).to(pow_x.dtype)
        rmsx = torch.sqrt((pow_x * mask).sum(dim=(1, 2), keepdim=True) / mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0))
        scalarx = self.norm_factor / (rmsx + EPS)
        x = x * scalarx
        return x, 1.0 / (scalar * scalarx + EPS)

    def _run_mdl(self, mdl_input, n):
        # Inlined Computation_Block -> MossFormerM -> MossformerBlock_GFSMN: the
        # 24 x [FLASH_ShareA_FFConvM + Gated_FSMN_Block_Dilated] run as leaf torch ops.
        # Each FLASH fuses to_hidden || to_qk into one Linear + one depthwise Conv (shared
        # ScaleNorm); each FSMN gate fuses to_u || to_v likewise (shared LayerNorm). The
        # dilated-dense memory stays a leaf module call. Pre-folded weights live in the
        # buffers built in __init__.
        mask_net = self.mossformer_ss.mask_net
        gfsmn = mask_net.mdl.intra_mdl.mossformerM
        flash_layers = gfsmn.layers
        fsmn_blocks = gfsmn.fsmn
        mm_norm = mask_net.mdl.intra_mdl.norm
        intra_norm = mask_net.mdl.intra_norm

        h = mdl_input.permute(0, 2, 1).contiguous()      # [B, n, 512]
        bsz = h.shape[0]                                 # batch = num_window under fold (1 otherwise)
        inv_n = 1.0 if self.fold_lin_inv_n else 1.0 / n

        # Rotary cos/sin sliced to the frame count, broadcast over the 4 OffsetScale heads
        rot_dim = self.rot_dim
        rcos = self.rot_cos[:, :n]                        # [1, n, 1, rot_dim]
        rsin = self.rot_sin[:, :n]

        # Group padding to a multiple of group_size (loop-invariant; static when not dynamic)
        group_size = self.flash_group_size
        if DYNAMIC_AXES:
            remainder = n % group_size
            padding = 0 if remainder == 0 else group_size - remainder
            padded_len = n + padding
            num_groups = padded_len // group_size
        else:
            padding = self.static_padding
            padded_len = self.static_padded_len
            num_groups = self.static_num_groups
        pad_A4 = self.pad_A4[:, :padding].expand(bsz, -1, -1, -1)
        pad_VU = self.pad_VU[:, :padding].expand(bsz, -1, -1)

        for i in range(len(flash_layers)):
            # ===== FLASH_ShareA_FFConvM (fused to_hidden || to_qk) =====
            residual = h
            x_shift, x_pass = h.chunk(2, dim=-1)
            x_shift = torch.cat((self.shift_pad.expand(bsz, -1, -1), x_shift[:, :-1, :]), dim=1)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

            base = normed_x / (torch.norm(normed_x, dim=-1, keepdim=True) + self.fl_norm_eps)
            proj = F.silu(F.linear(base, self._fl_in_w[i], self._fl_in_b[i]))
            proj = proj + F.conv1d(proj.transpose(1, 2), self._fl_in_c[i], None, padding=self.dw_pad, groups=self.fl_in_groups).transpose(1, 2)
            v, u, qk = torch.split(proj, [self.fl_vu, self.fl_vu, self.fl_qk], dim=-1)
            value_proj = proj[..., :self.fl_vu2]

            # OffsetScale (4 heads) kept stacked -> one rotary + one padding for all heads
            scaled = qk.unsqueeze(-2) * self._qkos_gamma[i] + self._qkos_beta[i]   # [1, n, 4, qk]
            mid = scaled[..., :rot_dim]
            half = torch.stack((-mid[..., 1::2], mid[..., 0::2]), dim=-1).flatten(-2)
            scaled = torch.cat((mid * rcos + half * rsin, scaled[..., rot_dim:]), dim=-1)
            if padding > 0:
                scaled = torch.cat((scaled, pad_A4), dim=1)               # [B, padded_len, 4, qk]
            scaled = scaled.reshape(bsz, num_groups, group_size, 4, self.fl_qk)
            quad_q, lin_q, quad_k, lin_k = scaled.split(1, dim=3)
            quad_q = quad_q.squeeze(3)
            lin_q = lin_q.squeeze(3)
            quad_k = quad_k.squeeze(3)
            lin_k = lin_k.squeeze(3)
            if padding > 0:
                vug = torch.cat((value_proj, pad_VU), dim=1).reshape(bsz, num_groups, group_size, self.fl_vu2)
            else:
                vug = value_proj.reshape(bsz, num_groups, group_size, self.fl_vu2)

            # Quadratic attention (padded keys are exact zeros -> no extra masking required)
            attn = F.relu(torch.matmul(quad_q, quad_k.transpose(-1, -2)))
            quad_out = torch.matmul(attn.square(), vug)

            # Linear attention (1/n applied once on the small reduced tensor)
            lin_k_flat = lin_k.permute(0, 3, 1, 2).reshape(bsz, 1, self.fl_qk, padded_len)
            lin_kvu = torch.matmul(lin_k_flat, vug.reshape(bsz, 1, padded_len, self.fl_vu2))
            if not self.fold_lin_inv_n:
                lin_kvu = lin_kvu * inv_n
            lin_out = torch.matmul(lin_q, lin_kvu)

            att_vu = (quad_out + lin_out).reshape(bsz, padded_len, self.fl_vu2)[:, :n, :]
            att_v, att_u = torch.split(att_vu, [self.fl_vu, self.fl_vu], dim=-1)
            out = (att_u * v) * torch.sigmoid(att_v * u)

            # to_out: ScaleNorm (gain folded into weight) -> Linear -> SiLU -> ConvModule
            y = out / (torch.norm(out, dim=-1, keepdim=True) + self.fl_out_norm_eps)
            y = F.silu(F.linear(y, self._fl_out_w[i], self._fl_out_b[i]))
            y = y + F.conv1d(y.transpose(1, 2), self._fl_out_c[i], None, padding=self.dw_pad, groups=self.model_dim).transpose(1, 2)
            h = residual + y

            # ===== Gated_FSMN_Block_Dilated (fused to_u || to_v) =====
            gblk = fsmn_blocks[i]
            blk_in = h
            c1 = gblk.conv1[0]
            c1y = F.prelu(F.conv1d(blk_in.transpose(1, 2), c1.weight, c1.bias), gblk.conv1[1].weight)
            n1 = gblk.norm1
            gf_in = F.layer_norm(c1y.transpose(1, 2), n1.normalized_shape, n1.weight, n1.bias, n1.eps)

            gf = gblk.gated_fsmn
            xn = F.layer_norm(gf_in, self.fs_ln_shape, None, None, self.fs_ln_eps)
            proj = F.silu(F.linear(xn, self._fs_uv_w[i], self._fs_uv_b[i]))
            proj = proj + F.conv1d(proj.transpose(1, 2), self._fs_uv_c[i], None, padding=self.dw_pad, groups=self.fs_uv_groups).transpose(1, 2)
            xu, xv = torch.split(proj, [self.fs_inner, self.fs_inner], dim=-1)

            # Dilated-dense FSMN memory branch (kept as a leaf module call: a squeeze() and
            # InstanceNorm/PReLU dense convs with no fusion benefit)
            xu = gf.fsmn(xu)

            y = xv * xu + gf_in

            n2 = gblk.norm2
            norm2_out = F.layer_norm(y, n2.normalized_shape, n2.weight, n2.bias, n2.eps).transpose(1, 2)
            c2 = gblk.conv2
            h = F.conv1d(norm2_out, c2.weight, c2.bias).transpose(1, 2) + blk_in

        # MossFormerM final LayerNorm, Computation_Block GroupNorm + skip connection
        h = F.layer_norm(h, mm_norm.normalized_shape, mm_norm.weight, mm_norm.bias, mm_norm.eps)
        h = h.permute(0, 2, 1).contiguous()
        h = F.group_norm(h, intra_norm.num_groups, intra_norm.weight, intra_norm.bias, intra_norm.eps)
        return h + mdl_input

    def forward(self, audio):
        audio = audio.float()
        if self.in_sample_rate > MODEL_SAMPLE_RATE:
            audio = torch.nn.functional.interpolate(
                audio,
                size=MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        audio = audio * self.inv_int16
        if self.in_sample_rate < MODEL_SAMPLE_RATE:
            audio = torch.nn.functional.interpolate(
                audio,
                size=MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            # Input length is already a whole number of windows (the tail was padded OUTSIDE
            # the model, in numpy, by the windowing loop), so fold (1, 1, num_window*W) ->
            # (num_window, 1, W). norm_audio + the whole network then run per window (batch).
            audio = audio.reshape(-1, 1, self.fold_window)
        audio, scale = self.norm_audio(audio)
        rms_in = torch.sqrt((audio ** 2).mean(dim=(1, 2), keepdim=True)) * scale * 32767.0   # [B, 1, 1] per window

        mask_net = self.mossformer_ss.mask_net
        # Inlined Encoder.forward: strided Conv1d + ReLU on the [B, 1, L] audio.
        x_enc = torch.nn.functional.relu(self.mossformer_ss.enc.conv1d(audio))    # [B, 512, n]
        n = x_enc.shape[-1] if DYNAMIC_AXES else self.static_frames

        # MaskNet front-end: GroupNorm -> 1x1 conv encoder -> + positional embedding
        mask = mask_net.norm(x_enc)
        mask = mask_net.conv1d_encoder(mask)
        mdl_input = mask + self.emb_pos[..., :n]        # [B, 512, n]

        # Inlined Computation_Block (fused FLASH + FSMN gate stack)
        mask = self._run_mdl(mdl_input, n)

        # ---- MaskNet tail (fused output || output_gate) ----
        # The speaker dimension folds into the batch alongside the window batch, giving a
        # combined batch of (num_window * num_spks), window-major / speaker-minor.
        mask = F.prelu(mask, mask_net.prelu.weight)
        gate_pair = F.conv1d(mask, self.tail_gate_w, self.tail_gate_b)                 # [B, spks*1024, n]
        gate_pair = gate_pair.reshape(-1, self.tail_channels * 2, n)                   # [B*spks, 1024, n]
        m_out, m_gate = torch.split(gate_pair, [self.tail_channels, self.tail_channels], dim=1)
        mask = torch.tanh(m_out) * torch.sigmoid(m_gate)
        mask = F.relu(F.conv1d(mask, mask_net.conv1_decoder.weight, None))             # [B*spks, 512, n]

        # Inlined Decoder.forward as one batched ConvTranspose1d for every window/speaker.
        dec = self.mossformer_ss.dec
        # Broadcast the encoder output across the speaker axis so it lines up with the mask.
        sep = x_enc.unsqueeze(1).expand(-1, self.num_spks, -1, -1).reshape(-1, self.model_dim, n) * mask   # [B*spks, 512, n]
        wav = torch.nn.functional.conv_transpose1d(
            sep, dec.weight, dec.bias, dec.stride, dec.padding,
            dec.output_padding, dec.groups, dec.dilation
        ).squeeze(1)                                                                   # [B*spks, L]
        rms_out = torch.sqrt((wav ** 2).mean(dim=1, keepdim=True))                     # [B*spks, 1]
        rms_in_bs = rms_in.reshape(-1, 1, 1).expand(-1, self.num_spks, 1).reshape(-1, 1)   # [B*spks, 1]
        gain = rms_in_bs / rms_out
        audio_out = wav
        if self.out_sample_rate < MODEL_SAMPLE_RATE:
            audio_out = torch.nn.functional.interpolate(
                audio_out.unsqueeze(0),
                size=OUTPUT_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else MODEL_TO_OUTPUT_SCALE,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        audio_out = audio_out * gain
        if self.out_sample_rate > MODEL_SAMPLE_RATE:
            audio_out = torch.nn.functional.interpolate(
                audio_out.unsqueeze(0),
                size=OUTPUT_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else MODEL_TO_OUTPUT_SCALE,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        audio_out = torch.nan_to_num(audio_out, nan=0.0, posinf=32767.0, neginf=-32768.0)
        if "int" in OUT_AUDIO_DTYPE.lower():
            audio_out = audio_out.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        else:
            audio_out = audio_out * self.inv_int16
            if "16" in OUT_AUDIO_DTYPE:
                audio_out = audio_out.to(torch.float16)
        if self.use_batch_fold:
            # audio_out is [B*spks, L] window-major/speaker-minor -> [B, spks, L]; stitch each
            # speaker's windows back into one contiguous track.
            audio_out = audio_out.reshape(-1, self.num_spks, audio_out.shape[-1])
            return audio_out[:, 0, :].reshape(1, -1), audio_out[:, 1, :].reshape(1, -1)
        return audio_out[0:1], audio_out[1:2]




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_MossFormer_SS_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    mossformer = load_mossformer2_ss_model(model_path).eval().float().to("cpu")
    mossformer = MOSSFORMER_SS(mossformer, INPUT_AUDIO_LENGTH, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH)
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.randint(low=-32768, high=32767, size=(1, 1, EXPORT_AUDIO_LENGTH), dtype=torch.int16).to(IN_TORCH_DTYPE)

    torch.onnx.export(
        mossformer,
        (audio,),
        onnx_model_A,
        input_names=['mix_audio'],
        output_names=['separated_0', 'separated_1'],
        do_constant_folding=True,
        dynamic_axes={
            'mix_audio': {2: 'audio_len'},
            'separated_0': {1: 'out_audio_len'},
            'separated_1': {1: 'out_audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17,
        dynamo=False
    )
    del mossformer
    del audio
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="MossFormer2_SS_16K", task="source_separation", model_family="mossformer2_ss",
    max_dynamic_audio_seconds=6, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=1, feature_kind="conv_encoder_decoder", center_pad=False, pad_mode=None,
    extra={"pad_head": PAD_HEAD, "enc_stride": 8, "output_sources": 2},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
