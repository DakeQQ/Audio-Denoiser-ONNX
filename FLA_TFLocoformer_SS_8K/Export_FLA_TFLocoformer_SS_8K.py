import gc
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, build_model_metadata, export_metadata_carrier, metadata_path_for_model
from STFT_Process import STFT_Process

# User settings.
model_path              = str(Path.home() / "Downloads" / "speech_flatflocoformer_separation_timefrequency_8k_middle_libri2mix360")  # The FLA-TFLocoformer-M download folder.
onnx_model_A            = str(Path(__file__).resolve().parent / "FLA_TFLocoformer_ONNX" / "FLA_TFLocoformer_SS_8K.onnx")
DYNAMIC_AXES            = False        # Utterance-level variance norm + time attention: run whole audio, static chunks degrade quality.
IN_SAMPLE_RATE          = 8000         # [8000, 16000, 22500, 24000, 44000, 48000]; input audio sample rate.
OUT_SAMPLE_RATE         = 8000         # [8000, 16000, 22500, 24000, 44000, 48000]; output audio sample rate.
INPUT_AUDIO_LENGTH      = IN_SAMPLE_RATE * 3  # Static window in input-rate samples; short windows lack the context to split the first words. 3 seconds is recommended.
PAD_HEAD                = 0            # Leading zeros shift the utterance statistics; keep 0 to match upstream.
IN_AUDIO_DTYPE          = 'F32'        # ['F16', 'F32'] use normalized audio [-1, 1]; 'INT16' uses PCM [-32768, 32767].
OUT_AUDIO_DTYPE         = 'F32'        # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor.
BATCH_WINDOW_SECONDS    = 1.5          # Minimum input length (seconds) that triggers window folding.
USE_BATCH_FOLD          = False        # Batch-fold long audio into fixed windows.

# Fixed FLA-TFLocoformer separation model and ONNX export parameters.
OPSET                   = 20
MODEL_SAMPLE_RATE       = 8000
WINDOW_TYPE             = 'hann'
NFFT                    = 128
WINDOW_LENGTH           = 128
HOP_LENGTH              = 64
PAD_MODE                = 'reflect'
INV_INT16               = float(1.0 / 32768.0)
MAX_DYNAMIC_AUDIO_SECONDS = 6

# Derived export dimensions.
onnx_model_Metadata     = str(metadata_path_for_model(onnx_model_A))
MODEL_AUDIO_LENGTH      = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
OUTPUT_AUDIO_LENGTH     = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
INPUT_TO_MODEL_SCALE    = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE   = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE   = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
FOLD_WINDOW_LENGTH      = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH
EXPORT_AUDIO_LENGTH     = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH
STATIC_WINDOW_LENGTH    = FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else MODEL_AUDIO_LENGTH
MAX_SIGNAL_LENGTH       = (int(MAX_DYNAMIC_AUDIO_SECONDS * MODEL_SAMPLE_RATE) if DYNAMIC_AXES else STATIC_WINDOW_LENGTH) // HOP_LENGTH + 1

# Pristine upstream network from the installed modelscope package (no site-package patching).
from modelscope.models.audio.separation.flatflocoformer.locoformer_block import FLATFLocoformerSeparator


def load_flatflocoformer_model(checkpoint_dir):
    """Builds FLATFLocoformerSeparator from configuration.json and strictly loads pytorch_model.pt."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / 'configuration.json', 'r', encoding='utf-8') as f:
        model_conf = json.load(f)['model']
    if (int(model_conf['n_fft']), int(model_conf['hop_length'])) != (NFFT, HOP_LENGTH) or not model_conf.get('normalize_variance', True):
        raise ValueError(f'Unsupported STFT configuration in configuration.json: {model_conf}')
    separator = FLATFLocoformerSeparator(**model_conf['separator_conf'])
    state = torch.load(checkpoint_dir / 'pytorch_model.pt', map_location='cpu', weights_only=True)
    prefix = 'separator.'
    state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    separator.load_state_dict(state, strict=True)
    return separator


def _rms_fold(norm):
    """Returns (gamma * sqrt(dim_per_group), eps * sqrt(dim_per_group)) for x / (||x_g|| + eps')."""
    root = float(norm.dim_per_group) ** 0.5
    return norm.gamma.detach().double() * root, float(norm.eps) * root


class FLA_TFLOCOFORMER_SS(torch.nn.Module):
    def __init__(self, separator, input_audio_len, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(FLA_TFLOCOFORMER_SS, self).__init__()
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.num_spks = separator.num_spk
        # Variance normalization is scale-invariant, so the PCM input scale only reappears via std at the output.
        in_scale = 1.0 if 'int' in IN_AUDIO_DTYPE.lower() else 32767.0
        out_scale = 1.0 if 'int' in OUT_AUDIO_DTYPE.lower() else INV_INT16
        self.out_gain = in_scale * out_scale
        self.use_batch_fold = use_batch_fold
        self.fold_window = fold_window

        logical_model_audio_len = int(round(input_audio_len * MODEL_SAMPLE_RATE / in_sample_rate))
        if use_batch_fold and (DYNAMIC_AXES or in_sample_rate != MODEL_SAMPLE_RATE or out_sample_rate != MODEL_SAMPLE_RATE or fold_window <= 0):
            raise ValueError('Batch folding requires static axes, equal input/model/output sample rates, and a positive fold window.')
        model_audio_len = fold_window if use_batch_fold else (MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else logical_model_audio_len)
        self.static_window_batch = ((logical_model_audio_len + fold_window - 1) // fold_window) if use_batch_fold else 1
        self.static_window_length = model_audio_len
        self.static_frames = model_audio_len // HOP_LENGTH + 1
        self.static_output_length = self.static_window_batch * model_audio_len
        self.f_bins = NFFT // 2 + 1

        self.stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH,
                                 max_frames=0 if DYNAMIC_AXES else MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE,
                                 center_pad=True, pad_mode=PAD_MODE, precompute_static=not DYNAMIC_AXES,
                                 static_input_length=model_audio_len).eval()
        # torch.istft(length=L) keeps the L % hop tail samples covered by the last half window, which
        # STFT_Process's centered output (frames - 1) * hop drops; so slice the full overlap-add instead.
        istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH,
                             max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True,
                             pad_mode=PAD_MODE, precompute_static=False)
        self.half_n_fft = NFFT // 2
        self.register_buffer('istft_kernel', istft.inverse_kernel.detach().contiguous())
        if DYNAMIC_AXES:
            self.register_buffer('win_sq_kernel', istft.win_sq_kernel.detach().contiguous())
        else:
            win_sum = F.conv_transpose1d(torch.ones(1, 1, self.static_frames), istft.win_sq_kernel, stride=HOP_LENGTH)
            win_sum = win_sum[..., self.half_n_fft:self.half_n_fft + model_audio_len]
            self.register_buffer('inv_win_sum', (win_sum.double().clamp_min(1e-11).reciprocal() * self.out_gain).float().contiguous())

        # Front Conv2d + GroupNorm(1, C).
        conv, gn = separator.conv[0], separator.conv[1]
        self.emb_dim = conv.out_channels
        self.register_buffer('front_w', conv.weight.detach().transpose(-1, -2).contiguous())
        self.register_buffer('front_b', conv.bias.detach().contiguous())
        self.register_buffer('front_gn_w', gn.weight.detach().reshape(1, -1, 1, 1).contiguous())
        self.register_buffer('front_gn_b', gn.bias.detach().reshape(1, -1, 1, 1).contiguous())
        self.front_gn_eps = float(gn.eps)
        self.register_buffer('gn_one', torch.ones(1, dtype=torch.float32))
        self.register_buffer('gn_zero', torch.zeros(1, dtype=torch.float32))
        # ConvTranspose2d(pad=1, k=3) == Conv2d(pad=1) with in/out swapped and the kernel flipped.
        self.register_buffer('back_w', separator.deconv.weight.detach().transpose(-1, -2).transpose(0, 1).flip(2, 3).contiguous())
        self.register_buffer('back_b', separator.deconv.bias.detach().contiguous())

        block0 = separator.f_blocks[0].block if len(separator.f_blocks) else separator.t_blocks[0].block
        norm0 = block0.ffn_norm[0]
        self.num_groups = int(norm0.num_groups)
        self.group_dim = int(norm0.dim_per_group)
        ffn0 = block0.ffn[0].branchs[0]
        self.ffn_inner = ffn0.deconv1d.in_channels
        self.ffn_kernel = ffn0.conv1d.kernel_size[0]
        if ffn0.conv1d.stride[0] != 1:
            raise ValueError('The exact Pad/Slice removal requires conv1d_shift == 1.')
        self.ffn_pad = self.ffn_kernel - 1

        # Rotary tables along the fixed frequency axis; interleaved pairs, sign folded into sin.
        mhsa0 = separator.f_blocks[0].block.attn
        self.n_heads = mhsa0.attns[0].n_heads
        self.head_dim = mhsa0.attns[0].d_k
        self.attn_dim = self.n_heads * self.head_dim
        freqs = mhsa0.attns[0].rope.freqs.detach().float()
        ang = torch.arange(self.f_bins, dtype=torch.float32).unsqueeze(-1) * freqs
        ang = torch.stack((ang, ang), dim=-1).flatten(-2)                              # [F, head_dim]
        sign = torch.tensor([-1.0, 1.0]).repeat(self.head_dim // 2)
        self.register_buffer('rot_cos', ang.cos().contiguous())
        self.register_buffer('rot_signed_sin', (ang.sin() * sign).contiguous())

        self.layers_type = list(separator.layers_type)
        self.blocks = []
        f_idx = t_idx = 0
        for li, kind in enumerate(self.layers_type):
            if kind == 'f':
                blk = separator.f_blocks[f_idx].block
                f_idx += 1
            else:
                blk = separator.t_blocks[t_idx].block
                t_idx += 1
            # f-layers run Conv2d on [B, C, F, T]; t-layers run Conv1d on [B*F, C, T] (faster in ORT CPU).
            kh = (self.ffn_kernel, 1) if kind == 'f' else (self.ffn_kernel,)
            pad = (self.ffn_pad, 0) if kind == 'f' else self.ffn_pad
            entry = {'kind': kind, 'ffn': [], 'pad': pad}
            for idx in (-1, 0):
                g, eps = _rms_fold(blk.ffn_norm[idx])
                br = blk.ffn[idx].branchs[0]
                w_in = (br.conv1d.weight.detach().double() * g.reshape(1, -1, 1)).float().reshape(br.conv1d.out_channels, -1, *kh)
                # ConvTranspose(pad=K-1) == Conv(pad=0) with in/out swapped and the kernel flipped.
                w_out = br.deconv1d.weight.detach().reshape(br.deconv1d.in_channels, -1, *kh).transpose(0, 1).flip(2)
                b_in = br.conv1d.bias.detach()
                n = self.ffn_inner
                names = [f'ffn_v_w_{li}_{idx}', f'ffn_v_b_{li}_{idx}', f'ffn_g_w_{li}_{idx}', f'ffn_g_b_{li}_{idx}', f'ffn_out_w_{li}_{idx}', f'ffn_out_b_{li}_{idx}']
                for name, t in zip(names, (w_in[:n], b_in[:n], w_in[n:], b_in[n:], w_out, br.deconv1d.bias.detach())):
                    self.register_buffer(name, t.contiguous())
                entry['ffn'].append((eps, names))
            if kind == 'f':
                g, eps = _rms_fold(blk.attn_norm)
                w = blk.attn.qkv.weight.detach().double() * g.reshape(1, -1)
                w[:self.attn_dim] *= self.head_dim ** -0.5                             # SDPA scale, commutes with RoPE.
                # Per-head [3H, C, d] weights let [B, T, 1, F, C] @ W emit [B, T, 3H, F, d] with no head transpose.
                self.register_buffer(f'qkv_w_{li}', w.float().reshape(3 * self.n_heads, self.head_dim, -1).transpose(1, 2).contiguous())
                self.register_buffer(f'agg_w_{li}', blk.attn.aggregate_heads[0].weight.detach().reshape(self.emb_dim, -1).t().contiguous())
                entry['attn_eps'] = eps
            else:
                ega = blk.attn.ega
                fla = ega.attn
                g_ln, eps = _rms_fold(ega.layer_norm)
                g_gate, eps_gate = _rms_fold(ega.gate_layer[0])
                if abs(eps - eps_gate) > 0.0:
                    raise ValueError('Shared normalization requires equal layer_norm / gate_layer eps.')
                inv_s = 1.0 / F.softplus(fla.scale.detach().double()).reshape(-1)        # relu(x)/s == relu(x/s) for s > 0
                wq = fla.q.weight.detach().double() * inv_s.unsqueeze(1)
                bq = fla.q.bias.detach().double() * inv_s
                wk, wv = fla.kv.weight.detach().double().split(self.attn_dim, dim=0)
                bk, bv = fla.kv.bias.detach().double().split(self.attn_dim, dim=0)
                # Weights are [C_out, C_in] and biases [C_out, 1] for W @ y on y = [B, F, C, T].
                proj = (('q', wq * g_ln, bq), ('k', wk * inv_s.unsqueeze(1) * g_ln, bk * inv_s), ('v', wv * g_ln, bv),
                        ('g', ega.gate_layer[1].weight.detach().double() * g_gate, ega.gate_layer[1].bias.detach().double()),
                        ('p', fla.proj.weight.detach().double(), fla.proj.bias.detach().double()))
                self.register_buffer(f'la_qkvg_w_{li}', torch.cat([p[1] for p in proj[:4]], dim=0).float().contiguous())
                self.register_buffer(f'la_qkvg_b_{li}', torch.cat([p[2] for p in proj[:4]], dim=0).float().reshape(-1, 1).contiguous())
                self.register_buffer(f'la_p_w_{li}', proj[4][1].float().contiguous())
                self.register_buffer(f'la_p_b_{li}', proj[4][2].float().reshape(-1, 1).contiguous())
                self.register_buffer(f'la_eps_{li}', (1e-6 * inv_s).float().reshape(-1, 1).contiguous())
                if fla.dwc.groups != fla.dwc.in_channels or fla.dwc.in_channels != self.head_dim:
                    raise ValueError('The per-head dwc path requires a depthwise conv over head_dim channels.')
                self.register_buffer(f'la_dwc_w_{li}', fla.dwc.weight.detach().contiguous())
                self.register_buffer(f'la_dwc_b_{li}', fla.dwc.bias.detach().contiguous())
                entry['attn_eps'] = eps
                entry['focus'] = int(fla.focusing_factor)
                entry['dwc_pad'] = int(fla.dwc.padding[0])
            self.blocks.append(entry)

    def rms_group_norm(self, x, eps, bsz, frames):
        # Gamma and the 1/sqrt(dim_per_group) factor are folded into the consumer weights and eps.
        y = x.reshape(bsz, self.num_groups, self.group_dim, self.f_bins, frames)
        y = y / (torch.norm(y, dim=2, keepdim=True) + eps)
        return y.reshape(bsz, self.emb_dim, self.f_bins, frames)

    def rms_group_norm_t(self, x, eps, bsz, frames):
        y = x.reshape(bsz * self.f_bins, self.num_groups, self.group_dim, frames)
        y = y / (torch.norm(y, dim=2, keepdim=True) + eps)
        return y.reshape(bsz * self.f_bins, self.emb_dim, frames)

    def swiglu_ffn(self, x, eps, names, pad, bsz, frames, kind):
        w_v, b_v, w_g, b_g, w_out, b_out = (getattr(self, name) for name in names)
        conv = F.conv2d if kind == 'f' else F.conv1d
        y = self.rms_group_norm(x, eps, bsz, frames) if kind == 'f' else self.rms_group_norm_t(x, eps, bsz, frames)
        value = conv(y, w_v, b_v, padding=pad)
        gate = conv(y, w_g, b_g, padding=pad)
        return x + conv(value * F.silu(gate), w_out, b_out)

    def freq_mhsa(self, x, li, eps, bsz, frames):
        y = self.rms_group_norm(x, eps, bsz, frames).permute(0, 3, 2, 1).unsqueeze(2)      # [B, T, 1, F, C]
        qkv = torch.matmul(y, getattr(self, f'qkv_w_{li}'))                                 # [B, T, 3H, F, d]
        qk, v = torch.split(qkv, [2 * self.n_heads, self.n_heads], dim=2)
        swapped = qk.reshape(bsz, frames, 2 * self.n_heads, self.f_bins, self.head_dim // 2, 2).flip(-1)
        qk = qk * self.rot_cos + swapped.reshape(qk.shape) * self.rot_signed_sin
        q, k = torch.split(qk, self.n_heads, dim=2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        o = torch.matmul(attn, v)                                                           # [B, T, H, F, d]
        o = torch.matmul(o.transpose(2, 3).reshape(bsz, frames, self.f_bins, self.attn_dim), getattr(self, f'agg_w_{li}'))
        return x + o.permute(0, 3, 2, 1)

    def time_linear_attn(self, x, li, eps, focus, dwc_pad, bsz, frames, eps_n):
        y = self.rms_group_norm_t(x, eps, bsz, frames)                                      # [B*F, C, T]
        heads = (bsz, self.f_bins, self.n_heads, self.head_dim, frames)
        qkvg = torch.matmul(getattr(self, f'la_qkvg_w_{li}'), y) + getattr(self, f'la_qkvg_b_{li}')  # [B*F, 4C, T]
        qk, v, g = torch.split(qkvg, [2 * self.attn_dim, self.attn_dim, self.attn_dim], dim=1)
        # relu(q)/s + 1e-6/s with 1/s folded into the projection.
        qk = F.relu(qk.reshape(-1, 2, self.attn_dim, frames)) + getattr(self, f'la_eps_{li}')  # [B*F, 2, C, T]
        qk_norm = torch.norm(qk, dim=2, keepdim=True)
        qk = qk ** focus
        qk = (qk / (torch.norm(qk, dim=2, keepdim=True) + 1e-8)) * qk_norm
        qk = qk.reshape(bsz, self.f_bins, 2 * self.n_heads, self.head_dim, frames)
        q, k = torch.split(qk, self.n_heads, dim=2)                                         # [B, F, H, d, T]
        vh = v.reshape(heads)
        # q.(k_sum/n) + 1e-6 and q.(k^T v)/n share 1/n, so it cancels into eps_n = 1e-6 * n.
        den = torch.matmul(k.sum(dim=-1).unsqueeze(-2), q) + eps_n                         # [B, F, H, 1, T]
        o = torch.matmul(torch.matmul(vh, k.transpose(-1, -2)), q) / den                   # [B, F, H, d, T]
        dwc = F.conv1d(v.reshape(-1, self.head_dim, frames), getattr(self, f'la_dwc_w_{li}'), getattr(self, f'la_dwc_b_{li}'),
                       padding=dwc_pad, groups=self.head_dim)
        o = o.reshape(bsz * self.f_bins, self.attn_dim, frames) + dwc.reshape(bsz * self.f_bins, self.attn_dim, frames)
        o = torch.matmul(getattr(self, f'la_p_w_{li}'), o) + getattr(self, f'la_p_b_{li}')
        # Upstream adds the input twice: gated residual inside the attention plus the block residual.
        return x + x + torch.sigmoid(g) * o

    def separate(self, spec, bsz, frames, eps_n):
        # stft_B channels are [real | imag], so [B, 2F, T] -> [B, 2, F, T] is a metadata-only reshape.
        x = F.conv2d(spec.reshape(bsz, 2, self.f_bins, frames), self.front_w, self.front_b, padding=1)
        if DYNAMIC_AXES:
            x = F.group_norm(x, 1, None, None, self.front_gn_eps)
        else:
            x = F.instance_norm(x.reshape(bsz, 1, -1), None, None, self.gn_one, self.gn_zero, True, 0.1, self.front_gn_eps)
            x = x.reshape(bsz, self.emb_dim, self.f_bins, frames)
        x = x * self.front_gn_w + self.front_gn_b
        for li, entry in enumerate(self.blocks):
            kind = entry['kind']
            if kind == 't':
                x = x.transpose(1, 2).reshape(bsz * self.f_bins, self.emb_dim, frames)
            eps_pre, names_pre = entry['ffn'][0]
            x = self.swiglu_ffn(x, eps_pre, names_pre, entry['pad'], bsz, frames, kind)
            if kind == 'f':
                x = self.freq_mhsa(x, li, entry['attn_eps'], bsz, frames)
            else:
                x = self.time_linear_attn(x, li, entry['attn_eps'], entry['focus'], entry['dwc_pad'], bsz, frames, eps_n)
            eps_post, names_post = entry['ffn'][1]
            x = self.swiglu_ffn(x, eps_post, names_post, entry['pad'], bsz, frames, kind)
            if kind == 't':
                x = x.reshape(bsz, self.f_bins, self.emb_dim, frames).transpose(1, 2)
        return F.conv2d(x, self.back_w, self.back_b, padding=1)                               # [B, spks*2, F, T]

    def forward(self, audio):
        audio = audio.float()
        if self.in_sample_rate != MODEL_SAMPLE_RATE:
            audio = F.interpolate(
                audio,
                size=MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            audio = audio.reshape(self.static_window_batch, 1, self.fold_window)
        bsz = audio.shape[0] if DYNAMIC_AXES else self.static_window_batch
        audio_len = audio.shape[-1] if DYNAMIC_AXES else self.static_window_length

        # Upstream variance normalization (unbiased std). A silent window gets zero gain instead of NaN.
        mean = audio.mean(dim=-1, keepdim=True)
        centered = audio - mean
        var = (centered * centered).sum(dim=-1, keepdim=True) / (audio_len - 1)
        std = torch.sqrt(var)
        inv_std = torch.where(std > 0.0, 1.0 / std, torch.zeros_like(std))
        audio = audio * inv_std

        stft = self.stft
        if stft._reflect_gather:
            audio = audio.index_select(2, stft.reflect_indices)
        elif stft._reflect_pad:
            audio = F.pad(audio, (stft.half_n_fft, stft.half_n_fft), mode='reflect')
        spec = F.conv1d(audio, stft.stft_kernel, stride=HOP_LENGTH, padding=stft._conv_padding)  # [B, 2F, T]
        frames = spec.shape[-1] if DYNAMIC_AXES else self.static_frames
        spec = self.separate(spec, bsz, frames, 1e-6 * frames)
        spec = spec.reshape(bsz * self.num_spks, 2 * self.f_bins, frames)
        wav = F.conv_transpose1d(spec, self.istft_kernel, stride=HOP_LENGTH)
        wav = wav[..., self.half_n_fft:self.half_n_fft + audio_len]                          # [B*spks, 1, L]
        if DYNAMIC_AXES:
            ones = torch.ones((1, 1, frames), dtype=wav.dtype, device=wav.device)
            win_sum = F.conv_transpose1d(ones, self.win_sq_kernel, stride=HOP_LENGTH)
            wav = wav / win_sum[..., self.half_n_fft:self.half_n_fft + audio_len].clamp_min(1e-11)
            std = std * self.out_gain
        else:
            wav = wav * self.inv_win_sum
        audio_out = wav.reshape(bsz, self.num_spks, -1) * std
        if self.out_sample_rate != MODEL_SAMPLE_RATE:
            audio_out = F.interpolate(
                audio_out.reshape(1, bsz * self.num_spks, -1),
                size=OUTPUT_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else MODEL_TO_OUTPUT_SCALE,
                mode='linear',
                align_corners=False
            ).reshape(bsz, self.num_spks, -1)
        # Upstream pipeline peak-normalizes each source to 0.5; raw outputs exceed full scale and clip.
        peak = audio_out.abs().amax(dim=(0, 2), keepdim=True)
        audio_out = audio_out * torch.where(peak > 0.0, (16384.0 if "int" in OUT_AUDIO_DTYPE.lower() else 0.5) / peak, torch.zeros_like(peak))
        if "int" in OUT_AUDIO_DTYPE.lower():
            # Clamp in int32: FP16 rounds +32767 to +32768, which would wrap when cast straight to int16.
            audio_out = audio_out.to(torch.int32).clamp(min=-32768, max=32767).to(torch.int16)
        elif "16" in OUT_AUDIO_DTYPE:
            audio_out = audio_out.to(torch.float16)
        if self.use_batch_fold:
            audio_out = audio_out.transpose(0, 1).reshape(self.num_spks, 1, self.static_output_length)
            return torch.split(audio_out, 1, dim=0)
        # bsz == 1 here: splitting [1, spks, L] on dim 1 yields [1, 1, L] outputs without a Transpose.
        return torch.split(audio_out, 1, dim=1) if bsz == 1 else torch.split(audio_out.transpose(0, 1), 1, dim=0)


def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_FLA_TFLocoformer_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


if __name__ == '__main__':
    print('Export start ...')
    Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        separator = load_flatflocoformer_model(model_path).eval().float().to("cpu")
        model = FLA_TFLOCOFORMER_SS(separator, INPUT_AUDIO_LENGTH, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH)
        del separator
        if "32" in IN_AUDIO_DTYPE:
            IN_TORCH_DTYPE = torch.float32
        elif "int" in IN_AUDIO_DTYPE.lower():
            IN_TORCH_DTYPE = torch.int16
        else:
            IN_TORCH_DTYPE = torch.float16
        if "int" in IN_AUDIO_DTYPE.lower():
            audio = torch.randint(low=-32768, high=32768, size=(1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
        else:
            audio = torch.rand((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE) * 2.0 - 1.0

        export_buffer = io.BytesIO()
        torch.onnx.export(
            model,
            (audio,),
            export_buffer,
            input_names=['mix_audio'],
            output_names=['separated_0', 'separated_1'],
            do_constant_folding=True,
            dynamic_axes={
                'mix_audio': {2: 'audio_len'},
                'separated_0': {2: 'out_audio_len'},
                'separated_1': {2: 'out_audio_len'}
            } if DYNAMIC_AXES else None,
            opset_version=OPSET,
            dynamo=False
        )
        export_bytes = export_buffer.getvalue()
        del model, export_buffer, audio
        gc.collect()
    model_metadata = build_audio_metadata_from_globals(
        globals(), producer=Path(__file__).name, model_name="FLA_TFLocoformer_SS_8K", task="source_separation", model_family="flatflocoformer",
        max_dynamic_audio_seconds=MAX_DYNAMIC_AUDIO_SECONDS, normalize_audio_default=False, input_channels=1, output_channels=1,
        num_audio_inputs=1, feature_kind="stft", center_pad=True, pad_mode=PAD_MODE,
        extra={"pad_head": PAD_HEAD, "output_sources": 2},
    )
    model_metadata = build_model_metadata(model_metadata)
    onnx_proto = onnx.load_model_from_string(export_bytes)
    del export_bytes
    for key, value in model_metadata.items():
        onnx_proto.metadata_props.add(key=key, value=value)
    final_bytes = onnx_proto.SerializeToString()
    del onnx_proto
    final_path = Path(onnx_model_A).expanduser().resolve()
    fd, staging = tempfile.mkstemp(prefix=f'.{final_path.name}.', suffix='.tmp', dir=final_path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(final_bytes)
            f.flush()
            os.fsync(f.fileno())
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(staging, 0o666 & ~umask)                                                   # mkstemp creates 0600.
        os.replace(staging, final_path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise
    del final_bytes
    print(f"Model saved to: {final_path}")
    export_metadata_carrier(onnx_model_Metadata, model_metadata, OPSET)
    print(f"Metadata saved to: {onnx_model_Metadata}")
    print('\nExport done!')
    _run_inference_demo()
