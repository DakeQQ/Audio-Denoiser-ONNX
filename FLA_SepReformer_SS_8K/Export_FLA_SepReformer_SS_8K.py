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
import torch.nn as nn
import torch.nn.functional as F

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, build_model_metadata, export_metadata_carrier, metadata_path_for_model

# User settings.
model_path              = str(Path.home() / "Downloads" / "speech_flatsepreformer_separation_temporal_8k_base_libri2mix100")  # The FLA-SepReformer download folder.
onnx_model_A            = str(Path(__file__).resolve().parent / "FLA_SepReformer_ONNX" / "FLA_SepReformer_SS_8K.onnx")
DYNAMIC_AXES            = True         # Group norms and linear attention span the whole utterance: run whole audio, static chunks degrade quality.
IN_SAMPLE_RATE          = 8000         # [8000, 16000, 22500, 24000, 44000, 48000]; input audio sample rate.
OUT_SAMPLE_RATE         = 8000         # [8000, 16000, 22500, 24000, 44000, 48000]; output audio sample rate.
INPUT_AUDIO_LENGTH      = IN_SAMPLE_RATE * 2  # Static window (DYNAMIC_AXES=False) in input-rate samples; short windows lack the context to split the first words.
PAD_HEAD                = 0            # Leading zeros shift the utterance statistics; keep 0 to match upstream.
IN_AUDIO_DTYPE          = 'F32'        # ['F16', 'F32'] use normalized audio [-1, 1]; 'INT16' uses PCM [-32768, 32767].
OUT_AUDIO_DTYPE         = 'F32'        # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor.

# Fixed FLA-SepReformer separation model and ONNX export parameters.
OPSET                   = 20
MODEL_SAMPLE_RATE       = 8000
ENC_KERNEL              = 16
ENC_STRIDE              = 4
NUM_SPKS                = 2
INV_INT16               = float(1.0 / 32768.0)
MAX_DYNAMIC_AUDIO_SECONDS = 20

# Derived export dimensions.
onnx_model_Metadata     = str(metadata_path_for_model(onnx_model_A))
MODEL_AUDIO_LENGTH      = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
OUTPUT_AUDIO_LENGTH     = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
INPUT_TO_MODEL_SCALE    = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE   = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE   = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
EXPORT_AUDIO_LENGTH     = INPUT_AUDIO_LENGTH

# Pristine upstream network from the installed modelscope package (no site-package patching).
from modelscope.models.audio.separation.tsepreformer.sepreformer_layers import TSepReformerSeparator


def load_sepreformer_model(checkpoint_dir):
    """Builds TSepReformerSeparator from configuration.json and strictly loads pytorch_model.pt."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / 'configuration.json', 'r', encoding='utf-8') as f:
        model_conf = json.load(f)['model']
    separator_conf = model_conf['separator_conf']
    enc_conf = separator_conf['module_audio_enc']
    if (model_conf.get('normalize_variance', False) or int(separator_conf['num_spks']) != NUM_SPKS
            or (int(enc_conf['kernel_size']), int(enc_conf['stride'])) != (ENC_KERNEL, ENC_STRIDE)):
        raise ValueError(f'Unsupported FLA-SepReformer configuration in configuration.json: {model_conf}')
    separator = TSepReformerSeparator(**separator_conf)
    state = torch.load(checkpoint_dir / 'pytorch_model.pt', map_location='cpu', weights_only=True)
    prefix = 'separator.'
    state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    separator.load_state_dict(state, strict=True)
    return separator


class FLA_SEPREFORMER_SS(torch.nn.Module):
    def __init__(self, separator, input_audio_len, in_sample_rate, out_sample_rate):
        super(FLA_SEPREFORMER_SS, self).__init__()
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.num_spks = separator.num_spk
        self.static_audio_len = int(round(input_audio_len * MODEL_SAMPLE_RATE / in_sample_rate))
        self.out_peak = 16384.0 if 'int' in OUT_AUDIO_DTYPE.lower() else 0.5

        enc = separator.audio_encoder.conv1d
        proj = separator.feature_projector
        dec = separator.audio_decoder
        if enc.bias is not None or proj.conv1d.bias is not None or dec.bias is not None or proj.conv1d.kernel_size[0] != 1:
            raise ValueError('The weight folding requires bias-free encoder / projector / decoder convs and a 1x1 projector.')
        # PCM input: x / 32768 is folded into the bias-free encoder conv.
        in_scale = INV_INT16 if 'int' in IN_AUDIO_DTYPE.lower() else 1.0
        self.register_buffer('enc_w', (enc.weight.detach() * in_scale).float().contiguous())
        # GroupNorm affine folded into the 1x1 projector: W (g * x_hat + b) = (W * g) x_hat + W b.
        w = proj.conv1d.weight.detach()
        self.register_buffer('proj_w', (w * proj.norm.weight.detach().view(1, -1, 1)).float().contiguous())
        self.register_buffer('proj_b', (w[..., 0] @ proj.norm.bias.detach()).float().contiguous())
        self.proj_eps = float(proj.norm.eps)
        self.register_buffer('dec_w', dec.weight.detach().float().contiguous())

        sep = separator.separator
        self.num_stages = sep.num_stages
        self.pad_multiple = 2 ** sep.num_stages
        if any(stage.downconv.down_conv.stride[0] != 2 for stage in sep.enc_stages):
            raise ValueError('The exact x2 nearest upsampling requires stride-2 downsampling.')
        # The relative position embedding is never consumed upstream, so the stages run with pos_k=None.
        self.enc_stages = sep.enc_stages
        self.bottleneck = sep.bottleneck_G
        self.spk_split = sep.spk_split_block
        self.simple_fusion = sep.simple_fusion
        self.dec_stages = sep.dec_stages
        self.end_conv1x1 = separator.out_layer.end_conv1x1

    def separate(self, x):
        skips = []
        for stage in self.enc_stages:
            x, skip = stage(x, None)
            skips.append(self.spk_split(skip))
        x, _ = self.bottleneck(x, None)
        x = self.spk_split(x)                                                               # [B*spks, C, T/16]
        for idx in range(self.num_stages):
            # Frames are a multiple of 2**num_stages, so upstream's nearest resize is an exact x2 repeat.
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.simple_fusion[idx](torch.cat([x, skips[self.num_stages - 1 - idx]], dim=1))
            x, _ = self.dec_stages[idx](x, None)
        return x                                                                            # [B*spks, C, T_pad]

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
        bsz = audio.shape[0]
        audio_len = audio.shape[-1] if DYNAMIC_AXES else self.static_audio_len
        enc = F.gelu(F.conv1d(audio, self.enc_w, stride=ENC_STRIDE))                       # [B, 256, T]
        frames = enc.shape[-1]
        x = F.conv1d(F.group_norm(enc, 1, None, None, self.proj_eps), self.proj_w, self.proj_b)
        # Upstream zero-pads the projected frames (after the group norm) to a multiple of 2**num_stages.
        if torch.onnx.is_in_onnx_export() and DYNAMIC_AXES:
            rest = torch.remainder(-torch._shape_as_tensor(x)[2], self.pad_multiple)
        else:
            rest = (-frames) % self.pad_multiple
        x = F.pad(x, (0, rest))
        x = self.separate(x)[..., :frames]
        # Native 1x1 convs operate directly on [B*spks, 128, frames] without layout transpositions
        x = self.end_conv1x1(x)                                                             # [B*spks, 256, T]
        wav = F.conv_transpose1d(x, self.dec_w, stride=ENC_STRIDE)
        # The decoder drops the (L - kernel) % stride tail; upstream zero-pads it back.
        wav = F.pad(wav, (0, ENC_STRIDE - 1))[..., :audio_len]
        audio_out = wav.reshape(bsz, self.num_spks, -1)
        if self.out_sample_rate != MODEL_SAMPLE_RATE:
            audio_out = F.interpolate(
                audio_out,
                size=OUTPUT_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else MODEL_TO_OUTPUT_SCALE,
                mode='linear',
                align_corners=False
            )
        # Upstream pipeline peak-normalizes each source to 0.5; raw outputs exceed full scale and clip.
        peak = audio_out.abs().amax(dim=-1, keepdim=True)
        audio_out = audio_out * torch.where(peak > 0.0, self.out_peak / peak, torch.zeros_like(peak))
        if "int" in OUT_AUDIO_DTYPE.lower():
            # Clamp in int32: FP16 rounds +32767 to +32768, which would wrap when cast straight to int16.
            audio_out = audio_out.to(torch.int32).clamp(min=-32768, max=32767).to(torch.int16)
        elif "16" in OUT_AUDIO_DTYPE:
            audio_out = audio_out.to(torch.float16)
        # Upstream speaker order is the reverse of MossFormer2_SS / FLA_TFLocoformer_SS_8K.
        wav_0, wav_1 = torch.split(audio_out, 1, dim=1)
        return (wav_1, wav_0,) if DYNAMIC_AXES else (wav_0, wav_1,)                        # spks x [B, 1, L]


# ---------------------------------------------------------------------------
# Pre-Export Graph Optimizations (Weight Folding, QKV Fusion, Cancelled Math)
# ---------------------------------------------------------------------------

def optimize_separator(separator):
    """Applies mathematically equivalent pre-export fusions and graph optimizations:
    1. Folds LayerScale into preceding Linear layers across all 56 GCFNs.
    2. Folds BatchNorm1d into preceding Linear and LayerScale into projection across all 22 CLAs.
    3. Folds BatchNorm1d into preceding depthwise Conv1d across all 4 DownConv layers.
    4. Fuses Q, K, V projections and folds LayerScale into linear_out across all 12 SpkAttentions.
    5. Fuses Q, KV projections, precomputes scale factor, and cancels sequence scaling in FLA.
    6. Converts OutputLayer end_conv1x1 into direct 1x1 Conv1ds, eliminating 2 full-tensor transposes.
    """
    sep = separator.separator

    # 1. Fold GCFNs: fold Layer_scale directly into net2[2] (Linear 384 -> 128)
    for name, module in separator.named_modules():
        if module.__class__.__name__ == 'GCFN':
            lin = module.net2[2]
            ls = module.Layer_scale
            if hasattr(ls, 'layer_scale') and ls.layer_scale is not None:
                scale = ls.layer_scale.detach().view(-1)
                lin.weight.data.mul_(scale.view(-1, 1))
                lin.bias.data.mul_(scale)
                module.Layer_scale = nn.Identity()

            def new_gcfn_forward(self, x):
                y = self.net1(x)
                y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
                y = self.net2(y)
                return x + y

            module.forward = new_gcfn_forward.__get__(module, module.__class__)

    # 2. Fold CLA: fold BN into linear2, fold Layer_scale into linear3[1], bypass permutes
    for name, module in separator.named_modules():
        if module.__class__.__name__ == 'CLA':
            lin2 = module.linear2
            bn = module.BN
            gamma = bn.weight.detach()
            beta = bn.bias.detach()
            mean = bn.running_mean.detach()
            var = bn.running_var.detach()
            eps = bn.eps
            std = torch.sqrt(var + eps)
            scale = gamma / std
            lin2.weight.data.mul_(scale.view(-1, 1))
            lin2.bias.data.sub_(mean).mul_(scale).add_(beta)
            module.BN = nn.Identity()

            lin3 = module.linear3[1]
            ls = module.Layer_scale
            if hasattr(ls, 'layer_scale') and ls.layer_scale is not None:
                scale_ls = ls.layer_scale.detach().view(-1)
                lin3.weight.data.mul_(scale_ls.view(-1, 1))
                lin3.bias.data.mul_(scale_ls)
                module.Layer_scale = nn.Identity()

            def new_cla_forward(self, x):
                y = self.layer_norm(x)
                y = self.linear1(y)
                y = self.GLU(y)
                y = self.dw_conv_1d(y.transpose(1, 2)).transpose(1, 2)
                y = self.linear2(y)
                y = self.linear3(y)
                return x + y

            module.forward = new_cla_forward.__get__(module, module.__class__)

    # 3. Fold DownConvLayer: fold BN into down_conv
    for stage in sep.enc_stages:
        dc = stage.downconv
        dw = dc.down_conv
        bn = dc.BN
        gamma = bn.weight.detach()
        beta = bn.bias.detach()
        mean = bn.running_mean.detach()
        var = bn.running_var.detach()
        eps = bn.eps
        std = torch.sqrt(var + eps)
        scale = gamma / std
        dw.weight.data.mul_(scale.view(-1, 1, 1))
        dw.bias.data.sub_(mean).mul_(scale).add_(beta)
        dc.BN = nn.Identity()

        def new_dc_forward(self, x):
            y = self.down_conv(x.permute(0, 2, 1))
            y = self.gelu(y)
            return y.permute(0, 2, 1)

        dc.forward = new_dc_forward.__get__(dc, dc.__class__)

    # 4. SpkAttention: fuse Q, K, V and fold Layer_scale
    class FusedMHA(nn.Module):
        def __init__(self, raw_mha):
            super().__init__()
            self.h = raw_mha.h
            self.d_k = raw_mha.d_k
            self.layer_norm = raw_mha.layer_norm

            wq = raw_mha.linear_q.weight.detach()
            bq = raw_mha.linear_q.bias.detach()
            wk = raw_mha.linear_k.weight.detach()
            bk = raw_mha.linear_k.bias.detach()
            wv = raw_mha.linear_v.weight.detach()
            bv = raw_mha.linear_v.bias.detach()

            self.w_qkv = nn.Linear(128, 384)
            self.w_qkv.weight.data.copy_(torch.cat([wq, wk, wv], dim=0))
            self.w_qkv.bias.data.copy_(torch.cat([bq, bk, bv], dim=0))

            self.linear_out = nn.Linear(128, 128)
            ls = raw_mha.Layer_scale.layer_scale.detach().view(-1)
            self.linear_out.weight.data.copy_(raw_mha.linear_out.weight.detach() * ls.view(-1, 1))
            self.linear_out.bias.data.copy_(raw_mha.linear_out.bias.detach() * ls)

        def forward(self, x, num_spk=2):
            n_batch, len_seq, c = x.shape
            x_norm = self.layer_norm(x)
            qkv = self.w_qkv(x_norm)
            q, k, v = torch.split(qkv, [128, 128, 128], dim=-1)

            q = q.view(n_batch, len_seq, self.h, self.d_k).transpose(1, 2)
            k = k.view(n_batch, len_seq, self.h, self.d_k).transpose(1, 2)
            v = v.view(n_batch, len_seq, self.h, self.d_k).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * 0.25
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(n_batch, len_seq, 128)
            return self.linear_out(out)

    for stage in sep.dec_stages:
        for i in (1, 2, 3):
            spk_attn = getattr(stage, f'spk_attn_{i}')
            spk_attn.self_attn = FusedMHA(spk_attn.self_attn)

    # 5. FocusedLinearAttention: QKV fusion + precomputed scale + n^-0.5 cancellation
    class OptimizedFLA(nn.Module):
        def __init__(self, raw_fla):
            super().__init__()
            self.num_heads = raw_fla.num_heads
            self.focusing_factor = raw_fla.focusing_factor

            inv_scale = (1.0 / F.softplus(raw_fla.scale.detach())).reshape(1, 1, 128)
            self.register_buffer('inv_scale', inv_scale.float())

            wq = raw_fla.q.weight.detach()
            bq = raw_fla.q.bias.detach()
            wk = raw_fla.kv.weight[:128].detach()
            bk = raw_fla.kv.bias[:128].detach()
            wv = raw_fla.kv.weight[128:].detach()
            bv = raw_fla.kv.bias[128:].detach()

            self.w_qkv = nn.Linear(128, 384)
            self.w_qkv.weight.data.copy_(torch.cat([wq, wk, wv], dim=0))
            self.w_qkv.bias.data.copy_(torch.cat([bq, bk, bv], dim=0))

            self.dwc = raw_fla.dwc
            self.proj = raw_fla.proj

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.w_qkv(x)
            q, k, v = torch.split(qkv, [128, 128, 128], dim=-1)
            n = N

            q = (F.relu(q) + 1e-6) * self.inv_scale
            k = (F.relu(k) + 1e-6) * self.inv_scale

            q_norm = q.norm(dim=-1, keepdim=True)
            k_norm = k.norm(dim=-1, keepdim=True)
            q = q ** self.focusing_factor
            k = k ** self.focusing_factor
            q = (q / (q.norm(dim=-1, keepdim=True) + 1e-8)) * q_norm
            k = (k / (k.norm(dim=-1, keepdim=True) + 1e-8)) * k_norm

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
            k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
            v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

            # Cancelled sequence length scaling: (q @ (k^T @ v)) / (q @ k_sum^T + 1e-6 * n)
            k_sum = k.sum(dim=-2, keepdim=True)
            denom = torch.matmul(q, k_sum.transpose(-2, -1)) + (1e-6 * n)
            kv = torch.matmul(k.transpose(-2, -1), v)
            attn_out = torch.matmul(q, kv) / denom

            out = attn_out.transpose(1, 2).reshape(B, N, C)

            if self.dwc is not None:
                v_dwc = v.reshape(B * self.num_heads, n, -1).permute(0, 2, 1)
                out = out + self.dwc(v_dwc).reshape(B, C, N).permute(0, 2, 1)

            return self.proj(out)

    for name, module in separator.named_modules():
        if module.__class__.__name__ == 'Linear_Attn_Template':
            module.attn = OptimizedFLA(module.attn)

    # 6. Output 1x1 Conv1d: replaces Linear(128, 512) -> GLU -> Linear(256, 256)
    class OptimizedEndConv1x1(nn.Module):
        def __init__(self, raw_end_conv):
            super().__init__()
            lin1 = raw_end_conv[0]
            lin2 = raw_end_conv[2]
            self.conv1 = nn.Conv1d(128, 512, kernel_size=1)
            self.conv1.weight.data.copy_(lin1.weight.detach().unsqueeze(-1))
            self.conv1.bias.data.copy_(lin1.bias.detach())
            self.conv2 = nn.Conv1d(256, 256, kernel_size=1)
            self.conv2.weight.data.copy_(lin2.weight.detach().unsqueeze(-1))
            self.conv2.bias.data.copy_(lin2.bias.detach())

        def forward(self, x):
            x = F.glu(self.conv1(x), dim=1)
            return self.conv2(x)

    separator.out_layer.end_conv1x1 = OptimizedEndConv1x1(separator.out_layer.end_conv1x1)
    return separator


def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_FLA_SepReformer_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


if __name__ == '__main__':
    print('Export start ...')
    Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        separator = load_sepreformer_model(model_path).eval().float().to("cpu")
        separator = optimize_separator(separator)
        model = FLA_SEPREFORMER_SS(separator, INPUT_AUDIO_LENGTH, IN_SAMPLE_RATE, OUT_SAMPLE_RATE).eval()
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
        globals(), producer=Path(__file__).name, model_name="FLA_SepReformer_SS_8K", task="source_separation", model_family="tsepreformer",
        max_dynamic_audio_seconds=MAX_DYNAMIC_AUDIO_SECONDS, normalize_audio_default=False, input_channels=1, output_channels=1,
        num_audio_inputs=1, feature_kind="waveform",
        extra={"pad_head": PAD_HEAD, "output_sources": NUM_SPKS, "enc_stride": ENC_STRIDE},
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

