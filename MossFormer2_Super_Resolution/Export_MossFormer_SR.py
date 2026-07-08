import gc
import subprocess
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata
from Example_Audio import model_audio_path


model_path           = "/home/DakeQQ/Downloads/MossFormer2_SR_48K"
parent_path          = Path(__file__).resolve().parent                                      # The folder that contains this script.
onnx_model_A         = str(parent_path / "MossFormer_ONNX" / "MossFormer2_SR.onnx")        # The exported onnx model path.
onnx_model_Metadata  = str(metadata_path_for_model(onnx_model_A))                           # The metadata carrier onnx model path.
test_audio           = model_audio_path("mossformer2_super_resolution")                     # The original audio path.
save_generated_audio = str(parent_path / "super_resolution.wav")                            # The output super resolution audio path.


DYNAMIC_AXES         = False    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET                = 18
IN_AUDIO_DTYPE       = 'INT16'  # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'INT16'  # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16            = float(1.0 / 32768.0)

ORIGINAL_SAMPLE_RATE = 16000    # The input audio sample rate. This value cannot be changed after the ONNX model is exported.
SUPER_SAMPLE_RATE    = 48000    # The target audio sample rate, do not edit the value.
INPUT_AUDIO_LENGTH   = 32000    # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 8000. Higher values yield better quality but time consume. It is better to set an integer multiple of the NFFT value.

WINDOW_TYPE          = 'hann'   # Type of window function used in the STFT (clearvoice mel_spectrogram and bandwidth_sub both use a Hann window)
N_MELS               = 80       # Number of Mel bands to generate in the Mel-spectrogram
NFFT                 = 1024     # Number of FFT components for the STFT process
WINDOW_LENGTH        = 1024     # Length of windowing, edit it carefully.
HOP_LENGTH           = 256      # Number of samples between successive frames in the STFT
MAX_SIGNAL_LENGTH    = 1280     # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.

NFFT_POST            = 256      # The MossFormer_SR parameter, do not edit the value.
WINDOW_LENGTH_POST   = 256      # The MossFormer_SR parameter, do not edit the value.
HOP_LENGTH_POST      = 128      # The MossFormer_SR parameter, do not edit the value.

IN_SAMPLE_RATE       = ORIGINAL_SAMPLE_RATE    # Input audio sample rate (16 kHz).
MODEL_SAMPLE_RATE    = SUPER_SAMPLE_RATE       # Internal STFT/network rate (48 kHz).
OUT_SAMPLE_RATE      = SUPER_SAMPLE_RATE       # Output audio sample rate (48 kHz = 3x the input).

from clearvoice.models.mossformer2_sr.mossformer2_sr_wrapper import MossFormer2_SR_48K


def load_mossformer2_sr_model(checkpoint_dir):
    """Builds the MossFormer2_SR_48K network and loads its checkpoints.

    This inlines ClearVoice's model construction and ``SpeechModel._load_model``
    checkpoint-loading logic so the export script is self-contained. The network
    is an ``nn.ModuleList`` holding the MossFormer MaskNet (``model_m``) and the
    HiFi-GAN Generator (``model_g``); their weights live in two separate
    checkpoints listed (one per line) in ``last_best_checkpoint``.
    """
    # HiFi-GAN Generator hyper-parameters taken verbatim from
    # clearvoice/config/inference/MossFormer2_SR_48K.json (do not edit).
    args = argparse.Namespace(
        resblock="1",
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=1024,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )
    wrapper = MossFormer2_SR_48K(args)
    model = torch.nn.ModuleList([wrapper.model_m, wrapper.model_g])

    # Read the two checkpoint filenames from 'last_best_checkpoint' (line 1 -> the
    # MossFormer MaskNet, line 2 -> the Generator) and load each sub-model.
    best_name = os.path.join(checkpoint_dir, 'last_best_checkpoint')
    with open(best_name, 'r') as f:
        checkpoint_names = [f.readline().strip(), f.readline().strip()]
    model_keys = ['mossformer', 'generator']
    for sub_model, checkpoint_name, model_key in zip(model, checkpoint_names, model_keys):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        pretrained_model = checkpoint[model_key] if model_key in checkpoint else checkpoint
        state = sub_model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
                state[key] = pretrained_model[key.replace('module.', '')]
            elif 'module.' + key in pretrained_model and state[key].shape == pretrained_model['module.' + key].shape:
                state[key] = pretrained_model['module.' + key]
        sub_model.load_state_dict(state)
    model[1].remove_weight_norm()
    return model


class MOSSFORMER_SR(torch.nn.Module):
    def __init__(
        self,
        mossformer_sr,
        pre_stft,
        pre_nfft,
        n_mels,
        original_sample_rate,
        super_sample_rate,
        input_audio_len: int = INPUT_AUDIO_LENGTH,
        crossover_hz: float = 5500.0,
        crossover_taps: int = 511,
        crossover_beta: float = 8.0,
        resample_halfwidth: int = 32,
    ):
        super(MOSSFORMER_SR, self).__init__()
        self.input_scale = 1.0 / 32768.0  # int16 PCM -> [-1, 1]; fused into the resample kernel
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.mossformer_sr = mossformer_sr
        self.mask_net = mossformer_sr[0].mossformer
        self.generator = mossformer_sr[1]
        self.pre_stft = pre_stft
        fbank = torchaudio.functional.melscale_fbanks(pre_nfft // 2 + 1, 0, 8000, n_mels, super_sample_rate, 'slaney', 'slaney')
        self.register_buffer('fbank', fbank.transpose(0, 1).unsqueeze(0).contiguous())
        self.scale_factor = float(super_sample_rate / original_sample_rate)

        model_audio_len = int(round(float(input_audio_len) * self.scale_factor))
        # HiFi-GAN mel framing (clearvoice meldataset.mel_spectrogram): the signal is reflect
        # padded by (n_fft - hop) / 2 on each side and a center=False STFT is taken. pre_stft is
        # built with center_pad=False and the reflect padding is applied explicitly in forward().
        self.mel_pad = (pre_nfft - pre_stft.hop_len) // 2
        self.static_frames = (model_audio_len + 2 * self.mel_pad - pre_nfft) // pre_stft.hop_len + 1
        self.static_audio_len = model_audio_len
        # The HiFi-GAN Generator upsamples each mel frame back to hop_len samples, so its output is
        # static_frames * hop_len samples. Pad it back up to the input length so it aligns with the
        # up-sampled input sample-for-sample for the bandwidth-substitution crossover below.
        self.gen_pad = model_audio_len - self.static_frames * pre_stft.hop_len

        # ---- Bandwidth-substitution crossover -------------------------------------------------
        # Speech super-resolution here = keep the pristine low band of the (up-sampled) input, add
        # the HiFi-GAN generator's synthesized high band (the content the low-resolution input
        # never had). The split is a single linear-phase (symmetric) windowed-sinc FIR pair which
        # reconstructs perfectly and adds no phase distortion:
        #     out = lowpass(input) + highpass(generator)
        #         = lowpass(input) + (generator - lowpass(generator))
        # This replaces (a) the original's zero-phase Butterworth filtfilt (not ONNX-exportable,
        # content-dependent cut-off -> level pumping across windows) and (b) the previous short
        # 256-pt STFT mask + time-domain edge taper (spectral blocking + amplitude-modulation
        # flutter on the synthesized band). `crossover_hz` trades brightness (lower -> more
        # generator, more "air") against low-band fidelity (higher -> more pristine input kept);
        # ~5.5 kHz keeps the perceptually critical 0-5.5 kHz (pitch + first formants) verbatim and
        # lets the generator restore the upper formants / fricatives and the missing >8 kHz octave.
        crossover_taps = int(crossover_taps) | 1  # force odd -> exact linear phase, integer delay
        c = (crossover_taps - 1) // 2
        nyq = 0.5 * float(super_sample_rate)
        fc = min(max(float(crossover_hz), 0.0), nyq)
        taps_idx = np.arange(crossover_taps, dtype=np.float32) - c
        window = np.kaiser(crossover_taps, float(crossover_beta)).astype(np.float32)
        h = np.sinc(2.0 * fc / float(super_sample_rate) * taps_idx) * window
        h = h / h.sum()  # unit-DC-gain low-pass prototype; the high-pass is (delta - h)
        self.register_buffer('xover_lp', torch.from_numpy(h).float().view(1, 1, crossover_taps).contiguous())
        self.xover_half = c

        # High-quality polyphase (windowed-sinc) up-sampler. Linear interpolation colours the
        # low band and leaks spectral images into the mel features (a mismatch with the
        # high-quality sinc resampling the model was trained on). For an integer ratio L the
        # windowed-sinc interpolation filter is applied as a single strided ConvTranspose1d,
        # yielding exactly L * input_len samples for any input length (ONNX/ORT friendly).
        L = int(round(self.scale_factor))
        self.resample_L = L
        if abs(L - self.scale_factor) < 1e-6 and L >= 2:
            K = int(resample_halfwidth)
            M = 2 * L * K + 1                          # symmetric tap count (odd)
            n = np.arange(M, dtype=np.float32) - (M - 1) / 2.0
            window = np.kaiser(M, 9.0).astype(np.float32)
            h = np.sinc(n / L) * window                # ideal interpolation kernel, Kaiser window
            for p in range(L):                         # unit DC gain per polyphase branch
                h[p::L] /= h[p::L].sum()
            # Align preserved source samples to output indices 0, L, 2L... while keeping
            # the exact L * input_len output length through ConvTranspose output padding.
            self.resample_pad = (M - 1) // 2
            self.resample_output_padding = L - 1
            # Fold the int16 -> [-1, 1] scale into the interpolation kernel (conv is linear, so
            # conv(audio / 32768, h) == conv(audio, h / 32768)).
            self.register_buffer('resample_kernel', torch.from_numpy(h * self.input_scale).float().view(1, 1, M))
        else:
            self.resample_kernel = None
            self.resample_pad = 0
            self.resample_output_padding = 0

        mask_net = self.mask_net
        gfsmn = mask_net.mdl.intra_mdl.mossformerM
        flash_layers = gfsmn.layers
        fsmn_blocks = gfsmn.fsmn
        flash0 = flash_layers[0]
        max_seq = max(MAX_SIGNAL_LENGTH, self.static_frames + 16)

        t = torch.arange(max_seq, dtype=torch.float32)
        sinu = t.unsqueeze(-1) * mask_net.pos_enc.inv_freq
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        emb_pos = (emb * mask_net.pos_enc.scale).transpose(0, -1).unsqueeze(0)
        self.register_buffer('emb_pos', emb_pos.contiguous().half())

        group_size = flash0.group_size
        qk_dim = flash0.qk_offset_scale.gamma.shape[-1]
        vu_dim = flash0.to_hidden.mdl[1].weight.shape[0] // 2
        model_dim = flash0.to_hidden.mdl[1].weight.shape[1]
        dw_kernel = flash0.to_hidden.mdl[3].sequential[1].conv.weight.shape[-1]
        self.model_dim = model_dim
        self.flash_group_size = group_size
        self.fl_vu = vu_dim
        self.fl_vu2 = vu_dim * 2
        self.fl_qk = qk_dim
        self.fl_in_groups = self.fl_vu2 + qk_dim
        self.dw_pad = (dw_kernel - 1) // 2
        self.fl_inv_g = float(1.0 / group_size)
        self.static_padding = (group_size - (self.static_frames % group_size)) % group_size
        self.static_padded_len = self.static_frames + self.static_padding
        self.static_num_groups = self.static_padded_len // group_size
        self.static_inv_n = float(1.0 / self.static_frames)
        self.fold_lin_inv_n = not DYNAMIC_AXES
        self.fl_scale = float(flash0.to_hidden.mdl[0].scale)
        self.fl_eps = float(flash0.to_hidden.mdl[0].eps)
        self.fl_norm_eps = float(self.fl_eps / self.fl_scale)
        self.fl_in_scale_fold = float(1.0 / self.fl_scale)
        self.fl_out_scale = float(flash0.to_out.mdl[0].scale)
        self.fl_out_eps = float(flash0.to_out.mdl[0].eps)
        self.fl_out_norm_eps = float(self.fl_out_eps / self.fl_out_scale)
        self.fl_out_scale_fold = float(1.0 / self.fl_out_scale)

        rot_freqs = flash0.rotary_pos_emb.freqs
        self.rot_dim = int(2 * rot_freqs.shape[0])
        rot_ang = torch.arange(max_seq, dtype=rot_freqs.dtype).unsqueeze(-1) * rot_freqs
        rot_ang = torch.stack((rot_ang, rot_ang), dim=-1).flatten(-2)
        self.register_buffer('rot_cos', rot_ang.cos().half().unsqueeze(0).unsqueeze(2).contiguous())
        self.register_buffer('rot_sin', rot_ang.sin().half().unsqueeze(0).unsqueeze(2).contiguous())
        self.register_buffer('shift_pad', torch.zeros((1, 1, model_dim // 2), dtype=torch.float32))
        self.register_buffer('pad_A4', torch.zeros((1, group_size, 4, qk_dim), dtype=torch.float32))
        self.register_buffer('pad_VU', torch.zeros((1, group_size, vu_dim * 2), dtype=torch.float32))

        self._fl_in_w, self._fl_in_b, self._fl_in_c = [], [], []
        self._fl_out_w, self._fl_out_b, self._fl_out_c = [], [], []
        self._qkos_gamma, self._qkos_beta = [], []
        for i, fl in enumerate(flash_layers):
            gh = fl.to_hidden.mdl[0].g.detach().float()
            gqk = fl.to_qk.mdl[0].g.detach().float()
            w_in = torch.cat((fl.to_hidden.mdl[1].weight.detach().float() * gh * self.fl_in_scale_fold,
                              fl.to_qk.mdl[1].weight.detach().float() * gqk * self.fl_in_scale_fold), dim=0).float().contiguous()
            b_in = torch.cat((fl.to_hidden.mdl[1].bias.detach(),
                              fl.to_qk.mdl[1].bias.detach()), dim=0).float().contiguous()
            c_in = torch.cat((fl.to_hidden.mdl[3].sequential[1].conv.weight.detach(),
                              fl.to_qk.mdl[3].sequential[1].conv.weight.detach()), dim=0).contiguous()
            w_out = (fl.to_out.mdl[1].weight.detach().float() * fl.to_out.mdl[0].g.detach().float() * self.fl_out_scale_fold).float().contiguous()
            qk_scale = torch.ones((4, 1), dtype=torch.float32)
            qk_scale[0, 0] = self.fl_inv_g
            if self.fold_lin_inv_n:
                qk_scale[3, 0] = self.static_inv_n
            qkos_gamma = (fl.qk_offset_scale.gamma.detach().float() * qk_scale).float().contiguous()
            qkos_beta = (fl.qk_offset_scale.beta.detach().float() * qk_scale).float().contiguous()
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

        gf0 = fsmn_blocks[0].gated_fsmn
        self.fs_inner = gf0.to_u.mdl[1].weight.shape[0]
        self.fs_uv_groups = self.fs_inner * 2
        self.fs_ln_shape = tuple(gf0.to_u.mdl[0].normalized_shape)
        self.fs_ln_eps = float(gf0.to_u.mdl[0].eps)
        self.register_buffer('fsmn_pad', torch.zeros((1, gf0.fsmn.output_dim, gf0.fsmn.lorder - 1), dtype=torch.float32))
        self._fs_uv_w, self._fs_uv_b, self._fs_uv_c, self._fs_mem_c = [], [], [], []
        for i, fb in enumerate(fsmn_blocks):
            gf = fb.gated_fsmn
            w_parts, b_parts, c_parts = [], [], []
            for branch in (gf.to_u, gf.to_v):
                ln, lin = branch.mdl[0], branch.mdl[1]
                w_parts.append(lin.weight.detach().float() * ln.weight.detach().float().unsqueeze(0))
                b_parts.append(lin.weight.detach().float() @ ln.bias.detach().float() + lin.bias.detach().float())
                c_parts.append(branch.mdl[3].sequential[1].conv.weight.detach())
            self.register_buffer(f'fs_uv_w_{i}', torch.cat(w_parts, dim=0).float().contiguous())
            self.register_buffer(f'fs_uv_b_{i}', torch.cat(b_parts, dim=0).float().contiguous())
            self.register_buffer(f'fs_uv_c_{i}', torch.cat(c_parts, dim=0).contiguous())
            self.register_buffer(f'fs_mem_c_{i}', gf.fsmn.conv1.weight.detach().squeeze(-1).contiguous())
            self._fs_uv_w.append(getattr(self, f'fs_uv_w_{i}'))
            self._fs_uv_b.append(getattr(self, f'fs_uv_b_{i}'))
            self._fs_uv_c.append(getattr(self, f'fs_uv_c_{i}'))
            self._fs_mem_c.append(getattr(self, f'fs_mem_c_{i}'))

        self.tail_channels = mask_net.conv1_decoder.in_channels
        spk_w = mask_net.conv1d_out.weight.detach()[:self.tail_channels, :, 0].float()
        spk_b = mask_net.conv1d_out.bias.detach()[:self.tail_channels].float()
        gate_w = torch.cat((mask_net.output[0].weight.detach(),
                            mask_net.output_gate[0].weight.detach()), dim=0).squeeze(-1).float()
        gate_b = torch.cat((mask_net.output[0].bias.detach(),
                            mask_net.output_gate[0].bias.detach()), dim=0).float()
        self.register_buffer('tail_gate_w', (gate_w @ spk_w).float().unsqueeze(-1).contiguous())
        self.register_buffer('tail_gate_b', (gate_w @ spk_b + gate_b).float().contiguous())

        gen = self.generator
        self.gen_num_upsamples = gen.num_upsamples
        self.gen_num_kernels = gen.num_kernels
        self.gen_resblock_scale = float(1.0 / gen.num_kernels)
        self._gen_snake_inv = []
        self._gen_res_c1_inv, self._gen_res_c2_inv = [], []
        for i, snake_act in enumerate(gen.snakes):
            inv_alpha = (snake_act.alpha.detach().float() + 1e-9).reciprocal().contiguous()
            self.register_buffer(f'gen_snake_inv_{i}', inv_alpha)
            self._gen_snake_inv.append(getattr(self, f'gen_snake_inv_{i}'))
        self.register_buffer('gen_snake_post_inv', (gen.snake_post.alpha.detach().float() + 1e-9).reciprocal().contiguous())
        for i, block in enumerate(gen.resblocks):
            c1_invs, c2_invs = [], []
            if hasattr(block, 'convs1'):
                for j, act in enumerate(block.convs1_activates):
                    self.register_buffer(f'gen_res_{i}_c1_inv_{j}', (act.alpha.detach().float() + 1e-9).reciprocal().contiguous())
                    c1_invs.append(getattr(self, f'gen_res_{i}_c1_inv_{j}'))
                for j, act in enumerate(block.convs2_activates):
                    self.register_buffer(f'gen_res_{i}_c2_inv_{j}', (act.alpha.detach().float() + 1e-9).reciprocal().contiguous())
                    c2_invs.append(getattr(self, f'gen_res_{i}_c2_inv_{j}'))
            else:
                for j, act in enumerate(block.convs_activates):
                    self.register_buffer(f'gen_res_{i}_c1_inv_{j}', (act.alpha.detach().float() + 1e-9).reciprocal().contiguous())
                    c1_invs.append(getattr(self, f'gen_res_{i}_c1_inv_{j}'))
            self._gen_res_c1_inv.append(c1_invs)
            self._gen_res_c2_inv.append(c2_invs)

    def _run_mdl(self, mdl_input, n):
        mask_net = self.mask_net
        gfsmn = mask_net.mdl.intra_mdl.mossformerM
        flash_layers = gfsmn.layers
        fsmn_blocks = gfsmn.fsmn
        mm_norm = mask_net.mdl.intra_mdl.norm
        intra_norm = mask_net.mdl.intra_norm

        h = mdl_input.permute(0, 2, 1).contiguous()
        inv_n = 1.0 if self.fold_lin_inv_n else 1.0 / n
        rcos = self.rot_cos[:, :n].float()
        rsin = self.rot_sin[:, :n].float()

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
        pad_A4 = self.pad_A4[:, :padding]
        pad_VU = self.pad_VU[:, :padding]

        for i in range(len(flash_layers)):
            residual = h
            x_shift, x_pass = h.chunk(2, dim=-1)
            x_shift = torch.cat((self.shift_pad, x_shift[:, :-1, :]), dim=1)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

            base = normed_x / torch.clamp(torch.norm(normed_x, dim=-1, keepdim=True), min=self.fl_norm_eps)
            proj = F.silu(F.linear(base, self._fl_in_w[i], self._fl_in_b[i]))
            proj = proj + F.conv1d(proj.transpose(1, 2), self._fl_in_c[i], None, padding=self.dw_pad, groups=self.fl_in_groups).transpose(1, 2)
            v, u, qk = torch.split(proj, [self.fl_vu, self.fl_vu, self.fl_qk], dim=-1)
            value_proj = proj[..., :self.fl_vu2]

            scaled = qk.unsqueeze(-2) * self._qkos_gamma[i] + self._qkos_beta[i]
            mid, tail = torch.split(scaled, [self.rot_dim, self.fl_qk - self.rot_dim], dim=-1)
            mid_even, mid_odd = mid.reshape(1, n, 4, self.rot_dim // 2, 2).split(1, dim=-1)
            half = torch.cat((-mid_odd, mid_even), dim=-1).flatten(-2)
            scaled = torch.cat((mid * rcos + half * rsin, tail), dim=-1)
            if padding > 0:
                scaled = torch.cat((scaled, pad_A4), dim=1)
            scaled = scaled.reshape(1, num_groups, group_size, 4, self.fl_qk)
            quad_q, lin_q, quad_k, lin_k = scaled.split(1, dim=3)
            quad_q = quad_q.squeeze(3)
            lin_q = lin_q.squeeze(3)
            quad_k = quad_k.squeeze(3)
            lin_k = lin_k.squeeze(3)
            if padding > 0:
                vug = torch.cat((value_proj, pad_VU), dim=1).reshape(1, num_groups, group_size, self.fl_vu2)
            else:
                vug = value_proj.reshape(1, num_groups, group_size, self.fl_vu2)

            attn = F.relu(torch.matmul(quad_q, quad_k.transpose(-1, -2)))
            quad_out = torch.matmul(attn.square(), vug)
            lin_k_flat = lin_k.permute(0, 3, 1, 2).reshape(1, 1, self.fl_qk, padded_len)
            lin_kvu = torch.matmul(lin_k_flat, vug.reshape(1, 1, padded_len, self.fl_vu2))
            if not self.fold_lin_inv_n:
                lin_kvu = lin_kvu * inv_n
            lin_out = torch.matmul(lin_q, lin_kvu)

            att_vu = (quad_out + lin_out).reshape(1, padded_len, self.fl_vu2)[:, :n, :]
            att_v, att_u = torch.split(att_vu, [self.fl_vu, self.fl_vu], dim=-1)
            out = (att_u * v) * torch.sigmoid(att_v * u)

            y = out / torch.clamp(torch.norm(out, dim=-1, keepdim=True), min=self.fl_out_norm_eps)
            y = F.silu(F.linear(y, self._fl_out_w[i], self._fl_out_b[i]))
            y = y + F.conv1d(y.transpose(1, 2), self._fl_out_c[i], None, padding=self.dw_pad, groups=self.model_dim).transpose(1, 2)
            h = residual + y

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

            uf = gf.fsmn
            f1 = F.relu(F.linear(xu, uf.linear.weight, uf.linear.bias))
            xp = F.linear(f1, uf.project.weight, None).transpose(1, 2)
            yy = torch.cat((self.fsmn_pad, xp, self.fsmn_pad), dim=2)
            yy = F.conv1d(yy, self._fs_mem_c[i], None, groups=uf.conv1.groups)
            xu = xu + (xp + yy).transpose(1, 2)

            y = xv * xu + gf_in
            n2 = gblk.norm2
            norm2_out = F.layer_norm(y, n2.normalized_shape, n2.weight, n2.bias, n2.eps).transpose(1, 2)
            c2 = gblk.conv2
            h = F.conv1d(norm2_out, c2.weight, c2.bias).transpose(1, 2) + blk_in

        h = F.layer_norm(h, mm_norm.normalized_shape, mm_norm.weight, mm_norm.bias, mm_norm.eps)
        h = h.permute(0, 2, 1).contiguous()
        h = F.group_norm(h, intra_norm.num_groups, intra_norm.weight, intra_norm.bias, intra_norm.eps)
        return h + mdl_input

    def _run_masknet(self, mel_features, n):
        mask_net = self.mask_net
        x = mask_net.norm(mel_features)
        x = mask_net.conv1d_encoder(x)
        x = self._run_mdl(x + self.emb_pos[..., :n].float(), n)
        x = F.prelu(x, mask_net.prelu.weight)
        gate_pair = F.conv1d(x, self.tail_gate_w, self.tail_gate_b)
        x_out, x_gate = torch.split(gate_pair, [self.tail_channels, self.tail_channels], dim=1)
        x = torch.tanh(x_out) * torch.sigmoid(x_gate)
        return F.relu(F.conv1d(x, mask_net.conv1_decoder.weight, None))

    def _snake(self, x, alpha, inv_alpha):
        return x + inv_alpha * torch.sin(alpha * x).square()

    def _conv1d(self, x, conv):
        return F.conv1d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)

    def _conv_transpose1d(self, x, conv):
        return F.conv_transpose1d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.output_padding, conv.groups, conv.dilation)

    def _run_resblock(self, block, block_index, x):
        if hasattr(block, 'convs1'):
            c1_invs = self._gen_res_c1_inv[block_index]
            c2_invs = self._gen_res_c2_inv[block_index]
            for j in range(len(block.convs1)):
                y = self._snake(x, block.convs1_activates[j].alpha, c1_invs[j])
                y = self._conv1d(y, block.convs1[j])
                y = self._snake(y, block.convs2_activates[j].alpha, c2_invs[j])
                y = self._conv1d(y, block.convs2[j])
                x = x + y
            return x
        c1_invs = self._gen_res_c1_inv[block_index]
        for j in range(len(block.convs)):
            y = self._snake(x, block.convs_activates[j].alpha, c1_invs[j])
            y = self._conv1d(y, block.convs[j])
            x = x + y
        return x

    def _run_generator(self, x):
        gen = self.generator
        x = self._conv1d(x, gen.conv_pre)
        for i in range(self.gen_num_upsamples):
            x = self._snake(x, gen.snakes[i].alpha, self._gen_snake_inv[i])
            x = self._conv_transpose1d(x, gen.ups[i])
            base = i * self.gen_num_kernels
            acc = self._run_resblock(gen.resblocks[base], base, x)
            for j in range(1, self.gen_num_kernels):
                acc = acc + self._run_resblock(gen.resblocks[base + j], base + j, x)
            x = acc * self.gen_resblock_scale
        x = self._snake(x, gen.snake_post.alpha, self.gen_snake_post_inv)
        x = self._conv1d(x, gen.conv_post)
        return torch.tanh(x)

    def _upsample(self, audio):
        # Polyphase windowed-sinc interpolation for an integer ratio; clean linear fallback. The
        # int16 -> [-1, 1] scale (1 / 32768) is fused into the resample kernel so the primary
        # integer-ratio path needs no separate divide; the fallbacks apply it explicitly.
        if self.resample_kernel is not None:
            return F.conv_transpose1d(audio, self.resample_kernel, stride=self.resample_L, padding=self.resample_pad, output_padding=self.resample_output_padding)
        audio = audio * self.input_scale
        if self.scale_factor != 1.0:
            return F.interpolate(audio, scale_factor=self.scale_factor, mode="linear", align_corners=True, recompute_scale_factor=False)
        return audio

    def forward(self, audio):
        # The clearvoice pipeline reads int16 PCM as float / MAX_WAV_VALUE (32768) and runs the
        # whole network on the [-1, 1] waveform. That int16 -> [-1, 1] scale is fused into the
        # resample kernel (see _upsample / __init__), so no standalone divide op is emitted. The
        # final waveform is scaled back by 32768 and clamped, so the model returns int16 samples
        # directly (see the tail of this method).
        audio = self._upsample(audio.float())
        # HiFi-GAN mel: explicit reflect pad of (n_fft - hop) / 2 then a center=False STFT.
        mp = self.mel_pad
        audio_mel = torch.cat([audio[..., 1:mp + 1].flip(2), audio, audio[..., -(mp + 1):-1].flip(2)], dim=2)
        real_part, imag_part = self.pre_stft(audio_mel)
        mel_features = torch.matmul(self.fbank, torch.sqrt(real_part * real_part + imag_part * imag_part)).clamp(min=1e-5).log()
        mel_features_len = mel_features.shape[-1] if DYNAMIC_AXES else self.static_frames
        mossformer_output = self._run_masknet(mel_features, mel_features_len)
        generated_wav = self._run_generator(mossformer_output)
        if DYNAMIC_AXES:
            # Match the input and generated-waveform lengths (clearvoice uses the shorter one).
            generated_wav = generated_wav[..., :audio.shape[-1]]
            audio = audio[..., :generated_wav.shape[-1]]
        elif self.gen_pad > 0:
            gp = self.gen_pad
            generated_wav = torch.cat([generated_wav, generated_wav[..., -(gp + 1):-1].flip(2)], dim=2)
        # Bandwidth substitution via a linear-phase complementary FIR crossover:
        #     out = lowpass(input) + highpass(generator) = lowpass(input) + (generator - lowpass(generator))
        # `audio` (pristine up-sampled input) supplies the low band; `generated_wav` supplies the
        # synthesized high band. Both are filtered by the SAME symmetric kernel, so the low/high
        # halves share an identical group delay and sum coherently with perfect reconstruction and
        # no spectral seam. Reflect padding (slice+flip, ONNX-friendly) removes convolution edge
        # transients; the per-window Hann overlap-add on the host smooths any residual boundary.
        c = self.xover_half
        audio_padded = torch.cat([audio[..., 1:c + 1].flip(2), audio, audio[..., -(c + 1):-1].flip(2)], dim=2)
        gen_padded = torch.cat([generated_wav[..., 1:c + 1].flip(2), generated_wav, generated_wav[..., -(c + 1):-1].flip(2)], dim=2)
        both_low = F.conv1d(torch.cat((audio_padded, gen_padded), dim=0), self.xover_lp)
        audio_low, gen_low = torch.split(both_low, 1, dim=0)
        wav_sub = generated_wav - gen_low + audio_low
        wav_sub = wav_sub[..., :audio.shape[-1] if DYNAMIC_AXES else self.static_audio_len]
        # Scale the [-1, 1] waveform to 16-bit PCM first. Integer output returns that PCM tensor;
        # floating output divides it back to [-1, 1] inside the graph, matching the shared dtype
        # contract without any host-side scaling.
        pcm = torch.clamp(wav_sub * 32767.0, -32768.0, 32767.0)
        if "int" in OUT_AUDIO_DTYPE.lower():
            return pcm.to(torch.int16)
        pcm = pcm * self.inv_int16
        if "32" in OUT_AUDIO_DTYPE:
            return pcm
        return pcm.to(torch.float16)





def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_MossFormer_SR_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    pre_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=False, pad_mode='reflect').eval()

    mossformer = load_mossformer2_sr_model(model_path).eval().float().to("cpu")
    mossformer = MOSSFORMER_SR(mossformer, pre_stft, NFFT, N_MELS, ORIGINAL_SAMPLE_RATE, SUPER_SAMPLE_RATE, input_audio_len=INPUT_AUDIO_LENGTH)
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    torch.onnx.export(
        mossformer,
        (audio,),
        onnx_model_A,
        input_names=['original_audio'],
        output_names=['super_resolution_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'original_audio': {2: 'audio_len'},
            'super_resolution_audio': {2: 'super_audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17,
        dynamo=False
    )
    del mossformer
    del audio
    del pre_stft
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="MossFormer2_SR", task="super_resolution", model_family="mossformer2_sr",
    max_dynamic_audio_seconds=6, normalize_audio_default=False,
    input_channels=1, output_channels=1, num_audio_inputs=1, feature_kind="mel_spectrogram_stft_super_resolution",
    center_pad=False, pad_mode="reflect",
    extra={
        "original_sample_rate": ORIGINAL_SAMPLE_RATE, "super_sample_rate": SUPER_SAMPLE_RATE,
        "scale_factor": float(SUPER_SAMPLE_RATE / ORIGINAL_SAMPLE_RATE), "n_mels": N_MELS,
        "nfft_post": NFFT_POST, "window_length_post": WINDOW_LENGTH_POST, "hop_length_post": HOP_LENGTH_POST,
    },
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
