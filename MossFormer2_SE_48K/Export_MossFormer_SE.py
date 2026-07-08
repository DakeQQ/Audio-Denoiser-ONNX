import argparse
import gc
import subprocess
import sys
import os
from pathlib import Path
from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
import numpy as np
import torch
import torch.nn.functional as F
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata

parent_path = Path(__file__).resolve().parent

model_path   = "/home/DakeQQ/Downloads/MossFormer2_SE_48K"                         # The MossFormer2_SE_48K download folder.
onnx_model_A = str(parent_path / "MossFormer_ONNX" / "MossFormer2_SE_48K.onnx")  # The exported onnx model path.
onnx_model_Metadata = str(metadata_path_for_model(onnx_model_A))                  # The metadata carrier onnx model path.


# ---- Model / audio settings ----
DYNAMIC_AXES       = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET              = 18
MODEL_SAMPLE_RATE  = 48000                   # MossFormer2_SE_48K runs at 48kHz internally.
IN_SAMPLE_RATE     = 48000                   # [8000, 16000, 22500, 24000, 44000, 48000]; input audio sample rate.
OUT_SAMPLE_RATE    = 48000                   # [8000, 16000, 22500, 24000, 44000, 48000]; output audio sample rate.
INPUT_AUDIO_LENGTH = 96000                   # Maximum input audio length in IN_SAMPLE_RATE samples. Higher values yield better quality but time consume. It is better to set an integer multiple of the NFFT value.
IN_AUDIO_DTYPE     = 'INT16'                 # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE    = 'INT16'                 # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16          = float(1.0 / 32768.0)
WINDOW_TYPE        = 'hamming'               # Type of window function used in the STFT
N_MELS             = 60                      # Number of Mel bands to generate in the Mel-spectrogram
NFFT               = 1920                    # Number of FFT components for the STFT process
WINDOW_LENGTH      = 1920                    # Length of windowing, edit it carefully.
HOP_LENGTH         = 384                     # Number of samples between successive frames in the STFT

# ---- Derived constants (computed from the settings above) ----
INPUT_TO_MODEL_SCALE  = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_AUDIO_LENGTH    = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
OUTPUT_AUDIO_LENGTH   = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
BATCH_WINDOW_SECONDS  = 1.5                 # When the configured input length is >= this many seconds, fold into fixed-length windows and batch-process them together (each window runs the full network independently, i.e. per-window attention).
USE_BATCH_FOLD        = False                # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length, rounded UP to a HOP multiple. center=False snip-edges needs (W-WINDOW_LENGTH)%HOP==0; holds since WINDOW_LENGTH and W are both HOP multiples.
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop.
MAX_SIGNAL_LENGTH     = 4096 if DYNAMIC_AXES else (((FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else MODEL_AUDIO_LENGTH) - WINDOW_LENGTH) // HOP_LENGTH + 1)  # Max frames after centerless STFT/fbank framing (per-window count in fold mode). Use a larger value for dynamic axes, such as 4096.


def load_mossformer2_model(checkpoint_dir):
    model = MossFormer2_SE_48K(argparse.Namespace(mode='inference')).model
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

class MOSSFORMER_SE(torch.nn.Module):
    def __init__(self, mossformer_se, stft_model, istft_model, nfft_stft, n_mels, in_sample_rate, out_sample_rate, max_signal_len, use_batch_fold=False, fold_window=0):
        super(MOSSFORMER_SE, self).__init__()
        self.mossformer_se = mossformer_se.mossformer
        self.istft_model = istft_model
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        # ---- Faithful Kaldi-fbank feature extractor --------------------------------------
        # The original clearvoice pipeline (clearvoice/utils/decode.py) computes input
        # features with torchaudio.compliance.kaldi.fbank, i.e. per-frame DC-offset
        # removal, 0.97 preemphasis, a symmetric Hamming window, an FFT size rounded up to
        # the next power of two (2048 for a 1920-sample frame), the power spectrum, a Kaldi
        # mel filterbank and a log. Every per-frame step is linear, so DC removal +
        # preemphasis + window + 2048-pt real-DFT are folded into a single Conv1d kernel
        # applied with stride = hop and no centre padding (= Kaldi snip_edges).
        self.feat_win = WINDOW_LENGTH
        self.feat_hop = HOP_LENGTH
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.n_mels = n_mels
        padded = 2 ** (self.feat_win - 1).bit_length()                 # next power of two -> 2048
        self.feat_fbins = padded // 2 + 1                              # 1025 one-sided bins
        fbank_kernel = self._build_kaldi_kernel(self.feat_win, padded, 0.97)
        stft_kernel = stft_model.stft_kernel.detach().float()
        if getattr(stft_model, '_center_pad', False) or stft_model.hop_len != self.feat_hop or stft_kernel.shape[-1] != fbank_kernel.shape[-1]:
            raise ValueError('The fused frontend requires center_pad=False and matching STFT/fbank stride and kernel width.')
        self.stft_fbins = stft_model.half_n_fft + 1
        self.register_buffer('frontend_kernel', torch.cat((fbank_kernel, stft_kernel), dim=0).contiguous())
        self.register_buffer('mel_banks', self._kaldi_mel_banks(n_mels, padded, 48000.0, 20.0, 0.0))
        self.log_eps = float(torch.finfo(torch.float32).eps)

        t = torch.arange(max_signal_len * 3, dtype=torch.float32)  # Create time steps
        sinu = t.unsqueeze(-1) * self.mossformer_se.pos_enc.inv_freq  # Calculate sine and cosine embeddings
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)  # Concatenate sine and cosine embeddings
        emb_pos = (emb * self.mossformer_se.pos_enc.scale).transpose(0, -1)  # Scale the embeddings
        # Stored as a float16 buffer to halve its initializer footprint. The forward pass
        # slices the few frames it needs first and casts only that small slice back to
        # float32, so the memory saving costs just one Cast over a tiny tensor.
        self.register_buffer('emb_pos', emb_pos.unsqueeze(0).half().contiguous())

        self.win_length_for_delta = 5
        self.n = (self.win_length_for_delta - 1) // 2
        denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3
        self.inv_denom = float(1.0 / denom)
        delta_kernel = torch.arange(-self.n, self.n + 1, 1, dtype=torch.float32).view(1, 1, -1) * self.inv_denom
        self.register_buffer('delta_weight', delta_kernel.contiguous())

        # Pre-allocated zero padding templates for the concat-based padding. Concatenating a
        # sliced zero buffer is markedly faster than F.pad in ONNX Runtime, so the templates
        # are built once here (dims are read from the live modules to keep the script
        # standalone) and sliced to the required width inside the forward pass.
        flash0 = self.mossformer_se.mdl.intra_mdl.mossformerM.layers[0]
        group_size = flash0.group_size
        qk_dim = flash0.qk_offset_scale.gamma.shape[-1]
        vu_dim = flash0.to_hidden.mdl[1].weight.shape[0] // 2
        model_dim = flash0.to_hidden.mdl[1].weight.shape[1]
        self.model_dim = model_dim
        self.register_buffer('pad_A4', torch.zeros((1, group_size, 4, qk_dim), dtype=torch.float16))
        self.register_buffer('pad_VU', torch.zeros((1, group_size, vu_dim * 2), dtype=torch.float16))
        self.register_buffer('shift_pad', torch.zeros((1, 1, model_dim // 2), dtype=torch.float32))
        uf0 = self.mossformer_se.mdl.intra_mdl.mossformerM.fsmn[0].gated_fsmn.fsmn
        self.register_buffer('fsmn_pad', torch.zeros((1, uf0.output_dim, uf0.lorder - 1), dtype=torch.float32))

        # Rotary cos/sin tables, shared by every FLASH layer (they depend only on the frame
        # position, not on the data). Precomputed once here for the maximum possible frame
        # count and sliced per call, exactly like emb_pos above. This keeps the Range / Mul /
        # Stack / Cos / Sin out of the exported graph (they become constant initializers) and
        # avoids recomputing the tables on every forward call. Stored as float16 to halve the
        # initializer footprint; the forward slices the live frames first and casts only that
        # small slice back to float32.
        rot_freqs = flash0.rotary_pos_emb.freqs
        self.rot_dim = int(2 * rot_freqs.shape[0])              # literal channel count rotated (e.g. 32)
        rot_ang = torch.arange(max_signal_len * 3, dtype=rot_freqs.dtype).unsqueeze(-1) * rot_freqs
        rot_ang = torch.stack((rot_ang, rot_ang), dim=-1).flatten(-2)
        self.register_buffer('rot_cos', rot_ang.cos().half().unsqueeze(0).unsqueeze(2).contiguous())
        self.register_buffer('rot_sin', rot_ang.sin().half().unsqueeze(0).unsqueeze(2).contiguous())

        # ------------------------------------------------------------------
        # Pre-fused projection weights (built once in float64, cast to float32).
        # Every FLASH layer feeds the SAME ScaleNorm-normalised input to to_hidden
        # (-> v, u) and to_qk (-> q, k); both share one denominator (identical dim/eps),
        # so the two projections collapse into ONE Linear + ONE depthwise Conv with the
        # scalar ScaleNorm gains folded into the weights. The to_out gain is folded too.
        # ------------------------------------------------------------------
        flash_layers = self.mossformer_se.mdl.intra_mdl.mossformerM.layers
        dw_kernel = flash0.to_hidden.mdl[3].sequential[1].conv.weight.shape[-1]
        self.dw_pad = (dw_kernel - 1) // 2
        self.fl_vu = vu_dim
        self.fl_vu2 = vu_dim * 2
        self.fl_qk = qk_dim
        self.fl_in_groups = self.fl_vu2 + qk_dim
        self.flash_group_size = group_size
        self.fl_inv_g = float(1.0 / group_size)
        self.fl_scale = float(flash0.to_hidden.mdl[0].scale)
        self.fl_eps = float(flash0.to_hidden.mdl[0].eps)
        self.fl_out_scale = float(flash0.to_out.mdl[0].scale)
        self.fl_out_eps = float(flash0.to_out.mdl[0].eps)
        self.fl_norm_eps = float(self.fl_eps / self.fl_scale)
        self.fl_out_norm_eps = float(self.fl_out_eps / self.fl_out_scale)
        self.fl_in_scale_fold = float(1.0 / self.fl_scale)
        self.fl_out_scale_fold = float(1.0 / self.fl_out_scale)
        self.fold_lin_inv_n = not DYNAMIC_AXES
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
            qk_scale[0, 0] = self.fl_inv_g
            if self.fold_lin_inv_n:
                qk_scale[3, 0] = 1.0 / float(max_signal_len)
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

        # FSMN gates: to_u and to_v share one (affine-free) LayerNorm; the LayerNorm affine
        # is folded into each Linear so both run as one fused Linear + depthwise Conv.
        fsmn_blocks = self.mossformer_se.mdl.intra_mdl.mossformerM.fsmn
        gf0 = fsmn_blocks[0].gated_fsmn
        self.fs_inner = gf0.to_u.mdl[1].weight.shape[0]
        self.fs_uv_groups = self.fs_inner * 2
        self.fs_ln_shape = tuple(gf0.to_u.mdl[0].normalized_shape)
        self.fs_ln_eps = float(gf0.to_u.mdl[0].eps)
        self._fs_uv_w, self._fs_uv_b, self._fs_uv_c, self._fs_mem_c = [], [], [], []
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
            self.register_buffer(f'fs_mem_c_{i}', gf.fsmn.conv1.weight.detach().squeeze(-1).contiguous())
            self._fs_uv_w.append(getattr(self, f'fs_uv_w_{i}'))
            self._fs_uv_b.append(getattr(self, f'fs_uv_b_{i}'))
            self._fs_uv_c.append(getattr(self, f'fs_uv_c_{i}'))
            self._fs_mem_c.append(getattr(self, f'fs_mem_c_{i}'))

        # The exported wrapper ultimately uses only speaker 0. Fold the first-speaker
        # conv1d_out rows into the shared output/output_gate 1x1 convolutions, then run the
        # two gate projections as one fused Conv1d. This removes the second speaker branch,
        # two reshapes, and one whole linear 1x1 stage from the traced tail.
        tail_channels = self.mossformer_se.conv1_decoder.in_channels
        self.tail_channels = tail_channels
        spk_w = self.mossformer_se.conv1d_out.weight.detach()[:tail_channels, :, 0].double()
        spk_b = self.mossformer_se.conv1d_out.bias.detach()[:tail_channels].double()
        gate_w = torch.cat((self.mossformer_se.output[0].weight.detach(),
                            self.mossformer_se.output_gate[0].weight.detach()), dim=0).squeeze(-1).double()
        gate_b = torch.cat((self.mossformer_se.output[0].bias.detach(),
                            self.mossformer_se.output_gate[0].bias.detach()), dim=0).double()
        self.register_buffer('tail_gate_w', (gate_w @ spk_w).float().unsqueeze(-1).contiguous())
        self.register_buffer('tail_gate_b', (gate_w @ spk_b + gate_b).float().contiguous())

        self.static_frames = max_signal_len

    def _build_kaldi_kernel(self, win_len, padded, preemph):
        """Fold per-frame DC removal, preemphasis, the symmetric Hamming window and the
        zero-padded real-DFT into one Conv1d weight of shape (2*(padded//2+1), 1, win_len).
        Row block 0 -> real part, block 1 -> imaginary part."""
        f_bins = padded // 2 + 1
        w = torch.hamming_window(win_len, periodic=False, alpha=0.54, beta=0.46, dtype=torch.float64)
        t = torch.arange(win_len, dtype=torch.float64)
        f = torch.arange(f_bins, dtype=torch.float64).unsqueeze(1)
        omega = (2.0 * torch.pi / padded) * f * t.unsqueeze(0)
        cos_rows = torch.cos(omega) * w.unsqueeze(0)            # real DFT rows  (f_bins, win_len)
        sin_rows = -torch.sin(omega) * w.unsqueeze(0)           # imag DFT rows
        # preemphasis P: s[j] -= preemph * s[max(0, j-1)]  (Kaldi replicates the first sample)
        shift = torch.zeros(win_len, win_len, dtype=torch.float64)
        shift[0, 0] = 1.0
        ar = torch.arange(1, win_len)
        shift[ar, ar - 1] = 1.0
        P = torch.eye(win_len, dtype=torch.float64) - preemph * shift
        # per-frame DC-offset removal M = I - 1/N (subtract the frame mean)
        M = torch.eye(win_len, dtype=torch.float64) - (1.0 / win_len)
        PM = P @ M
        kernel = torch.cat([cos_rows @ PM, sin_rows @ PM], dim=0).unsqueeze(1)   # (2*f_bins, 1, win_len)
        return kernel.float()

    def _kaldi_mel_banks(self, num_bins, padded, sample_freq, low_freq, high_freq):
        """Reproduce torchaudio.compliance.kaldi.get_mel_banks (VTLN off) padded to the rfft
        bin count, returned as a (1, num_bins, padded//2+1) matmul matrix."""
        num_fft_bins = padded // 2
        nyquist = 0.5 * sample_freq
        if high_freq <= 0.0:
            high_freq = high_freq + nyquist
        fft_bin_width = sample_freq / padded
        mel_low = 1127.0 * float(np.log(1.0 + low_freq / 700.0))
        mel_high = 1127.0 * float(np.log(1.0 + high_freq / 700.0))
        mel_delta = (mel_high - mel_low) / (num_bins + 1)
        b = torch.arange(num_bins, dtype=torch.float64).unsqueeze(1)
        left_mel = mel_low + b * mel_delta
        center_mel = mel_low + (b + 1.0) * mel_delta
        right_mel = mel_low + (b + 2.0) * mel_delta
        mel = (1127.0 * torch.log(1.0 + (fft_bin_width * torch.arange(num_fft_bins, dtype=torch.float64)) / 700.0)).unsqueeze(0)
        up = (mel - left_mel) / (center_mel - left_mel)
        down = (right_mel - mel) / (right_mel - center_mel)
        bins = torch.clamp(torch.minimum(up, down), min=0.0)
        bins = torch.nn.functional.pad(bins, (0, 1), mode='constant', value=0.0)   # (num_bins, num_fft_bins+1)
        return bins.unsqueeze(0).float()

    def compute_deltas(self, specgram: torch.Tensor, mode: str = "replicate", time_dim: int = 256) -> torch.Tensor:
        bsz = specgram.shape[0]                          # batch (num_window under fold); reused for the reshape back
        specgram = specgram.reshape(-1, 1, time_dim)
        padded_specgram = torch.nn.functional.pad(specgram, (self.n, self.n), mode=mode)
        output = F.conv1d(padded_specgram, self.delta_weight)
        output = output.reshape(bsz, self.n_mels, time_dim)
        return output

    def forward(self, audio):
        audio = audio.float()
        if "int" not in IN_AUDIO_DTYPE.lower():
            audio = audio * 32768.0      # F16/F32 inputs arrive in [-1, 1]; lift them to the int16 amplitude the Kaldi fbank expects.
        if self.in_sample_rate != MODEL_SAMPLE_RATE:
            audio = torch.nn.functional.interpolate(
                audio,
                size=MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else None,
                scale_factor=None if not DYNAMIC_AXES else INPUT_TO_MODEL_SCALE,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            # Input length is already a whole number of windows (the tail was padded OUTSIDE
            # (num_window, 1, W) and run the whole network batched. Each window is processed
            # independently, so attention spans one window (the accepted fold tradeoff).
            audio = audio.reshape(-1, 1, self.fold_window)
        # ---- Kaldi log-mel filterbank features (snip-edges framing) ----
        # One Conv1d computes both frontends: the Kaldi fbank DFT rows and the analysis
        # STFT rows. The kernels are only concatenated for speed; the two branches remain
        # mathematically separate so the trained Kaldi feature frontend is preserved.
        frontend = torch.nn.functional.conv1d(audio, self.frontend_kernel, stride=self.feat_hop)
        kaldi_frames, stft_frames = torch.split(frontend, [self.feat_fbins * 2, self.stft_fbins * 2], dim=1)
        kaldi_frames = kaldi_frames * kaldi_frames                                               # square the real & imag Kaldi DFT rows
        kaldi_real_power, kaldi_imag_power = torch.split(kaldi_frames, self.feat_fbins, dim=1)   # (1, f_bins, m) squared parts
        power = kaldi_real_power + kaldi_imag_power                                              # power spectrum
        mel_features = torch.matmul(self.mel_banks, power).clamp(min=self.log_eps).log()         # (1, n_mels, m)
        # Reuse the known (static) frame count instead of reading it back from the tensor
        # shape; this keeps the rest of the pipeline free of dynamic Shape/Gather/Range ops.
        mel_features_len = mel_features.shape[-1].unsqueeze(0) if DYNAMIC_AXES else self.static_frames
        mel_features_delta = self.compute_deltas(mel_features, time_dim=mel_features_len)
        mel_features_delta_2 = self.compute_deltas(mel_features_delta, time_dim=mel_features_len)
        mel_features = torch.cat([mel_features, mel_features_delta, mel_features_delta_2], dim=1)
        real_part, imag_part = torch.split(stft_frames, self.stft_fbins, dim=1)
        x = self.mossformer_se.norm(mel_features)
        x = self.mossformer_se.conv1d_encoder(x)
        x = x + self.emb_pos[..., :mel_features_len].float()

        # ---- Inlined self.mossformer_se.mdl(x, mel_features_len) ----
        # Computation_Block -> MossFormerM -> MossformerBlock_GFSMN, expanded so the
        # 24 x [FLASH_ShareA_FFConvM + Gated_FSMN_Block] run as leaf torch ops. Each FLASH
        # layer fuses to_hidden||to_qk into one Linear + one depthwise Conv (shared
        # ScaleNorm); each FSMN gate fuses to_u||to_v likewise (shared LayerNorm). The
        # pre-folded weights live in the buffers built in __init__.
        gfsmn = self.mossformer_se.mdl.intra_mdl.mossformerM
        flash_layers = gfsmn.layers
        fsmn_blocks = gfsmn.fsmn
        mm_norm = self.mossformer_se.mdl.intra_mdl.norm
        intra_norm = self.mossformer_se.mdl.intra_norm

        mdl_input = x                                    # [B, 512, n]
        h = x.permute(0, 2, 1).contiguous()              # [B, n, 512]
        bsz = h.shape[0]                                 # batch = num_window under fold (1 otherwise)
        inv_n = 1.0 / mel_features_len

        # Rotary cos/sin tables sliced to the frame count and broadcast over the 4 heads
        rot_dim = self.rot_dim
        rcos = self.rot_cos[:, :mel_features_len].float()   # [1, n, 1, rot_dim]
        rsin = self.rot_sin[:, :mel_features_len].float()

        # Group padding to a multiple of group_size (loop-invariant; static when not dynamic)
        group_size = self.flash_group_size
        remainder = mel_features_len % group_size
        padding = 0 if remainder == 0 else group_size - remainder
        padded_len = mel_features_len + padding

        pad_A4 = self.pad_A4[:, :padding].float()
        pad_VU = self.pad_VU[:, :padding].float()

        for i in range(len(flash_layers)):
            # ===== FLASH_ShareA_FFConvM (fused to_hidden || to_qk) =====
            residual = h
            x_shift, x_pass = h.chunk(2, dim=-1)
            x_shift = torch.cat((self.shift_pad.expand(bsz, -1, -1), x_shift[:, :-1, :]), dim=1)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

            base = normed_x / (torch.norm(normed_x, dim=-1, keepdim=True) + self.fl_norm_eps)
            proj = F.silu(F.linear(base, self._fl_in_w[i], self._fl_in_b[i]))
            cw = self._fl_in_c[i]
            proj = proj + F.conv1d(proj.transpose(1, 2), cw, None, padding=self.dw_pad, groups=self.fl_in_groups).transpose(1, 2)
            v, u, qk = torch.split(proj, [self.fl_vu, self.fl_vu, self.fl_qk], dim=-1)
            value_proj = proj[..., :self.fl_vu2]

            # OffsetScale (4 heads) kept stacked -> one rotary + one padding for all heads
            scaled = qk.unsqueeze(-2) * self._qkos_gamma[i] + self._qkos_beta[i]  # [1, n, 4, qk]
            mid = scaled[..., :rot_dim]
            half = torch.stack((-mid[..., 1::2], mid[..., 0::2]), dim=-1).flatten(-2)
            scaled = torch.cat((mid * rcos + half * rsin, scaled[..., rot_dim:]), dim=-1)
            if padding > 0:
                scaled = torch.cat((scaled, pad_A4.expand(bsz, -1, -1, -1)), dim=1)  # [B, padded_len, 4, qk]
            scaled = scaled.reshape(bsz, -1, group_size, 4, self.fl_qk)
            quad_q, lin_q, quad_k, lin_k = scaled.split(1, dim=3)
            quad_q = quad_q.squeeze(3)
            lin_q = lin_q.squeeze(3)
            quad_k = quad_k.squeeze(3)
            lin_k = lin_k.squeeze(3)
            if padding > 0:
                vug = torch.cat((value_proj, pad_VU.expand(bsz, -1, -1)), dim=1).reshape(bsz, -1, group_size, self.fl_vu2)
            else:
                vug = value_proj.reshape(bsz, -1, group_size, self.fl_vu2)

            # Quadratic attention (padded keys are exact zeros -> no extra masking required)
            attn = F.relu(torch.matmul(quad_q, quad_k.transpose(-1, -2)))
            quad_out = torch.matmul(attn.square(), vug)

            # Linear attention
            lin_k_flat = lin_k.permute(0, 3, 1, 2).reshape(bsz, 1, self.fl_qk, padded_len)
            lin_kvu = torch.matmul(lin_k_flat, vug.reshape(bsz, 1, padded_len, self.fl_vu2))
            if not self.fold_lin_inv_n:
                lin_kvu = lin_kvu * inv_n
            lin_out = torch.matmul(lin_q, lin_kvu)

            att_vu = (quad_out + lin_out).reshape(bsz, padded_len, self.fl_vu2)[:, :mel_features_len, :]
            att_v, att_u = torch.split(att_vu, [self.fl_vu, self.fl_vu], dim=-1)
            out = (att_u * v) * torch.sigmoid(att_v * u)

            # to_out: ScaleNorm (gain folded into weight) -> Linear -> SiLU -> ConvModule
            y = out / (torch.norm(out, dim=-1, keepdim=True) + self.fl_out_norm_eps)
            y = F.silu(F.linear(y, self._fl_out_w[i], self._fl_out_b[i]))
            y = y + F.conv1d(y.transpose(1, 2), self._fl_out_c[i], None, padding=self.dw_pad, groups=self.model_dim).transpose(1, 2)
            h = residual + y

            # ===== Gated_FSMN_Block (fused to_u || to_v) =====
            gblk = fsmn_blocks[i]
            blk_in = h
            c1 = gblk.conv1[0]
            c1y = F.prelu(F.conv1d(blk_in.transpose(1, 2), c1.weight, c1.bias), gblk.conv1[1].weight)
            n1 = gblk.norm1
            gf_in = F.layer_norm(c1y.transpose(1, 2), n1.normalized_shape, n1.weight, n1.bias, n1.eps)

            gf = gblk.gated_fsmn
            xn = F.layer_norm(gf_in, self.fs_ln_shape, None, None, self.fs_ln_eps)
            proj = F.silu(F.linear(xn, self._fs_uv_w[i], self._fs_uv_b[i]))
            cw = self._fs_uv_c[i]
            proj = proj + F.conv1d(proj.transpose(1, 2), cw, None, padding=self.dw_pad, groups=self.fs_uv_groups).transpose(1, 2)
            xu, xv = torch.split(proj, [self.fs_inner, self.fs_inner], dim=-1)

            # UniDeepFsmn memory branch on xu
            uf = gf.fsmn
            f1 = F.relu(F.linear(xu, uf.linear.weight, uf.linear.bias))
            xp = F.linear(f1, uf.project.weight, None).transpose(1, 2)
            yy = torch.cat((self.fsmn_pad.expand(bsz, -1, -1), xp, self.fsmn_pad.expand(bsz, -1, -1)), dim=2)
            yy = F.conv1d(yy, self._fs_mem_c[i], None, groups=uf.conv1.groups)
            xu = xu + (xp + yy).transpose(1, 2)

            y = xv * xu + gf_in

            n2 = gblk.norm2
            norm2_out = F.layer_norm(y, n2.normalized_shape, n2.weight, n2.bias, n2.eps).transpose(1, 2)
            c2 = gblk.conv2
            h = F.conv1d(norm2_out, c2.weight, c2.bias).transpose(1, 2) + blk_in

        # MossFormerM final LayerNorm
        h = F.layer_norm(h, mm_norm.normalized_shape, mm_norm.weight, mm_norm.bias, mm_norm.eps)
        # Computation_Block: back to [B, 512, S], intra normalization, skip connection
        h = h.permute(0, 2, 1).contiguous()
        h = F.group_norm(h, intra_norm.num_groups, intra_norm.weight, intra_norm.bias, intra_norm.eps)
        x = h + mdl_input
        # ---- end inlined mdl ----

        x = F.prelu(x, self.mossformer_se.prelu.weight)
        gate_pair = F.conv1d(x, self.tail_gate_w, self.tail_gate_b)
        x_out, x_gate = torch.split(gate_pair, [self.tail_channels, self.tail_channels], dim=1)
        x = torch.tanh(x_out) * torch.sigmoid(x_gate)
        x = F.relu(F.conv1d(x, self.mossformer_se.conv1_decoder.weight, None))
        real_part *= x
        imag_part *= x
        audio = self.istft_model(real_part, imag_part)
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
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=False, pad_mode='constant').eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=False, pad_mode='constant').eval()
    mossformer = load_mossformer2_model(model_path).eval().float().to("cpu")
    mossformer = MOSSFORMER_SE(mossformer, custom_stft, custom_istft, NFFT, N_MELS, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, MAX_SIGNAL_LENGTH, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH)
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
        do_constant_folding=True,
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'out_audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17,
        dynamo=False
    )
    del mossformer
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="MossFormer2_SE_48K", task="denoise", model_family="mossformer2_se",
    max_dynamic_audio_seconds=6, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=1, feature_kind="kaldi_fbank_stft", center_pad=False, pad_mode="constant", extra={"n_mels": N_MELS},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
