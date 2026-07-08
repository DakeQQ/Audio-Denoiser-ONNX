import gc
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


parent_path          = Path(__file__).resolve().parent                     # The folder that contains this script.
project_path         = "/home/DakeQQ/Downloads/NKF-AEC-gh-pages"            # The NKF-AEC GitHub project download path. https://github.com/jfsean/NKF-AEC
checkpoint_path      = project_path + "/src/nkf_epoch70.pt"               # The pretrained checkpoint path.
onnx_model_A         = str(parent_path / "NKF_AEC_ONNX" / "NKF_AEC.onnx") # The exported onnx model path.
onnx_model_Metadata  = str(metadata_path_for_model(onnx_model_A))          # The metadata carrier onnx model path.


DYNAMIC_AXES        = False          # Only support static axes. Do not edit.
IN_SAMPLE_RATE      = 16000          # The NKF-AEC model runs internally at 16 kHz and resamples the input when needed.
OUT_SAMPLE_RATE     = 16000          # Output sample rate after the internal 16 kHz Kalman-filter model finishes.
INPUT_AUDIO_LENGTH  = 32000          # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 32000.
WINDOW_TYPE         = 'hann'         # Type of window function used in the STFT
NFFT                = 1024           # Number of FFT components for the STFT process
WINDOW_LENGTH       = 1024           # Length of windowing, edit it carefully.
HOP_LENGTH          = 256            # Number of samples between successive frames in the STFT
MAX_SIGNAL_LENGTH   = 256            # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.

# Batch-fold constants
USE_BATCH_FOLD      = False          # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
MODEL_SAMPLE_RATE   = 16000          # NKF runs internally at 16 kHz.
BATCH_WINDOW_SECONDS = 1.5           # Minimum input length (seconds) that triggers window folding.
FOLD_WINDOW_LENGTH  = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length rounded UP to HOP_LENGTH.
EXPORT_AUDIO_LENGTH = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH
if USE_BATCH_FOLD:
    # In fold mode NKF sees one window per batch row. The ISTFT (center_pad=True) derives its
    # output length statically from max_frames, so for NKF's even NFFT (1024, half=512) an
    # oversized max_frames would make each window emit half_n_fft extra samples (shape_out !=
    # shape_in). max_frames must equal the exact per-window frame count = W // HOP + 1.
    MAX_SIGNAL_LENGTH = FOLD_WINDOW_LENGTH // HOP_LENGTH + 1

# NKF model parameters
FILTER_ORDER        = 4              # Kalman filter order (L), do not edit the value.
FC_DIM              = 18             # Fully-connected layer dimension, do not edit the value.
RNN_LAYERS          = 1              # Number of GRU layers, do not edit the value.
RNN_DIM             = 18             # GRU hidden dimension, do not edit the value.

IN_AUDIO_DTYPE      = 'INT16'        # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE     = 'INT16'        # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16           = float(1.0 / 32768.0)
OPSET               = 18             # ONNX opset.


class ComplexGRU_Real(nn.Module):
    """ComplexGRU decomposed into real/imaginary parts for ONNX export."""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True):
        super().__init__()
        self.gru_r = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first)
        self.gru_i = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x_real, x_imag, h_rr, h_ir, h_ri, h_ii):
        Frr, h_rr = self.gru_r(x_real, h_rr)
        Fir, h_ir = self.gru_r(x_imag, h_ir)
        Fri, h_ri = self.gru_i(x_real, h_ri)
        Fii, h_ii = self.gru_i(x_imag, h_ii)
        # complex(Frr - Fii, Fri + Fir)
        out_real = Frr - Fii
        out_imag = Fri + Fir
        return out_real, out_imag, h_rr, h_ir, h_ri, h_ii


class ComplexDense_Real(nn.Module):
    """ComplexDense decomposed into real/imaginary parts for ONNX export.
    Fused: batched matmul via stacked weights (2, out, in) applied to stacked input."""
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.linear_real = nn.Linear(in_channel, out_channel, bias=bias)
        self.linear_imag = nn.Linear(in_channel, out_channel, bias=bias)
        self.out_channel = out_channel
        self._fused = False

    def fuse_weights_(self):
        """Stack real/imag weights for batched matmul: (2, in, out) pre-transposed."""
        with torch.no_grad():
            # stacked_weight: (2, in, out) — pre-transposed for bmm
            self.register_buffer('stacked_weight', torch.stack([self.linear_real.weight, self.linear_imag.weight], dim=0).transpose(1, 2).contiguous())
            if self.linear_real.bias is not None:
                # stacked_bias: (2, 1, out)
                self.register_buffer('stacked_bias', torch.stack([self.linear_real.bias, self.linear_imag.bias], dim=0).unsqueeze(1))
            else:
                self.register_buffer('stacked_bias', None)
        self._fused = True

    def forward(self, x_real, x_imag):
        if self._fused and x_real.shape[0] == 1 and x_imag.shape[0] == 1:
            # x_real, x_imag: (1, F, in) -> cat: (2, F, in)
            x_stacked = torch.cat([x_real, x_imag], dim=0)
            # bmm: (2, F, in) @ (2, in, out) -> (2, F, out)
            out = torch.bmm(x_stacked, self.stacked_weight)
            if self.stacked_bias is not None:
                out = out + self.stacked_bias
            return out.split(1, dim=0)
        return self.linear_real(x_real), self.linear_imag(x_imag)


class ComplexPReLU_Real(nn.Module):
    """ComplexPReLU decomposed into real/imaginary parts for ONNX export."""
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU()
        self.register_buffer('cached_weight', torch.ones(1, dtype=torch.float32), persistent=False)
        self.use_cached_weight = False

    def cache_weight_(self):
        with torch.no_grad():
            self.cached_weight.copy_(self.prelu.weight.detach())
        self.use_cached_weight = True

    def forward(self, x_real, x_imag):
        weight = self.cached_weight if self.use_cached_weight else self.prelu.weight
        return torch.nn.functional.prelu(x_real, weight), torch.nn.functional.prelu(x_imag, weight)


class KGNet_Real(nn.Module):
    """KGNet decomposed into real/imaginary parts for ONNX export."""
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim
        self.n_freq = NFFT // 2 + 1

        # fc_in: ComplexDense(2*L+1, fc_dim) + ComplexPReLU
        self.fc_in_dense = ComplexDense_Real(2 * L + 1, fc_dim, bias=True)
        self.fc_in_act = ComplexPReLU_Real()

        # complex_gru
        self.complex_gru = ComplexGRU_Real(fc_dim, rnn_dim, rnn_layers)

        # fc_out: ComplexDense(rnn_dim, fc_dim) + PReLU + ComplexDense(fc_dim, L)
        self.fc_out_dense1 = ComplexDense_Real(rnn_dim, fc_dim, bias=True)
        self.fc_out_act = ComplexPReLU_Real()
        self.fc_out_dense2 = ComplexDense_Real(fc_dim, L, bias=True)

    def cache_prelu_weights_(self):
        self.fc_in_act.cache_weight_()
        self.fc_out_act.cache_weight_()

    def fuse_dense_weights_(self):
        """Fuse ComplexDense weights for batched matmul."""
        self.fc_in_dense.fuse_weights_()
        self.fc_out_dense1.fuse_weights_()
        self.fc_out_dense2.fuse_weights_()

    def forward(self, input_real, input_imag, h_rr, h_ir, h_ri, h_ii):
        # input_real, input_imag: (1, F, 2*L+1)
        b = input_real.shape[0]
        x_real, x_imag = self.fc_in_dense(input_real, input_imag)
        x_real, x_imag = self.fc_in_act(x_real, x_imag)

        # GRU expects (batch=F, seq=1, fc_dim) per window: (b, F, fc_dim) -> (b*F, 1, fc_dim)
        x_real = x_real.view(b * self.n_freq, 1, -1)
        x_imag = x_imag.view(b * self.n_freq, 1, -1)

        # ComplexGRU
        x_real, x_imag, h_rr, h_ir, h_ri, h_ii = self.complex_gru(x_real, x_imag, h_rr, h_ir, h_ri, h_ii)

        # Back to (b, F, rnn_dim) from (b*F, 1, rnn_dim)
        x_real = x_real.view(b, self.n_freq, -1)
        x_imag = x_imag.view(b, self.n_freq, -1)

        # fc_out: fused dense layers, all (1, F, *)
        x_real, x_imag = self.fc_out_dense1(x_real, x_imag)
        x_real, x_imag = self.fc_out_act(x_real, x_imag)
        x_real, x_imag = self.fc_out_dense2(x_real, x_imag)

        return x_real, x_imag, h_rr, h_ir, h_ri, h_ii


class NKF(nn.Module):
    """
    NKF-AEC model fully decomposed for ONNX export:
    - No complex tensors (uses real/imag pairs)
    - Uses custom STFT_Process instead of torch.stft/istft
    - All operations are ONNX-compatible
    """
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim, custom_stft, custom_istft, max_frames, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim
        self.custom_stft = custom_stft
        self.custom_istft = custom_istft
        self.kg_net = KGNet_Real(L, fc_dim, rnn_layers, rnn_dim)
        self.n_freq = NFFT // 2 + 1  # 513 frequency bins
        self.max_frames = max_frames
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.output_pcm_scale = 32767.0
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.out_sample_rate_scale = out_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before_centering = self.in_sample_rate_scale > 1.0
        self.resample_after_centering = self.in_sample_rate_scale < 1.0
        # Output sandwich: resample DOWN before the PCM scale and UP after it so the scale
        # multiply always runs on the smaller (model-rate) tensor (mirrors the input sandwich above).
        self.output_resample_before_pcm = self.out_sample_rate_scale < 1.0   # out < model -> downsample first
        self.output_resample_after_pcm = self.out_sample_rate_scale > 1.0    # out > model -> upsample after
        self.use_batch_fold = use_batch_fold
        self.fold_window = fold_window

        self.register_buffer('h_prior_real_buffer', torch.zeros(1, self.n_freq, L, dtype=torch.float32))
        self.register_buffer('h_prior_imag_buffer', torch.zeros(1, self.n_freq, L, dtype=torch.float32))
        self.register_buffer('h_rr_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
        self.register_buffer('h_ir_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
        self.register_buffer('h_ri_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
        self.register_buffer('h_ii_buffer', torch.zeros(rnn_layers, self.n_freq, rnn_dim, dtype=torch.float32))
        self.register_buffer('zeros_pad', torch.zeros(1, self.n_freq, L - 1, dtype=torch.float32))

    def cache_export_constants_(self):
        self.kg_net.cache_prelu_weights_()
        self.kg_net.fuse_dense_weights_()

    def forward(self, far_end_audio, near_end_audio):
        """
        Args:
            far_end_audio: (1, 1, audio_len) int16, far-end reference signal
            near_end_audio: (1, 1, audio_len) int16, near-end microphone signal
        Returns:
            enhanced: (1, 1, audio_len) int16, echo-cancelled output
        """
        if self.use_batch_fold:
            far_end_audio = far_end_audio.reshape(-1, 1, self.fold_window)
            near_end_audio = near_end_audio.reshape(-1, 1, self.fold_window)
        b = far_end_audio.shape[0]
        audio_len = far_end_audio.shape[-1] if DYNAMIC_AXES else INPUT_AUDIO_LENGTH
        audio_pair = torch.cat([far_end_audio, near_end_audio], dim=0).float()
        if self.resample_before_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower():
            audio_pair = audio_pair * self.inv_int16      # int16 PCM -> [-1, 1]; F16/F32 inputs already arrive normalized.
        audio_pair = audio_pair - audio_pair.mean(dim=2, keepdim=True)
        if self.resample_after_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        pair_real, pair_imag = self.custom_stft(audio_pair)
        # ref: (1, F, T), mic: (1, F, T)
        ref_real, mic_real = pair_real.split(b, dim=0)
        ref_imag, mic_imag = pair_imag.split(b, dim=0)

        T = ref_real.shape[2]
        if T > self.max_frames:
            raise ValueError(f"Input produces {T} frames, but MAX_SIGNAL_LENGTH is {self.max_frames}.")

        # Use registered buffers directly (already zero-initialized)
        h_prior_real = self.h_prior_real_buffer.expand(b, -1, -1)
        h_prior_imag = self.h_prior_imag_buffer.expand(b, -1, -1)
        h_rr = self.h_rr_buffer.repeat(1, b, 1)
        h_ir = self.h_ir_buffer.repeat(1, b, 1)
        h_ri = self.h_ri_buffer.repeat(1, b, 1)
        h_ii = self.h_ii_buffer.repeat(1, b, 1)

        ref_real_padded = torch.cat([self.zeros_pad.expand(b, -1, -1), ref_real], dim=2)
        ref_imag_padded = torch.cat([self.zeros_pad.expand(b, -1, -1), ref_imag], dim=2)

        # Frame-by-frame Kalman filter loop
        echo_frames_real = []
        echo_frames_imag = []

        # === t=0: all h-buffers are zero, skip redundant +-*/ with zeros ===
        xt_real = ref_real_padded[..., :self.L]  # (1, F, L)
        xt_imag = ref_imag_padded[..., :self.L]  # (1, F, L)
        mic_real_t = mic_real[..., [0]]  # (1, F, 1)
        mic_imag_t = mic_imag[..., [0]]  # (1, F, 1)

        # dh=0 (h_post-h_prior=0-0), e=mic (h_prior=0 so dot product vanishes)
        # input = cat([xt, mic, zeros_L]) — reuse h_prior_real/imag as the L-wide zero block
        input_real = torch.cat([xt_real, mic_real_t, h_prior_real], dim=2)
        input_imag = torch.cat([xt_imag, mic_imag_t, h_prior_imag], dim=2)

        # KGNet forward
        kg_real, kg_imag, h_rr, h_ir, h_ri, h_ii = self.kg_net(input_real, input_imag, h_rr, h_ir, h_ri, h_ii)

        # h_post = kg * e (h_prior=0, so addition term vanishes)
        h_post_real = kg_real * mic_real_t - kg_imag * mic_imag_t
        h_post_imag = kg_real * mic_imag_t + kg_imag * mic_real_t
        # h_prior remains zero (mirrors the swap of two zero buffers in the original)

        # Echo estimate for t=0
        echo_frames_real.append((xt_real * h_post_real - xt_imag * h_post_imag).sum(dim=2, keepdim=True))
        echo_frames_imag.append((xt_real * h_post_imag + xt_imag * h_post_real).sum(dim=2, keepdim=True))

        # === t=1..T-1: normal Kalman filter iterations ===
        for t in range(1, T):
            xt_real = ref_real_padded[..., t: t + self.L]  # (1, F, L)
            xt_imag = ref_imag_padded[..., t: t + self.L]  # (1, F, L)
            mic_real_t = mic_real[..., t:t + 1]  # (1, F, 1)
            mic_imag_t = mic_imag[..., t:t + 1]  # (1, F, 1)

            # h_prior update: compute dh and swap in one step
            dh_real = h_post_real - h_prior_real  # (1, F, L)
            dh_imag = h_post_imag - h_prior_imag
            h_prior_real, h_post_real = h_post_real, h_prior_real
            h_prior_imag, h_post_imag = h_post_imag, h_prior_imag

            # Compute error using fused dot product
            e_real = mic_real_t - (xt_real * h_prior_real - xt_imag * h_prior_imag).sum(dim=2, keepdim=True)
            e_imag = mic_imag_t - (xt_real * h_prior_imag + xt_imag * h_prior_real).sum(dim=2, keepdim=True)

            # Build input feature: [xt, e, dh] concatenated along feature dim
            # xt: (1, F, L), e: (1, F, 1), dh: (1, F, L) -> total: (1, F, 2*L+1)
            input_real = torch.cat([xt_real, e_real, dh_real], dim=2)
            input_imag = torch.cat([xt_imag, e_imag, dh_imag], dim=2)

            # KGNet forward
            kg_real, kg_imag, h_rr, h_ir, h_ri, h_ii = self.kg_net(input_real, input_imag, h_rr, h_ir, h_ri, h_ii)

            # h_post update (ONNX-compatible, no in-place ops)
            h_post_real = h_prior_real + kg_real * e_real - kg_imag * e_imag
            h_post_imag = h_prior_imag + kg_real * e_imag + kg_imag * e_real

            # Echo estimate via dot product -> (1, F, 1) each, collected for cat
            echo_frames_real.append((xt_real * h_post_real - xt_imag * h_post_imag).sum(dim=2, keepdim=True))
            echo_frames_imag.append((xt_real * h_post_imag + xt_imag * h_post_real).sum(dim=2, keepdim=True))

        # Cat frame estimates: list of T x (1, F, 1) -> (1, F, T)
        echo_hat_real = torch.cat(echo_frames_real, dim=2)
        echo_hat_imag = torch.cat(echo_frames_imag, dim=2)

        # Compute enhanced signal: s_hat = ISTFT(y - echo_hat)
        out_real = mic_real - echo_hat_real  # (1, F, T)
        out_imag = mic_imag - echo_hat_imag  # (1, F, T)

        # ISTFT
        aec_results = self.custom_istft(out_real, out_imag)  # (1, 1, audio_len)

        if self.use_batch_fold:
            aec_results = aec_results.reshape(1, 1, -1)

        aec_results = aec_results[..., :audio_len]

        if self.output_resample_before_pcm:
            aec_results = torch.nn.functional.interpolate(
                aec_results,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            aec_results = aec_results * self.output_pcm_scale      # [-1, 1] -> int16 PCM; F16/F32 outputs stay normalized.
        if self.output_resample_after_pcm:
            aec_results = torch.nn.functional.interpolate(
                aec_results,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            return aec_results.to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return aec_results
        return aec_results.to(torch.float16)


def load_nkf_weights(export_model, original_state_dict):
    """
    Map weights from the original NKF model (which uses ComplexGRU/ComplexDense)
    to our decomposed real/imag model.
    """
    new_state_dict = {}

    # KGNet.fc_in: Sequential(ComplexDense(2*L+1, fc_dim), ComplexPReLU)
    # Original keys: kg_net.fc_in.0.linear_real.weight, .bias, kg_net.fc_in.0.linear_imag.weight, .bias
    new_state_dict['kg_net.fc_in_dense.linear_real.weight'] = original_state_dict['kg_net.fc_in.0.linear_real.weight']
    new_state_dict['kg_net.fc_in_dense.linear_real.bias'] = original_state_dict['kg_net.fc_in.0.linear_real.bias']
    new_state_dict['kg_net.fc_in_dense.linear_imag.weight'] = original_state_dict['kg_net.fc_in.0.linear_imag.weight']
    new_state_dict['kg_net.fc_in_dense.linear_imag.bias'] = original_state_dict['kg_net.fc_in.0.linear_imag.bias']

    # ComplexPReLU: kg_net.fc_in.1.prelu.weight
    new_state_dict['kg_net.fc_in_act.prelu.weight'] = original_state_dict['kg_net.fc_in.1.prelu.weight']

    # ComplexGRU: kg_net.complex_gru.gru_r and gru_i
    for gru_name in ['gru_r', 'gru_i']:
        for param in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']:
            key_src = f'kg_net.complex_gru.{gru_name}.{param}'
            key_dst = f'kg_net.complex_gru.{gru_name}.{param}'
            new_state_dict[key_dst] = original_state_dict[key_src]

    # fc_out: Sequential(ComplexDense(rnn_dim, fc_dim), ComplexPReLU, ComplexDense(fc_dim, L))
    # fc_out.0 -> fc_out_dense1
    new_state_dict['kg_net.fc_out_dense1.linear_real.weight'] = original_state_dict['kg_net.fc_out.0.linear_real.weight']
    new_state_dict['kg_net.fc_out_dense1.linear_real.bias'] = original_state_dict['kg_net.fc_out.0.linear_real.bias']
    new_state_dict['kg_net.fc_out_dense1.linear_imag.weight'] = original_state_dict['kg_net.fc_out.0.linear_imag.weight']
    new_state_dict['kg_net.fc_out_dense1.linear_imag.bias'] = original_state_dict['kg_net.fc_out.0.linear_imag.bias']

    # fc_out.1 -> fc_out_act (PReLU)
    new_state_dict['kg_net.fc_out_act.prelu.weight'] = original_state_dict['kg_net.fc_out.1.prelu.weight']

    # fc_out.2 -> fc_out_dense2
    new_state_dict['kg_net.fc_out_dense2.linear_real.weight'] = original_state_dict['kg_net.fc_out.2.linear_real.weight']
    new_state_dict['kg_net.fc_out_dense2.linear_real.bias'] = original_state_dict['kg_net.fc_out.2.linear_real.bias']
    new_state_dict['kg_net.fc_out_dense2.linear_imag.weight'] = original_state_dict['kg_net.fc_out.2.linear_imag.weight']
    new_state_dict['kg_net.fc_out_dense2.linear_imag.bias'] = original_state_dict['kg_net.fc_out.2.linear_imag.bias']

    export_model.load_state_dict(new_state_dict, strict=False)
    return export_model




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_NKF_AEC_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant").eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant").eval()
    nkf_export = NKF(
        L=FILTER_ORDER,
        fc_dim=FC_DIM,
        rnn_layers=RNN_LAYERS,
        rnn_dim=RNN_DIM,
        custom_stft=custom_stft,
        custom_istft=custom_istft,
        max_frames=MAX_SIGNAL_LENGTH,
        in_sample_rate=IN_SAMPLE_RATE,
        out_sample_rate=OUT_SAMPLE_RATE,
        use_batch_fold=USE_BATCH_FOLD,
        fold_window=FOLD_WINDOW_LENGTH
    ).eval()
    original_state_dict = torch.load(checkpoint_path, map_location='cpu')
    nkf_export = load_nkf_weights(nkf_export, original_state_dict)
    nkf_export = nkf_export.float().eval()
    nkf_export.cache_export_constants_()

    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    near_end_audio = torch.ones(1, 1, EXPORT_AUDIO_LENGTH, dtype=IN_TORCH_DTYPE)
    far_end_audio = torch.ones(1, 1, EXPORT_AUDIO_LENGTH, dtype=IN_TORCH_DTYPE)

    Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        nkf_export,
        (far_end_audio, near_end_audio),
        onnx_model_A,
        input_names=['far_end_audio', 'near_end_audio'],
        output_names=['aec_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'far_end_audio': {2: 'audio_len'},
            'near_end_audio': {2: 'audio_len'},
            'aec_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del nkf_export
    del near_end_audio
    del far_end_audio
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="NKF_AEC", task="aec", model_family="nkf_aec",
    max_dynamic_audio_seconds=4, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=2, feature_kind="stft_kalman_filter", center_pad=True, pad_mode="constant",
    extra={"filter_order": FILTER_ORDER, "fc_dim": FC_DIM, "rnn_layers": RNN_LAYERS, "rnn_dim": RNN_DIM},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
