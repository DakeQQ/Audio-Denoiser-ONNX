import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torch.nn as nn
from pydub import AudioSegment
from STFT_Process import STFT_Process


project_path        = "/home/DakeQQ/Downloads/NKF-AEC-gh-pages"             # The NKF-AEC GitHub project download path. https://github.com/jfsean/NKF-AEC
checkpoint_path     = project_path + "/src/nkf_epoch70.pt"                  # The pretrained checkpoint path.
onnx_model_A        = "/home/DakeQQ/Downloads/NKF_AEC_ONNX/NKF_AEC.onnx"    # The exported onnx model path.
test_near_end_audio = "./examples/nearend_mic1.wav"                         # The near end audio path.
test_far_end_audio  = "./examples/farend_speech1.wav"                       # The far end audio path.
save_aec_output     = "./aec.wav"                                           # The output Acoustic Echo Cancellation audio path.


DYNAMIC_AXES       = False           # Only support static axes. Do not edit.
IN_SAMPLE_RATE     = 16000           # The NKF-AEC model runs internally at 16 kHz and resamples the input when needed.
OUT_SAMPLE_RATE    = 16000           # Output sample rate after the internal 16 kHz Kalman-filter model finishes.
INPUT_AUDIO_LENGTH = 32000           # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 32000.
MAX_SIGNAL_LENGTH  = 128             # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE        = 'hann'          # Type of window function used in the STFT
NFFT               = 1024            # Number of FFT components for the STFT process
WINDOW_LENGTH      = 1024            # Length of windowing, edit it carefully.
HOP_LENGTH         = 256             # Number of samples between successive frames in the STFT

# NKF model parameters
FILTER_ORDER       = 4               # Kalman filter order (L), do not edit the value.
FC_DIM             = 18              # Fully-connected layer dimension, do not edit the value.
RNN_LAYERS         = 1               # Number of GRU layers, do not edit the value.
RNN_DIM            = 18              # GRU hidden dimension, do not edit the value.

MAX_THREADS        = 4               # Number of parallel threads for test audio denoising.
NORMALIZE_AUDIO    = False           # Normalize the input audio to a target RMS level (e.g., 8192) before processing. It can help improve the performance of the model, especially for low-volume audio. Set it to True if you want to enable it.
OPSET              = 18              # ONNX opset.


def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-7))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio


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
        if self._fused:
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
        x_real, x_imag = self.fc_in_dense(input_real, input_imag)
        x_real, x_imag = self.fc_in_act(x_real, x_imag)

        # GRU expects (batch=F, seq=1, fc_dim): (1, F, fc_dim) -> (F, 1, fc_dim)
        x_real = x_real.view(self.n_freq, 1, -1)
        x_imag = x_imag.view(self.n_freq, 1, -1)

        # ComplexGRU
        x_real, x_imag, h_rr, h_ir, h_ri, h_ii = self.complex_gru(x_real, x_imag, h_rr, h_ir, h_ri, h_ii)

        # Back to (1, F, rnn_dim) from (F, 1, rnn_dim)
        x_real = x_real.view(1, self.n_freq, -1)
        x_imag = x_imag.view(1, self.n_freq, -1)

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
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim, custom_stft, custom_istft, max_frames, in_sample_rate, out_sample_rate):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim
        self.custom_stft = custom_stft
        self.custom_istft = custom_istft
        self.kg_net = KGNet_Real(L, fc_dim, rnn_layers, rnn_dim)
        self.n_freq = NFFT // 2 + 1  # 513 frequency bins
        self.max_frames = max_frames
        self.inv_int16 = float(1.0 / 32768.0)
        self.output_pcm_scale = 32767.0
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.out_sample_rate_scale = out_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before_centering = self.in_sample_rate_scale > 1.0
        self.resample_after_centering = self.in_sample_rate_scale < 1.0
        self.output_resample_before_pcm = self.out_sample_rate_scale > 1.0
        self.output_resample_after_pcm = self.out_sample_rate_scale < 1.0

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
        audio_pair = torch.cat([far_end_audio, near_end_audio], dim=0).float()
        if self.resample_before_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        audio_pair *= self.inv_int16
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
        ref_real, mic_real = pair_real.split(1, dim=0)
        ref_imag, mic_imag = pair_imag.split(1, dim=0)

        T = ref_real.shape[2]
        if T > self.max_frames:
            raise ValueError(f"Input produces {T} frames, but MAX_SIGNAL_LENGTH is {self.max_frames}.")

        # Use registered buffers directly (already zero-initialized)
        h_prior_real = self.h_prior_real_buffer
        h_prior_imag = self.h_prior_imag_buffer
        h_rr = self.h_rr_buffer
        h_ir = self.h_ir_buffer
        h_ri = self.h_ri_buffer
        h_ii = self.h_ii_buffer

        ref_real_padded = torch.cat([self.zeros_pad, ref_real], dim=2)
        ref_imag_padded = torch.cat([self.zeros_pad, ref_imag], dim=2)

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
        if self.output_resample_before_pcm:
            aec_results = torch.nn.functional.interpolate(
                aec_results,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        aec_results *= self.output_pcm_scale
        if self.output_resample_after_pcm:
            aec_results = torch.nn.functional.interpolate(
                aec_results,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        return aec_results.clamp_(-32768.0, 32767.0).to(torch.int16)


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
        out_sample_rate=OUT_SAMPLE_RATE
    ).eval()
    original_state_dict = torch.load(checkpoint_path, map_location='cpu')
    nkf_export = load_nkf_weights(nkf_export, original_state_dict)
    nkf_export = nkf_export.float().eval()
    nkf_export.cache_export_constants_()

    near_end_audio = torch.ones(1, 1, INPUT_AUDIO_LENGTH, dtype=torch.int16)
    far_end_audio = torch.ones(1, 1, INPUT_AUDIO_LENGTH, dtype=torch.int16)

    Path(onnx_model_A).parent.mkdir(parents=True, exist_ok=True)
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
print('\nExport done!\n\nStart to run NKF-AEC by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=["CPUExecutionProvider"], provider_options=None)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name


# Load the input audio
print(f"\nTest Input Near_End Audio: {test_near_end_audio}\nTest Input Far_End Audio: {test_far_end_audio}")
near_end_audio = np.array(AudioSegment.from_file(test_near_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
far_end_audio = np.array(AudioSegment.from_file(test_far_end_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
near_end_audio_len = len(near_end_audio)
far_end_audio_len = len(far_end_audio)
min_len = min(near_end_audio_len, far_end_audio_len)
if NORMALIZE_AUDIO:
    near_end_audio = normalise_audio(near_end_audio)
    far_end_audio = normalise_audio(far_end_audio)
near_end_audio = near_end_audio[:min_len]
far_end_audio = far_end_audio[:min_len]
near_end_audio = near_end_audio.reshape(1, 1, -1)
far_end_audio = far_end_audio.reshape(1, 1, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = max(4 * IN_SAMPLE_RATE, min_len)  # Default to slice in 4 seconds. You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in


def align_audio(audio, audio_len):
    stride_step = INPUT_AUDIO_LENGTH
    if audio_len > INPUT_AUDIO_LENGTH:
        if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int) & (OUT_SAMPLE_RATE == IN_SAMPLE_RATE):
            stride_step = shape_value_out
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        final_slice = audio[:, :, -pad_amount:].astype(np.float32)
        white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        audio_float = audio.astype(np.float32)
        white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]
    return audio, aligned_len, stride_step


near_end_audio, _, _ = align_audio(near_end_audio, min_len)
far_end_audio, aligned_len, stride_step = align_audio(far_end_audio, min_len)

min_len = int(min_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)

inv_audio_len = float(100.0 / min_len)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _near_end_audio, _far_end_audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _far_end_audio[:, :, _slice_start: _slice_end], in_name_A1: _near_end_audio[:, :, _slice_start: _slice_end]})[0]


# Start to run NKF-AEC
print("\nRunning the NKF-AEC by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS if MAX_THREADS > 0 else 2) as executor:  # Parallel denoised the audio.
    futures = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, near_end_audio, far_end_audio))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.3f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
enhanced_wav = np.concatenate(saved, axis=-1).reshape(-1)[:min_len]
end_time = time.time()
print(f"Complete: 100.00%")

# Save the enhanced wav.
elapsed = end_time - start_time
audio_duration = min_len / OUT_SAMPLE_RATE
rtf = elapsed / audio_duration
sf.write(save_aec_output, enhanced_wav, OUT_SAMPLE_RATE, format='WAVEX')
print(f"\nAEC Process Complete.\n\nSaving to: {save_aec_output}.\n\nTime Cost: {elapsed:.3f} Seconds\nAudio Duration: {audio_duration:.3f} Seconds\nRTF: {rtf:.4f}")
