import gc
import subprocess
import sys
from pathlib import Path

import torch
import torch.fft
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


parent_path          = Path(__file__).resolve().parent                                                     # The folder that contains this script.
project_path         = str(Path.home() / "Downloads" / "Deep-echo-path-modeling-for-acoustic-echo-cancellation-main")  # The Deep_Echo_AEC GitHub project download path. https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation
onnx_model_A         = str(parent_path / "Deep_Echo_AEC_ONNX" / "Deep_Echo_AEC.onnx")                      # The exported onnx model path.
onnx_model_Metadata  = str(metadata_path_for_model(onnx_model_A))                                          # The metadata carrier onnx model path.


DYNAMIC_AXES         = False                                                                      # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
IN_SAMPLE_RATE       = 16000                                                                      # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE      = 16000                                                                      # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH   = 32000                                                                      # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.
MAX_SIGNAL_LENGTH    = 2048 if DYNAMIC_AXES else 256                                              # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
IN_AUDIO_DTYPE       = 'INT16'                                                                    # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'INT16'                                                                    # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16            = float(1.0 / 32768.0)
OPSET                = 20                                                                         # ONNX opset.
WINDOW_TYPE          = 'hamming'                                                                  # Type of window function used in the STFT.
NFFT                 = 319                                                                        # Number of FFT components for the STFT process.
WINDOW_LENGTH        = 319                                                                        # Length of windowing, edit it carefully.
HOP_LENGTH           = 160                                                                        # Number of samples between successive frames in the STFT.
ECHO_ORDER           = 10                                                                         # Number of Deep_Echo path orders. Do not edit.
MODEL_SAMPLE_RATE    = 16000                                                                      # The Deep_Echo model runs internally at 16 kHz.
BATCH_WINDOW_SECONDS = 1.5                                                                        # Minimum input length (seconds) that triggers window folding.
FOLD_WINDOW_LENGTH   = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length rounded UP to a HOP multiple.
USE_BATCH_FOLD       = False                                                                      # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
EXPORT_AUDIO_LENGTH  = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH

if USE_BATCH_FOLD:
    MAX_SIGNAL_LENGTH = FOLD_WINDOW_LENGTH // HOP_LENGTH + 10  # Per-window STFT frame count (+margin); the default 128 would truncate a folded window's backend STFT.

MODEL_BATCH_SIZE     = EXPORT_AUDIO_LENGTH // FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else 1
MODEL_AUDIO_LENGTH   = FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else (EXPORT_AUDIO_LENGTH * MODEL_SAMPLE_RATE) // IN_SAMPLE_RATE
MODEL_STFT_FRAMES    = (MODEL_AUDIO_LENGTH + 2 * (NFFT // 2) - NFFT) // HOP_LENGTH + 1
STATIC_AUDIO_LENGTH  = 0 if DYNAMIC_AXES else MODEL_AUDIO_LENGTH
STATIC_STFT_FRAMES   = 0 if DYNAMIC_AXES else MODEL_STFT_FRAMES
if not DYNAMIC_AXES:
    MAX_SIGNAL_LENGTH = MODEL_STFT_FRAMES


def maybe_strip_prefix(state_dict, prefix: str):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'network', 'net'):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported checkpoint format. Expected a state_dict-like mapping.')

    for prefix in ('module.', 'model.', 'network.', 'net.'):
        checkpoint = maybe_strip_prefix(checkpoint, prefix)

    return checkpoint


class CFB(torch.nn.Module):
    def __init__(self, in_channels=None, out_channels=None, batch_size=1, frames=0, build_ceps_basis=True, external_lstm_state=False):
        super(CFB, self).__init__()
        self.conv_gate = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv_input = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, groups=1, bias=True)
        self.ceps_unit = CepsUnit(
            ch=out_channels,
            batch_size=batch_size,
            frames=frames,
            build_basis=build_ceps_basis,
            external_lstm_state=external_lstm_state,
        )
        self.LN0 = LayerNorm(in_channels, f=160)
        self.LN1 = LayerNorm(out_channels, f=160)
        self.LN2 = LayerNorm(out_channels, f=160)

    def forward(self, x, norm_eps=None, ceps_norm_eps=None, stft_kernel=None, inverse_basis=None, initial_state=None):
        g = torch.sigmoid(self.conv_gate(self.LN0(x, norm_eps)))
        x = self.conv_input(x)
        gx = g * x
        y = self.conv(self.LN1(gx, norm_eps))
        y = y + self.ceps_unit(
            self.LN2(x - gx, norm_eps),
            ceps_norm_eps,
            stft_kernel,
            inverse_basis,
            initial_state,
        )
        return y


class CepsUnit(torch.nn.Module):
    def __init__(self, ch, batch_size=1, frames=0, build_basis=True, external_lstm_state=False):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.batch_size = batch_size
        self.frames = frames
        self.ch_lstm_f = CH_LSTM_F(
            ch * 2,
            ch,
            ch * 2,
            f=81,
            batch_size=batch_size,
            frames=frames,
            register_state=not external_lstm_state,
        )
        # The cepstral DFT can drive sum(x^2) above FLOAT16_MAX even though the
        # normalized result is small. Quarter-scaling both x and epsilon is
        # algebraically neutral, but keeps this complete reduction chain in FP16.
        self.LN = LayerNorm(ch * 2, f=81, sum_scale=0.25)
        self.f = 81
        self.f2 = 81 * 2

        n_fft = NFFT // 2 + 1
        self.n_fft = n_fft
        self.hop_len = n_fft

        if build_basis:
            window = torch.ones(n_fft)
            time_idx = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
            freq_idx = torch.arange(self.f, dtype=torch.float32).unsqueeze(1)
            omega = 2 * torch.pi * freq_idx * time_idx / n_fft
            cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
            sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
            self.register_buffer('stft_kernel', torch.cat([cos_kernel, sin_kernel], dim=0))

            fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
            fourier_basis_real_imag = torch.vstack([
                torch.real(fourier_basis[:self.f]),
                torch.imag(fourier_basis[:self.f])
            ]).float()
            inverse_basis_pinv = torch.linalg.pinv(fourier_basis_real_imag).T
            inverse_basis = window.unsqueeze(0) * inverse_basis_pinv.unsqueeze(1)
            self.register_buffer('inverse_basis', inverse_basis)

    def forward(self, x0, norm_eps=None, stft_kernel=None, inverse_basis=None, initial_state=None):
        batch = self.batch_size if self.frames else x0.shape[0]
        frames = self.frames if self.frames else x0.shape[3]
        if stft_kernel is None:
            stft_kernel = self.stft_kernel
        if inverse_basis is None:
            inverse_basis = self.inverse_basis
        x_reshaped = x0.transpose(2, 3).contiguous().reshape(-1, 1, self.n_fft)
        stft_out = torch.nn.functional.conv1d(x_reshaped, stft_kernel, stride=self.hop_len)
        stft_pair = stft_out.reshape(batch, self.ch, frames, 2, self.f).permute(0, 3, 1, 4, 2).contiguous()
        stft_real, stft_imag = stft_pair.split(1, dim=1)

        lstm_output = self.ch_lstm_f(
            self.LN(stft_pair.reshape(batch, self.ch * 2, self.f, frames), norm_eps),
            initial_state,
        )
        processed_pair = lstm_output.reshape(batch, 2, self.ch, self.f, frames)
        processed_real, processed_imag = processed_pair.split(1, dim=1)
        out_real = processed_real * stft_real - processed_imag * stft_imag
        out_imag = processed_real * stft_imag + processed_imag * stft_real

        packed = torch.cat((out_real, out_imag), dim=3).permute(0, 1, 2, 4, 3).contiguous().reshape(-1, self.f2, 1)
        inv = torch.nn.functional.conv_transpose1d(packed, inverse_basis, stride=self.hop_len)
        return inv.reshape(batch, self.ch, frames, self.n_fft).transpose(2, 3).contiguous()


class LayerNorm(torch.nn.Module):
    def __init__(self, c, f, sum_scale=1.0):
        super(LayerNorm, self).__init__()
        self.w = torch.nn.Parameter(torch.ones(1, c, f, 1))
        self.b = torch.nn.Parameter(torch.rand(1, c, f, 1) * 1e-4)
        denom = float(max(c * f - 1, 1))
        self.register_buffer('weight_scale', torch.tensor(denom ** 0.5, dtype=torch.float32))
        self.norm_eps = 1e-8 * denom
        self.sum_scale = float(sum_scale)
        self.sum_scale_squared = float(sum_scale * sum_scale)
        self.var_scale_fused = False

    def fuse_var_scale_(self):
        if self.var_scale_fused:
            return
        with torch.no_grad():
            self.w.mul_(self.weight_scale.to(dtype=self.w.dtype))
        self.var_scale_fused = True

    def forward(self, x, norm_eps=None):
        mean = x.mean((1, 2), keepdim=True)
        x = x - mean
        eps = self.norm_eps if norm_eps is None else norm_eps
        if self.sum_scale != 1.0:
            x = x * self.sum_scale
            eps = eps * self.sum_scale_squared
        inv_std = torch.rsqrt(x.square().sum((1, 2), keepdim=True) + eps)
        return x * inv_std * self.w + self.b


def fuse_layer_norm_scales_(module):
    for submodule in module.modules():
        if isinstance(submodule, LayerNorm):
            submodule.fuse_var_scale_()


class CH_LSTM_T(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1, f=160, batch_size=1, frames=0):
        super(CH_LSTM_T, self).__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.bi = 1 if not bi else 2
        self.linear = torch.nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f
        self.batch_size = batch_size
        self.frames = frames
        self.register_buffer(
            'initial_state',
            torch.zeros(num_layers * self.bi, batch_size * f, feat_ch),
        )

    def forward(self, x):
        batch = self.batch_size
        frames = self.frames if self.frames else x.shape[3]
        x = x.permute(0, 2, 3, 1).contiguous().reshape(batch * self.f, frames, self.lstm2.input_size)
        x, _ = self.lstm2(x, (self.initial_state, self.initial_state))
        x = self.linear(x)
        return x.reshape(batch, self.f, frames, self.out_ch).permute(0, 3, 1, 2).contiguous()


class CH_LSTM_F(torch.nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1, f=160, batch_size=1, frames=0, register_state=True):
        super(CH_LSTM_F, self).__init__()
        self.lstm2 = torch.nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.bi = 1 if not bi else 2
        self.linear = torch.nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch
        self.f = f
        self.batch_size = batch_size
        self.frames = frames
        if frames and register_state:
            self.register_buffer(
                'initial_state',
                torch.zeros(num_layers * self.bi, batch_size * frames, feat_ch),
            )

    def forward(self, x, initial_state=None):
        batch = self.batch_size if self.frames else x.shape[0]
        frames = self.frames if self.frames else x.shape[3]
        x = x.transpose(1, 3).contiguous().reshape(batch * frames, self.f, self.lstm2.input_size)
        if initial_state is not None:
            x, _ = self.lstm2(x, (initial_state, initial_state))
        elif hasattr(self, 'initial_state'):
            x, _ = self.lstm2(x, (self.initial_state, self.initial_state))
        else:
            x, _ = self.lstm2(x)
        x = self.linear(x)
        return x.reshape(batch, frames, self.f, self.out_ch).transpose(1, 3).contiguous()


class NET(torch.nn.Module):
    def __init__(self, order=ECHO_ORDER, channels=20, custom_istft=None, batch_size=1, frames=0):
        super(NET, self).__init__()
        self.order = order
        self.batch_size = batch_size
        self.frames = frames
        self.n_freq = NFFT // 2 + 1
        self.custom_istft = custom_istft
        delay_kernel = torch.eye(order, dtype=torch.float32).repeat(2, 1).reshape(2 * order, 1, 1, order)
        self.register_buffer('delay_bank_kernel', delay_kernel)
        self.register_buffer('delay_pad', torch.zeros(batch_size, 2, self.n_freq, order - 1))
        self.register_buffer('norm_eps', torch.tensor(1e-8 * (channels * self.n_freq - 1), dtype=torch.float32))
        self.register_buffer('ceps_norm_eps', torch.tensor(1e-8 * (2 * channels * 81 - 1), dtype=torch.float32))
        if frames:
            self.register_buffer('bidir_initial_state', torch.zeros(2, batch_size * frames, channels))

        self.in_ch_lstm = CH_LSTM_F(4, channels, channels, batch_size=batch_size, frames=frames, register_state=False)
        self.in_conv = torch.nn.Conv2d(in_channels=4 + channels, out_channels=channels, kernel_size=(1, 1))
        self.cfb_e1 = CFB(channels, channels, batch_size=batch_size, frames=frames, external_lstm_state=True)
        self.ln = LayerNorm(channels, 160)
        self.ch_lstm = CH_LSTM_T(in_ch=channels, feat_ch=channels * 2, out_ch=channels, num_layers=2, batch_size=batch_size, frames=frames)
        self.cfb_d1 = CFB(
            channels,
            channels,
            batch_size=batch_size,
            frames=frames,
            build_ceps_basis=False,
            external_lstm_state=True,
        )
        self.out_ch_lstm = CH_LSTM_T(2 * channels, channels, channels * 2, batch_size=batch_size, frames=frames)
        self.out_conv = torch.nn.Conv2d(in_channels=channels * 3, out_channels=order * 2, kernel_size=(1, 1), padding=(0, 0), bias=True)

    def apply_echo_path(self, far_comp, est_echo_path):
        padding_far = torch.cat((self.delay_pad, far_comp), dim=-1)
        delayed_far = torch.nn.functional.conv2d(padding_far, self.delay_bank_kernel, groups=2)
        delayed_far = delayed_far.reshape(self.batch_size, 2, self.order, self.n_freq, self.frames or -1)
        far_real, far_imag = delayed_far.split(1, dim=1)
        path_real, path_imag = est_echo_path.split(1, dim=1)
        est_echo_real = far_real * path_real - far_imag * path_imag
        est_echo_imag = far_real * path_imag + far_imag * path_real
        return torch.cat([est_echo_real.sum(dim=2), est_echo_imag.sum(dim=2)], dim=1)

    def forward(self, x):
        frames = self.frames if self.frames else x.shape[3]
        initial_state = self.bidir_initial_state if self.frames else None
        ceps_basis = self.cfb_e1.ceps_unit
        mix_comp = x[:, 0::2].contiguous()
        far_comp = x[:, 1::2].contiguous()
        e0 = self.in_ch_lstm(x, initial_state)
        e0 = self.in_conv(torch.cat([e0, x], dim=1))
        e1 = self.cfb_e1(
            e0,
            self.norm_eps,
            self.ceps_norm_eps,
            ceps_basis.stft_kernel,
            ceps_basis.inverse_basis,
            initial_state,
        )
        lstm_out = self.ch_lstm(self.ln(e1, self.norm_eps))
        d1 = self.cfb_d1(
            e1 * lstm_out,
            self.norm_eps,
            self.ceps_norm_eps,
            ceps_basis.stft_kernel,
            ceps_basis.inverse_basis,
            initial_state,
        )
        d0 = self.out_ch_lstm(torch.cat([e0, d1], dim=1))
        out = self.out_conv(torch.cat([d0, d1], dim=1))
        est_echo_path = out.reshape(self.batch_size, 2, self.order, self.n_freq, frames)
        enhanced = mix_comp - self.apply_echo_path(far_comp, est_echo_path)
        return self.custom_istft.forward_packed(enhanced.reshape(self.batch_size, 2 * self.n_freq, frames))


class DeepEchoAEC(torch.nn.Module):
    def __init__(self, iccrn, custom_stft, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0, model_batch_size=1, model_audio_length=0, model_frames=0, input_scale_folded=False):
        super(DeepEchoAEC, self).__init__()
        self.iccrn = iccrn
        self.use_batch_fold = use_batch_fold          # Fold long near/far inputs into fixed windows batched on dim0
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.model_batch_size = model_batch_size
        self.model_audio_length = model_audio_length
        self.model_frames = model_frames
        self.n_freq = NFFT // 2 + 1
        self.custom_stft = custom_stft
        self.input_scale_folded = input_scale_folded
        if not input_scale_folded:
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

    def forward(self, near_end_audio, far_end_audio):
        if self.use_batch_fold:
            near_end_audio = near_end_audio.reshape(-1, 1, self.fold_window)
            far_end_audio = far_end_audio.reshape(-1, 1, self.fold_window)
        audio_pair = torch.cat((near_end_audio, far_end_audio), dim=1).float()
        if self.resample_before_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower() and not self.input_scale_folded:
            audio_pair = audio_pair * self.inv_int16      # int16 PCM -> [-1, 1]; F16/F32 inputs already arrive normalized.
        audio_pair = audio_pair - audio_pair.mean(dim=2, keepdim=True)
        if self.resample_after_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        audio_length = self.model_audio_length or audio_pair.shape[2]
        frames = self.model_frames or -1
        packed_pair = self.custom_stft(audio_pair.reshape(self.model_batch_size * 2, 1, audio_length))
        # Reorder source-major packed spectra to the checkpoint's complex-major channel order:
        # [near_real, far_real, near_imag, far_imag].
        spec_input = packed_pair.reshape(self.model_batch_size, 2, 2, self.n_freq, frames)
        spec_input = spec_input.permute(0, 2, 1, 3, 4).reshape(self.model_batch_size, 4, self.n_freq, frames)
        aec_results = self.iccrn(spec_input)

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
        if self.use_batch_fold:
            aec_results = aec_results.reshape(1, 1, -1)
        if "int" in OUT_AUDIO_DTYPE.lower():
            return aec_results.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return aec_results
        return aec_results.to(torch.float16)




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_Deep_Echo_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    custom_stft = STFT_Process(
        model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0,
        window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant", packed_output=True,
        # Keep the DFT table near [-1, 1] for FP16 precision. DeepEchoAEC applies
        # 1/32768 explicitly before centering, which also prevents ReduceMean overflow.
        input_scale=1.0,
    ).eval()
    custom_istft = STFT_Process(
        model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH,
        max_frames=MODEL_STFT_FRAMES if not DYNAMIC_AXES else MAX_SIGNAL_LENGTH,
        window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant",
        output_length=MODEL_AUDIO_LENGTH if not DYNAMIC_AXES else 0,
        static_norm_divisor=not DYNAMIC_AXES,
    ).eval()
    iccrn = NET(
        order=ECHO_ORDER,
        custom_istft=custom_istft,
        batch_size=MODEL_BATCH_SIZE,
        frames=STATIC_STFT_FRAMES,
    )
    checkpoint_path = Path(project_path).expanduser().resolve() / 'model.ckpt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)
    iccrn.load_state_dict(state_dict, strict=False)
    fuse_layer_norm_scales_(iccrn)
    iccrn = iccrn.float().eval()

    deep_echo = DeepEchoAEC(
        iccrn, custom_stft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, USE_BATCH_FOLD, FOLD_WINDOW_LENGTH,
        MODEL_BATCH_SIZE, STATIC_AUDIO_LENGTH, STATIC_STFT_FRAMES,
        input_scale_folded=False,
    )
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    near_end_audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    far_end_audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)

    torch.onnx.export(
        deep_echo,
        (near_end_audio, far_end_audio),
        onnx_model_A,
        input_names=['near_end_audio', 'far_end_audio'],
        output_names=['aec_audio'],
        dynamic_axes={
            'near_end_audio': {2: 'near_audio_len'},
            'far_end_audio': {2: 'far_audio_len'},
            'aec_audio': {2: 'aec_audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del deep_echo
    del iccrn
    del near_end_audio
    del far_end_audio
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="Deep_Echo_AEC", task="aec", model_family="deep_echo",
    max_dynamic_audio_seconds=30, normalize_audio_default=False, batch_fold_inference_default=False,
    input_channels=1, output_channels=1, num_audio_inputs=2, feature_kind="stft", center_pad=True,
    pad_mode="constant", extra={"echo_order": ECHO_ORDER},
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
