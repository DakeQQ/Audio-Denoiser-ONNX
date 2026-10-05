import atexit
import gc
import importlib.util
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import (
    build_audio_metadata_from_globals,
    export_metadata_carrier,
    metadata_path_for_model,
    stamp_export_metadata,
)


parent_path         = Path(__file__).resolve().parent                               # The folder that contains this script.

# User settings.
project_path_A      = str(Path.home() / "Downloads" / "speech_dfsmn_aec_psm_16k")   # The DFSMN AEC download path.
project_path_B      = str(parent_path.parent / "RTAEC")                             # The light-AEC project path. Keywords in this path select the backend.
                                                                                    # [RTAEC, SDAEC, Deep_Echo, NKF] are supported. RTAEC is the in-repo ../RTAEC folder (no checkpoint needed). https://github.com/ZhaoF-i/SDAEC ; https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation ; https://github.com/jfsean/NKF-AEC
onnx_model_A        = str(parent_path / "DFSMN_AEC_ONNX" / "DFSMN_AEC.onnx")        # Final exported model.
DYNAMIC_AXES        = False                           # DFSMN_AEC exports a static audio-length graph only.
IN_SAMPLE_RATE       = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE      = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as output.
IN_AUDIO_DTYPE       = 'F32'                          # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE      = 'F32'                          # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
OUTPUT_VAD_RESULT    = False                          # If true, expose the DFSMN speech probability as a second ONNX output.
USE_BATCH_FOLD       = True                           # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
BATCH_WINDOW_SECONDS = 1.5                            # Minimum input length (seconds) that triggers window folding.
INPUT_AUDIO_LENGTH   = 32000                          # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the NFFT value.
AUTO_DELAY_ALIGN     = True                           # Track and align the near/far bulk delay per segment before the light AEC backend (static profiles only).

# VAD post-processing settings.
FUSION_THRESHOLD    = 0.3                             # Merge adjacent speech segments separated by no more than this many seconds.
MIN_SPEECH_DURATION = 0.2                             # Drop speech segments shorter than this many seconds.
SPEAKING_SCORE      = 0.5                             # Probability threshold used to enter the speaking state.
SILENCE_SCORE       = 0.5                             # Probability threshold used to return to silence.
LOOK_AHEAD          = 0.3                             # Seconds of future VAD frames used to smooth state transitions.

# Fixed DFSMN-AEC model, backend, and ONNX export parameters.
temporary_export_paths = set()
OPSET                = 20
INV_INT16            = float(1.0 / 32768.0)
# DFSMN_AEC
MODEL_SAMPLE_RATE = 16000                           # The DFSMN AEC model runs internally at 16 kHz; the Kaldi fbank is computed at this rate.
WINDOW_TYPE     = 'hamming_symmetric'               # Mask STFT/ISTFT window: symmetric hamming, matching the original torch.hamming_window(640, periodic=False).
NFFT_A          = 1024                              # Kaldi fbank FFT size = next_power_of_two(frame_length); 40 ms @ 16 kHz -> 640 -> 1024. Edit it carefully.
NFFT_A2         = 640                               # Mask STFT FFT size (= n_fft from the dey_mini.yaml loss config), edit it carefully.
WINDOW_LENGTH_A = 640                               # Kaldi frame length (40 ms @ 16 kHz) and mask STFT window length, edit it carefully.
HOP_LENGTH_A    = 320                               # Frame shift (20 ms @ 16 kHz) for both the Kaldi fbank and the mask STFT.
N_MELS          = 80                                # Number of Mel bands (= feature_size in dey_mini.yaml), edit it carefully.
PRE_EMPHASIZE   = 0.97                              # Kaldi per-frame pre-emphasis coefficient.


# Light-AEC backend
def infer_light_aec_model(project_path: str) -> str:
    project_path_key = project_path.lower().replace('-', '_').replace(' ', '_')
    if 'deep_echo' in project_path_key or ('deep' in project_path_key and 'echo' in project_path_key):
        return 'Deep_Echo'
    if 'nkf' in project_path_key:
        return 'NKF'
    if 'sdaec' in project_path_key:
        return 'SDAEC'
    if 'rtaec' in project_path_key:
        return 'RTAEC'
    raise ValueError(
        f"Unable to infer LIGHT_AEC_MODEL from project_path_B: {project_path}. "
        "Expected keywords for 'RTAEC', 'SDAEC', 'Deep_Echo', or 'NKF'.")


# Each backend is exported by its own sibling script; only its model classes are reused here.
LIGHT_AEC_SCRIPTS = {
    'RTAEC': ('RTAEC', 'Export_RTAEC.py'),
    'SDAEC': ('SDAEC', 'Export_SDAEC.py'),
    'Deep_Echo': ('Deep_Echo_AEC', 'Export_Deep_Echo.py'),
    'NKF': ('NKF_AEC', 'Export_NKF_AEC.py'),
}


def load_light_aec_module(light_aec_model: str):
    folder, script_name = LIGHT_AEC_SCRIPTS[light_aec_model]
    script_path = parent_path.parent / folder / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f'{light_aec_model} exporter not found: {script_path}')
    if str(script_path.parent) not in sys.path:
        sys.path.append(str(script_path.parent))   # Backend-local imports, e.g. NKF's Rewrite_ONNX_Initializer_Identities.
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # DFSMN_AEC normalizes and resamples before the backend, so it always sees 16 kHz float audio.
    module.IN_AUDIO_DTYPE = module.OUT_AUDIO_DTYPE = 'F32'
    module.IN_SAMPLE_RATE = module.OUT_SAMPLE_RATE = MODEL_SAMPLE_RATE
    if light_aec_model != 'RTAEC':
        module.project_path = project_path_B
    if light_aec_model == 'NKF':
        module.checkpoint_path = project_path_B + '/src/nkf_epoch70.pt'
    return module


LIGHT_AEC_MODEL = infer_light_aec_model(project_path_B)
light_aec_module = load_light_aec_module(LIGHT_AEC_MODEL)

if DYNAMIC_AXES:
    raise ValueError('DFSMN_AEC supports static export profiles only (DYNAMIC_AXES=False).')
if LIGHT_AEC_MODEL == 'NKF':
    INPUT_AUDIO_LENGTH = max(32000, INPUT_AUDIO_LENGTH)  # NKF AEC model is recommended >= 32000
if LIGHT_AEC_MODEL == 'RTAEC':
    USE_BATCH_FOLD = False                                 # The block-Wiener solver is single-batch.

WINDOW_TYPE_B   = light_aec_module.WINDOW_TYPE
NFFT_B          = light_aec_module.NFFT
WINDOW_LENGTH_B = light_aec_module.WINDOW_LENGTH
HOP_LENGTH_B    = light_aec_module.HOP_LENGTH
TARGET_TDE_LAG  = light_aec_module.TARGET_TDE_LAG
ALPHA_K         = getattr(light_aec_module, 'ALPHA_K', None)
ECHO_ORDER      = getattr(light_aec_module, 'ECHO_ORDER', None)
FILTER_ORDER    = getattr(light_aec_module, 'FILTER_ORDER', None)


# Derived export dimensions.
onnx_model_Metadata = str(metadata_path_for_model(onnx_model_A))
# Batch-fold: for a static, same-rate (in==model==out) input at least BATCH_WINDOW_SECONDS long, fold the near/far
# waveforms into fixed-length windows and batch-process them together (each window is an independent AEC clip -> the
# adaptive filter / recurrent state stays WITHIN a window; the window count rides in the batch dimension).
FOLD_WINDOW_LENGTH   = (
    (int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH_A - 1) // HOP_LENGTH_A
) * HOP_LENGTH_A  # Per-window model-rate length, rounded UP to a mask-hop (HOP_LENGTH_A) multiple so the center=False mask STFT/ISTFT reconstructs exactly W samples per window.
EXPORT_AUDIO_LENGTH = (
    ((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH
) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is zero-padded OUTSIDE the model by the windowing loop.
if USE_BATCH_FOLD and (
    IN_SAMPLE_RATE != MODEL_SAMPLE_RATE
    or OUT_SAMPLE_RATE != MODEL_SAMPLE_RATE
    or INPUT_AUDIO_LENGTH < int(BATCH_WINDOW_SECONDS * IN_SAMPLE_RATE)
):
    raise ValueError(
        'USE_BATCH_FOLD requires identical input/model/output sample rates '
        'and an input at least BATCH_WINDOW_SECONDS long.')


def stft_frame_count(audio_length: int, n_fft: int, hop_length: int, center_pad: bool) -> int:
    padded_length = audio_length + (2 * (n_fft // 2) if center_pad else 0)
    return (padded_length - n_fft) // hop_length + 1


MODEL_AUDIO_LENGTH = FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else int(
    round(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
MODEL_BATCH = EXPORT_AUDIO_LENGTH // FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else 1
MASK_FRAMES_A2 = stft_frame_count(MODEL_AUDIO_LENGTH, NFFT_A2, HOP_LENGTH_A, False)
BACKEND_FRAMES_B = stft_frame_count(MODEL_AUDIO_LENGTH, NFFT_B, HOP_LENGTH_B, True)
# Static shape profile that the backend classes read from their module globals.
light_aec_module.STATIC_MODEL_BATCH = MODEL_BATCH
light_aec_module.STATIC_SIGNAL_LENGTH = BACKEND_FRAMES_B
light_aec_module.INPUT_AUDIO_LENGTH = light_aec_module.EXPORT_AUDIO_LENGTH = MODEL_BATCH * MODEL_AUDIO_LENGTH


def get_kaldi_mel_banks(num_bins, window_length_padded, sample_freq, low_freq=20.0, high_freq=0.0):
    """Reproduce torchaudio.compliance.kaldi.get_mel_banks (VTLN off) using pure PyTorch."""
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high_freq = 1127.0 * math.log(1.0 + high_freq / 700.0)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    bin_idx = torch.arange(num_bins, dtype=torch.float64).unsqueeze(1)
    left_mel = mel_low_freq + bin_idx * mel_freq_delta
    center_mel = mel_low_freq + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_idx + 2.0) * mel_freq_delta
    mel = (1127.0 * torch.log(1.0 + (fft_bin_width * torch.arange(num_fft_bins, dtype=torch.float64)) / 700.0)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    return torch.clamp(torch.min(up_slope, down_slope), min=0.0)


def build_kaldi_fbank_conv(n_fft, win_length, n_mels, sample_rate, preemphasis_coefficient,
                           low_freq=20.0, high_freq=0.0, waveform_scale=32768.0):
    """Fold torchaudio.compliance.kaldi.fbank into a single Conv1d weight + Mel matrix.

    The original DFSMN-AEC preprocessor extracts features with
    ``kaldi.fbank(utt, frame_length=40, frame_shift=20, num_mel_bins=80,
    sample_frequency=16000, window_type='hamming')`` on int16-scale audio. Every
    per-frame step is linear, so DC-offset removal, 0.97 pre-emphasis, the symmetric
    Hamming window and the real/imag DFT collapse into one stride-``hop`` convolution.

    ``waveform_scale`` is baked into the weight. Use 32768 for normalized waveforms or
    1 for float-cast int16 PCM values; both reproduce the int16-scale Kaldi spectrum.

    Returns:
        conv_weight: (2 * (n_fft // 2 + 1), 1, win_length) float32 Conv1d weight.
        mel_banks:   (1, n_mels, n_fft // 2 + 1) float32 Mel filterbank.
    """
    f_bins = n_fft // 2 + 1
    window = torch.hamming_window(win_length, periodic=False, alpha=0.54, beta=0.46, dtype=torch.float64)
    t = torch.arange(win_length, dtype=torch.float64).unsqueeze(0)
    f = torch.arange(f_bins, dtype=torch.float64).unsqueeze(1)
    omega = (2.0 * torch.pi / n_fft) * f * t
    windowed_cos = torch.cos(omega) * window.unsqueeze(0)
    windowed_sin = -torch.sin(omega) * window.unsqueeze(0)
    # Kaldi order: per-frame DC removal, then pre-emphasis, then window + DFT.
    # Apply the two transforms directly to each DFT row instead of constructing and
    # multiplying dense win_length x win_length matrices during model construction.
    dft_weight = torch.cat([windowed_cos, windowed_sin], dim=0)
    transformed = dft_weight.clone()
    transformed[:, 0] *= 1.0 - preemphasis_coefficient
    transformed[:, :-1] -= preemphasis_coefficient * dft_weight[:, 1:]
    transformed -= transformed.mean(dim=1, keepdim=True)
    conv_weight = transformed.unsqueeze(1) * waveform_scale
    mel_banks = get_kaldi_mel_banks(n_mels, n_fft, float(sample_rate), low_freq, high_freq)
    mel_banks = torch.nn.functional.pad(mel_banks, (0, 1)).unsqueeze(0)
    return conv_weight.float(), mel_banks.float()


class DFSMN_AEC(torch.nn.Module):
    def __init__(self, dfsmn_aec, light_aec, light_aec_type, custom_stft_A2, custom_istft_A2, nfft_A, win_length_A, hop_length_A, pre_emphasis, in_sample_rate, out_sample_rate, n_mels, use_batch_fold=False, fold_window=0, static_batch=0, static_audio_length=0, mask_frames=0, output_vad_result=False):
        super(DFSMN_AEC, self).__init__()
        if light_aec is None:
            raise ValueError('light_aec must be provided.')
        self.use_batch_fold = use_batch_fold          # Fold long near/far inputs into fixed windows batched on dim0
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.static_batch = static_batch
        self.static_audio_length = static_audio_length
        self.mask_frames = mask_frames
        self.output_vad_result = output_vad_result
        self.dfsmn_aec = dfsmn_aec.model.to('cpu').float()
        # Inlined UniDeepFsmn causal FSMN memory (replaces modeling_modified/uni_deep_fsmn.py).
        # Every deepfsmn layer shares input_dim and padding_left (= dilation * (lorder - 1)),
        # so a single zero pad template feeds all of them.
        first_uni_deep_fsmn = self.dfsmn_aec.deepfsmn[0]
        pad_batch = static_batch if static_batch else 1
        self.register_buffer(
            'fsmn_pad',
            torch.zeros(pad_batch, first_uni_deep_fsmn.output_dim, first_uni_deep_fsmn.padding_left))
        self.fsmn_conv_weights = []
        self.fsmn_conv_biases = []
        self.fsmn_conv_strides = []
        self.fsmn_conv_dilations = []
        self.fsmn_conv_groups = []
        for index, layer in enumerate(self.dfsmn_aec.deepfsmn):
            weight_name = f'fsmn_conv_weight_{index}'
            self.register_buffer(weight_name, layer.conv1.weight.detach().squeeze(-1).contiguous())
            self.fsmn_conv_weights.append(getattr(self, weight_name))
            if layer.conv1.bias is None:
                self.fsmn_conv_biases.append(None)
            else:
                bias_name = f'fsmn_conv_bias_{index}'
                self.register_buffer(bias_name, layer.conv1.bias.detach().contiguous())
                self.fsmn_conv_biases.append(getattr(self, bias_name))
            self.fsmn_conv_strides.append(layer.conv1.stride[0])
            self.fsmn_conv_dilations.append(layer.conv1.dilation[0])
            self.fsmn_conv_groups.append(layer.conv1.groups)

        shift = dfsmn_aec.preprocessor.feature.shift.float()
        scale = dfsmn_aec.preprocessor.feature.scale.float()
        first_affine = self.dfsmn_aec.linear1.linear
        with torch.no_grad():
            folded_weight = (
                first_affine.weight.double() * scale.double().unsqueeze(0)).float()
            folded_bias = (
                first_affine.bias.double()
                + first_affine.weight.double() @ (shift.double() * scale.double())).float()
        self.register_buffer('feature_linear_weight', folded_weight)
        self.register_buffer('feature_linear_bias', folded_bias)
        self.light_aec = light_aec
        self.light_aec_type = light_aec_type
        self.custom_stft_A2 = custom_stft_A2
        self.custom_istft_A2 = custom_istft_A2
        self.register_buffer(
            'input_pcm_scale', torch.tensor(INV_INT16, dtype=torch.float32))
        self.register_buffer(
            'fbank_power_scale', torch.tensor(32768.0 * 32768.0, dtype=torch.float32))
        self.register_buffer(
            'output_pcm_scale', torch.tensor(32767.0, dtype=torch.float32))
        self.register_buffer(
            'output_min', torch.tensor(-32768.0, dtype=torch.float32))
        self.register_buffer(
            'output_max', torch.tensor(32767.0, dtype=torch.float32))
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

        # Keep the fbank convolution in the normalized waveform domain. Its power
        # is restored by the exact 2**30 factor immediately before the FP32 Mel/log
        # island, avoiding both tiny FP16 DFT weights and overflowing FP16 power.
        fbank_waveform_scale = 1.0
        fbank_conv_weight, mel_banks = build_kaldi_fbank_conv(
            nfft_A, win_length_A, n_mels, MODEL_SAMPLE_RATE, pre_emphasis,
            waveform_scale=fbank_waveform_scale)
        self.register_buffer('fbank_conv_weight', fbank_conv_weight)
        self.register_buffer('mel_banks', mel_banks.unsqueeze(1))
        self.feat_hop_length = hop_length_A
        self.feat_n_freq = nfft_A // 2 + 1
        self.feat_frames = mask_frames
        self.feat_dim = n_mels * 3
        self.n_mels = n_mels
        self.temp_feature_scale = 1.0
        self.log_floor = float(torch.finfo(torch.float32).eps)
        self.factor = float(1.15)  # Matches the original DFSMN echo estimate scaling.

    def _preprocess_audio_pair(self, near_end_audio, far_end_audio):
        audio_pair = torch.cat([near_end_audio, far_end_audio], dim=0).float()
        if self.resample_before_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in IN_AUDIO_DTYPE.lower():
            audio_pair = audio_pair * self.input_pcm_scale  # int16 PCM -> [-1, 1]; float inputs already arrive normalized.
        if self.resample_after_centering:
            audio_pair = torch.nn.functional.interpolate(
                audio_pair,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        return audio_pair

    def _run_light_aec_backend(self, audio_pair, batch):
        near_audio, far_audio = audio_pair.split(batch, dim=0)
        if self.light_aec_type == 'NKF':
            # NKF takes [far, near] and returns its batch-folded windows concatenated along time.
            return self.light_aec(far_audio, near_audio).reshape(batch, 1, self.static_audio_length)
        return self.light_aec(near_audio, far_audio)

    def _uni_deep_fsmn(self, index, uni_deep_fsmn, x):
        # Inlined UniDeepFsmn.compute1: linear -> ReLU -> (Identity) norm -> project -> causal depthwise FSMN.
        # The zero-concat left padding reproduces the modeling_modified forward exactly while calling only
        # the pristine modelscope submodules, so no site-packages patching is required.
        hidden = uni_deep_fsmn.project(uni_deep_fsmn.norm(uni_deep_fsmn.act(uni_deep_fsmn.linear(x))))
        hidden = hidden.transpose(1, 2)
        pad = self.fsmn_pad if self.static_batch else self.fsmn_pad.expand(hidden.shape[0], -1, -1)
        memory = torch.nn.functional.conv1d(
            torch.cat([pad, hidden], dim=2),
            self.fsmn_conv_weights[index],
            bias=self.fsmn_conv_biases[index],
            stride=self.fsmn_conv_strides[index],
            dilation=self.fsmn_conv_dilations[index],
            groups=self.fsmn_conv_groups[index])
        if uni_deep_fsmn.skip_connect:
            memory = memory + hidden
        memory = memory.transpose(1, 2)
        return x + memory

    def forward(self, near_end_audio, far_end_audio):
        # 0. Optionally fold long near/far inputs into fixed windows batched on dim0. _preprocess stacks near/far on
        #    dim0, so a fold of `batch` windows makes the pair (2*batch, 1, W): near block first, far block second.
        if self.use_batch_fold:
            near_end_audio = near_end_audio.reshape(self.static_batch, 1, self.fold_window)
            far_end_audio = far_end_audio.reshape(self.static_batch, 1, self.fold_window)
        batch = self.static_batch if self.static_batch else near_end_audio.shape[0]

        # 1. Convert int16 near/far inputs into the float waveform domain expected by the light-AEC backend.
        audio_pair = self._preprocess_audio_pair(near_end_audio, far_end_audio)

        # 2. Run the selected light-AEC backend to obtain a temporary echo-cancelled waveform.
        temp_aec = self._run_light_aec_backend(audio_pair, batch)
        if not self.static_audio_length and temp_aec.shape[-1] > audio_pair.shape[-1]:
            temp_aec = temp_aec[:, :, :audio_pair.shape[-1]]

        # 3. Build the DFSMN Kaldi-fbank features from the near-end mic, the temporary AEC
        #    output (the linear-AEC residual = out_linear) and the echo estimate
        #    (out_echo ~= near - factor * temp_aec). Since the folded fbank Conv is linear,
        #    derive the echo spectrum from near/temp spectra instead of running a third Conv.
        temp_aec_spectrum_A2 = self.custom_stft_A2(temp_aec)
        near_feature = (
            audio_pair[:batch, :, :self.static_audio_length]
            if self.static_audio_length
            else audio_pair[:batch, :, :temp_aec.shape[-1]])
        temp_feature = (
            temp_aec if self.temp_feature_scale == 1.0
            else temp_aec * self.temp_feature_scale)
        fbank_pair = torch.nn.functional.conv1d(
            torch.cat([near_feature, temp_feature], dim=0),
            self.fbank_conv_weight,
            stride=self.feat_hop_length)
        feat_frames = self.feat_frames if self.feat_frames else fbank_pair.shape[-1]
        fbank_pair = fbank_pair.reshape(
            2, batch, 2, self.feat_n_freq, feat_frames).permute(1, 0, 2, 3, 4)
        near_spectrum, temp_spectrum = fbank_pair.split(1, dim=1)
        echo_spectrum = near_spectrum - self.factor * temp_spectrum
        spectra = torch.cat([fbank_pair, echo_spectrum], dim=1)
        # The network was trained on int16-domain Kaldi powers. Restore that exact
        # scale only after the normalized-domain convolution. Optimize_ONNX.py keeps
        # this square/reduction/scale/Mel/log chain as one contiguous FP32 island.
        power = spectra.square().sum(dim=2) * self.fbank_power_scale
        mel_energies = torch.matmul(self.mel_banks, power).clamp(min=self.log_floor).log()
        feat = mel_energies.permute(0, 3, 1, 2).reshape(batch, feat_frames, self.feat_dim)
        x1 = torch.nn.functional.linear(feat, self.feature_linear_weight, self.feature_linear_bias)
        x2 = self.dfsmn_aec.relu(x1)
        x3 = x2
        for index, uni_deep_fsmn in enumerate(self.dfsmn_aec.deepfsmn):
            x3 = self._uni_deep_fsmn(index, uni_deep_fsmn, x3)
        vad_results = (
            self.dfsmn_aec.sig(self.dfsmn_aec.linear3(x3)).reshape(-1)
            if self.output_vad_result else None)
        masks = self.dfsmn_aec.sig(self.dfsmn_aec.linear2(x3)).transpose(1, 2)

        # 4. Apply one broadcast mask to the packed complex spectrum and synthesize the final waveform.
        mask_frames = self.mask_frames if self.mask_frames else temp_aec_spectrum_A2.shape[-1]
        masked_spectrum = temp_aec_spectrum_A2.reshape(
            batch, 2, NFFT_A2 // 2 + 1, mask_frames) * masks.unsqueeze(1)
        audio = self.custom_istft_A2(masked_spectrum.reshape(
            batch, 2 * (NFFT_A2 // 2 + 1), mask_frames))
        if self.output_resample_before_pcm:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            audio = audio * self.output_pcm_scale      # [-1, 1] -> int16 PCM; F16/F32 outputs stay normalized.
        if self.output_resample_after_pcm:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        if self.use_batch_fold:
            audio = audio.reshape(1, 1, self.static_batch * self.static_audio_length)
        if "int" in OUT_AUDIO_DTYPE.lower():
            audio = audio.clamp(min=self.output_min, max=self.output_max).to(torch.int16)
        elif "32" not in OUT_AUDIO_DTYPE:
            audio = audio.to(torch.float16)
        if self.output_vad_result:
            return audio, vad_results
        return audio




def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_DFSMN_ONNX_AEC.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


def _remove_temporary_export_artifacts():
    for temporary_path in temporary_export_paths:
        temporary_path.unlink(missing_ok=True)
    print('Removed temporary export artifacts; retained the final ONNX model and metadata carrier.')


_temporary_artifact_cleanup = atexit.register(_remove_temporary_export_artifacts)


print('Export start ...')
export_directory = Path(onnx_model_A).expanduser().resolve().parent
export_directory.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    prefix='DFSMN_AEC.', suffix='.onnx', dir=export_directory, delete=False
) as temporary_export:
    temporary_model_path = temporary_export.name
temporary_export_paths.update((
    Path(temporary_model_path),
    Path(metadata_path_for_model(temporary_model_path)),
))
with torch.inference_mode():
    custom_stft_A2 = STFT_Process(
        model_type='stft_B_packed', n_fft=NFFT_A2, hop_len=HOP_LENGTH_A,
        win_length=WINDOW_LENGTH_A, max_frames=0, window_type=WINDOW_TYPE,
        center_pad=False, pad_mode='constant',
        input_scale=1.0).eval()
    custom_istft_A2 = STFT_Process(
        model_type='istft_B_packed', n_fft=NFFT_A2, hop_len=HOP_LENGTH_A,
        win_length=WINDOW_LENGTH_A, max_frames=MASK_FRAMES_A2,
        window_type=WINDOW_TYPE, center_pad=False, pad_mode='constant',
        static_frames=MASK_FRAMES_A2,
        output_length=MODEL_AUDIO_LENGTH,
        output_scale=1.0).eval()
    # NKF consumes (real, imag) tuples; the other backends use packed spectra.
    backend_packed = LIGHT_AEC_MODEL != 'NKF'
    custom_stft_B = STFT_Process(
        model_type='stft_B_packed' if backend_packed else 'stft_B',
        n_fft=NFFT_B, hop_len=HOP_LENGTH_B, win_length=WINDOW_LENGTH_B,
        max_frames=0, window_type=WINDOW_TYPE_B, center_pad=True, pad_mode='constant',
        input_scale=1.0).eval()
    custom_istft_B = STFT_Process(
        model_type='istft_B_packed' if backend_packed else 'istft_B',
        n_fft=NFFT_B, hop_len=HOP_LENGTH_B, win_length=WINDOW_LENGTH_B,
        max_frames=BACKEND_FRAMES_B, window_type=WINDOW_TYPE_B, center_pad=True, pad_mode='constant',
        static_frames=BACKEND_FRAMES_B, output_length=MODEL_AUDIO_LENGTH,
        output_scale=1.0).eval()
    custom_istft_B.forward_packed = custom_istft_B.forward
    m = light_aec_module

    if LIGHT_AEC_MODEL == 'SDAEC':
        iccrn = m.NET(max_frames=BACKEND_FRAMES_B, custom_istft=custom_istft_B)
        iccrn.load_state_dict(torch.load(project_path_B + '/Model/ICCRN.ckpt', map_location='cpu'), strict=False)
        for module in iccrn.modules():
            if isinstance(module, m.LayerNorm):
                module.fuse_var_scale_()
        iccrn.prepare_for_export_()
        alpha_predictor = m.AlphaPredictor(m.ALPHA_K)
        alpha_predictor.load_state_dict(torch.load(project_path_B + '/Model/alpha.ckpt', map_location='cpu'), strict=False)
        light_aec = m.SDAEC(
            iccrn.float().eval(), alpha_predictor.float().eval(), custom_stft_B, NFFT_B, m.ALPHA_K,
            BACKEND_FRAMES_B, MODEL_SAMPLE_RATE, MODEL_SAMPLE_RATE,
            auto_delay_align=AUTO_DELAY_ALIGN, num_tde_lags=m.NUM_TDE_LAGS, target_tde_lag=TARGET_TDE_LAG)
    elif LIGHT_AEC_MODEL == 'Deep_Echo':
        iccrn = m.NET(
            order=ECHO_ORDER, custom_istft=custom_istft_B,
            batch_size=MODEL_BATCH, frames=BACKEND_FRAMES_B)
        checkpoint_path = Path(project_path_B).expanduser().resolve() / 'model.ckpt'
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        iccrn.load_state_dict(m.extract_state_dict(checkpoint), strict=False)
        m.fuse_layer_norm_scales_(iccrn)
        light_aec = m.DeepEchoAEC(
            iccrn.float().eval(), custom_stft_B, MODEL_SAMPLE_RATE, MODEL_SAMPLE_RATE,
            False, 0, MODEL_BATCH, MODEL_AUDIO_LENGTH, BACKEND_FRAMES_B,
            input_scale_folded=False, auto_delay_align=AUTO_DELAY_ALIGN, target_tde_lag=TARGET_TDE_LAG)
    elif LIGHT_AEC_MODEL == 'RTAEC':
        rtaec_model = m.RTAECModel(
            custom_istft_B,
            output_length=MODEL_AUDIO_LENGTH,
            auto_delay_align=AUTO_DELAY_ALIGN,
            target_lag=TARGET_TDE_LAG,
        ).float().eval()
        light_aec = m.RTAEC(
            rtaec_model, custom_stft_B, MODEL_SAMPLE_RATE, MODEL_SAMPLE_RATE,
            False, 0, MODEL_AUDIO_LENGTH, MODEL_AUDIO_LENGTH)
    else:
        # The batch-fold path also covers a single window, so the window count always rides in dim0.
        light_aec = m.NKF(
            L=FILTER_ORDER, fc_dim=m.FC_DIM, rnn_layers=m.RNN_LAYERS, rnn_dim=m.RNN_DIM,
            custom_stft=custom_stft_B, custom_istft=custom_istft_B, max_frames=BACKEND_FRAMES_B,
            in_sample_rate=MODEL_SAMPLE_RATE, out_sample_rate=MODEL_SAMPLE_RATE,
            use_batch_fold=True, fold_window=MODEL_AUDIO_LENGTH,
            auto_delay_align=AUTO_DELAY_ALIGN, target_tde_lag=TARGET_TDE_LAG)
        light_aec = m.load_nkf_weights(light_aec, torch.load(m.checkpoint_path, map_location='cpu'))
        light_aec.cache_export_constants_()
    light_aec = light_aec.float().eval()

    dfsmn_pipeline = pipeline(
        Tasks.acoustic_echo_cancellation,
        model=project_path_A,
        device='cpu',
        trust_remote_code=True
    )

    dfsmn_aec = DFSMN_AEC(
        dfsmn_pipeline,
        light_aec=light_aec,
        light_aec_type=LIGHT_AEC_MODEL,
        custom_stft_A2=custom_stft_A2,
        custom_istft_A2=custom_istft_A2,
        nfft_A=NFFT_A,
        win_length_A=WINDOW_LENGTH_A,
        hop_length_A=HOP_LENGTH_A,
        pre_emphasis=PRE_EMPHASIZE,
        in_sample_rate=IN_SAMPLE_RATE,
        out_sample_rate=OUT_SAMPLE_RATE,
        n_mels=N_MELS,
        use_batch_fold=USE_BATCH_FOLD,
        fold_window=FOLD_WINDOW_LENGTH,
        static_batch=MODEL_BATCH,
        static_audio_length=MODEL_AUDIO_LENGTH,
        mask_frames=MASK_FRAMES_A2,
        output_vad_result=OUTPUT_VAD_RESULT,
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
        dfsmn_aec,
        (near_end_audio, far_end_audio),
        temporary_model_path,
        input_names=['near_end_audio', 'far_end_audio'],
        output_names=(['aec_audio', 'vad_results'] if OUTPUT_VAD_RESULT else ['aec_audio']),
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del dfsmn_aec
    del dfsmn_pipeline
    del light_aec
    del near_end_audio
    del far_end_audio
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="DFSMN_AEC", task="aec", model_family="dfsmn_aec",
    max_dynamic_audio_seconds=30, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=2, feature_kind="kaldi_fbank_stft_aec", center_pad=False, pad_mode="constant",
    extra={
        "light_aec_model": LIGHT_AEC_MODEL, "n_mels": N_MELS, "nfft_a": NFFT_A, "nfft_a2": NFFT_A2,
        "window_length_a": WINDOW_LENGTH_A, "hop_length_a": HOP_LENGTH_A, "nfft_b": NFFT_B,
        "window_length_b": WINDOW_LENGTH_B, "hop_length_b": HOP_LENGTH_B, "window_type_b": WINDOW_TYPE_B,
        "preemphasize": PRE_EMPHASIZE, "alpha_k": ALPHA_K,
        "echo_order": ECHO_ORDER, "filter_order": FILTER_ORDER,
        "output_vad_result": OUTPUT_VAD_RESULT, "num_outputs": 2 if OUTPUT_VAD_RESULT else 1,
        "output_frame_shift_seconds": HOP_LENGTH_A / MODEL_SAMPLE_RATE,
        "output_frame_shift_samples": HOP_LENGTH_A, "fbank_window_length_samples": WINDOW_LENGTH_A,
        "speaking_score": SPEAKING_SCORE, "silence_score": SILENCE_SCORE,
        "look_ahead_seconds": LOOK_AHEAD, "fusion_threshold_seconds": FUSION_THRESHOLD,
        "min_speech_duration_seconds": MIN_SPEECH_DURATION,
        "auto_delay_align": AUTO_DELAY_ALIGN, "delay_align": "gcc_phat_segment_tracking",
        "target_tde_lag": TARGET_TDE_LAG,
    },
)
stamp_export_metadata(temporary_model_path, model_metadata, OPSET)
shutil.copyfile(temporary_model_path, onnx_model_A)
export_metadata_carrier(onnx_model_Metadata, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
try:
    _run_inference_demo()
finally:
    # The raw export and its metadata carrier are intermediate artifacts. Retain the
    # separately exported final metadata carrier beside the final ONNX model so both
    # inference and Optimize_ONNX.py can copy/load the required runtime metadata.
    atexit.unregister(_remove_temporary_export_artifacts)
    _remove_temporary_export_artifacts()
