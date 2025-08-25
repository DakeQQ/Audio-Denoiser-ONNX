import gc
import time
import yaml
from ml_collections import ConfigDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import soundfile as sf
from pydub import AudioSegment
from modeling_modified.mel_band_roformer import MelBandRoformer
from STFT_Process import STFT_Process                                                             # The custom STFT/ISTFT can be exported in ONNX format.


project_path = "/home/DakeQQ/Downloads/Mel-Band-Roformer-Vocal-Model-main"                        # The Mel-Band-Roformer GitHub project path.
model_path = "/home/DakeQQ/Downloads/Mel-Band-Roformer-Vocal-Model-main/MelBandRoformer.ckpt"     # The model download path.
onnx_model_A = "/home/DakeQQ/Downloads/MelBandRoformer_ONNX/MelBandRoformer.onnx"                 # The exported onnx model path.
test_noisy_audio = "./test.wav"                                                                   # The noisy audio path.
save_denoised_audio = "./test_denoised.wav"                                                       # The output denoised audio path.

ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
KEEP_ORIGINAL_SAMPLE_RATE = True        # If False, the model outputs audio at 16kHz; otherwise, it uses the original sample rate.
SAMPLE_RATE = 44100                     # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH = 44100              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 22050 and less than 362800. Higher values yield better quality but time consume. It is better to set an integer multiple of the HOP_LENGTH value.
MAX_SIGNAL_LENGTH = 2048 if DYNAMIC_AXES else (INPUT_AUDIO_LENGTH // 50 + 1)  # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
NFFT = 2048                             # Number of FFT components for the STFT process
WINDOW_LENGTH = 2048                    # Length of windowing, edit it carefully.
HOP_LENGTH = 441                        # Number of samples between successive frames in the STFT
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.


SAMPLE_RATE_SCALE = float(44100.0 / SAMPLE_RATE)


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class MelBandRoformer_Modified(torch.nn.Module):
    """
    Mono-only wrapper around the original network.
    - Accepts (B, C, T) int16 audio. If C=2, downmix to mono.
    - Performs STFT -> Model -> iSTFT for a single channel only.
    - Returns (B, 1, T) int16 audio.
    """
    def __init__(self, mel_band_roformer, stft_model, istft_model, nfft, max_signal_len, sample_rate):
        super(MelBandRoformer_Modified, self).__init__()
        self.mel_band_roformer = mel_band_roformer
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.inv_int16 = float(1.0 / 32768.0)
        # Mono zeros buffer over frequency bins (complex last dim size 2: real/imag)
        self.zeros = torch.zeros((1, 1, (nfft // 2 + 1), max_signal_len, 2), dtype=torch.int8)
        self.sample_rate = sample_rate

    def forward(self, audio):
        # audio: (B, C, T) int16
        audio = audio.float()

        # Optional resample to 44.1k pipeline rate
        if SAMPLE_RATE_SCALE < 1.0:
            audio *= self.inv_int16
            if self.sample_rate != 44100:
                audio = torch.nn.functional.interpolate(
                    audio, scale_factor=SAMPLE_RATE_SCALE, mode='linear', align_corners=True
                )
        else:
            if self.sample_rate != 44100:
                audio = torch.nn.functional.interpolate(
                    audio, scale_factor=SAMPLE_RATE_SCALE, mode='linear', align_corners=True
                )
            audio *= self.inv_int16

        # STFT for mono
        real, imag = self.stft_model(audio, 'constant')  # (B, F, T)
        stft_repr = torch.stack((real, imag), dim=-1)    # (B, F, T, 2)
        stft_repr = stft_repr.unsqueeze(1)               # (B, 1, F, T, 2)

        # Model forward (mono)
        real_m, imag_m = self.mel_band_roformer(stft_repr, self.zeros)

        # iSTFT for mono
        audio = self.istft_model(real_m, imag_m)       # (B, 1, T)

        # Resample back to original rate if needed and scale back to int16 range
        if SAMPLE_RATE_SCALE < 1.0:
            audio *= 32767.0
            if KEEP_ORIGINAL_SAMPLE_RATE and self.sample_rate != 44100:
                audio = torch.nn.functional.interpolate(
                    audio, scale_factor=1.0 / SAMPLE_RATE_SCALE, mode='linear', align_corners=True
                )
        else:
            if KEEP_ORIGINAL_SAMPLE_RATE and self.sample_rate != 44100:
                audio = torch.nn.functional.interpolate(
                    audio, scale_factor=1.0 / SAMPLE_RATE_SCALE, mode='linear', align_corners=True
                )
            audio *= 32767.0

        return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)


def _unwrap_state_dict(state):
    # support common checkpoint wrappers or DDP "module." prefix
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[len('module.'):]] = v
            else:
                new_state[k] = v
        return new_state
    return state


def convert_stereo_to_mono_weights(stereo_model: MelBandRoformer, mono_model: MelBandRoformer):
    """
    Fold a stereo checkpoint loaded in stereo_model into mono_model:
    - BandSplit RMSNorm gamma: sum L and R per (real, imag) -> keep (real, imag)
    - BandSplit Linear: sum corresponding L/R columns; bias copied
    - MaskEstimator MLP last Linear (pre-GLU): sum corresponding L/R rows per (real, imag) in both GLU halves; bias summed similarly
    All other matching-shaped params are copied directly.
    """
    with torch.no_grad():
        # Copy all matching params first
        mono_sd = mono_model.state_dict()
        stereo_sd = stereo_model.state_dict()
        for k, v in mono_sd.items():
            if k in stereo_sd and stereo_sd[k].shape == v.shape:
                mono_sd[k] = stereo_sd[k].clone()
        mono_model.load_state_dict(mono_sd, strict=False)

        # Fold BandSplit per band
        for b in range(len(mono_model.band_split.to_features)):
            st_feat = stereo_model.band_split.to_features[b]
            mo_feat = mono_model.band_split.to_features[b]
            fi = int(mono_model.num_freqs_per_band[b].item())  # frequencies in this band

            # RMSNorm gamma: (4fi,) -> (2fi,) by summing L/R for real,imag
            gamma_st = st_feat[0].gamma.data  # (4fi,)
            gamma_st_v = gamma_st.view(fi, 4)  # [real_L, imag_L, real_R, imag_R]
            gamma_m = torch.stack([
                gamma_st_v[:, 0] + gamma_st_v[:, 2],  # real
                gamma_st_v[:, 1] + gamma_st_v[:, 3],  # imag
            ], dim=-1).reshape(-1)
            mo_feat[0].gamma.data.copy_(gamma_m)

            # Linear weight: (dim, 4fi) -> (dim, 2fi); bias copied
            Wst = st_feat[1].weight.data
            dim_out = Wst.shape[0]
            Wst_v = Wst.view(dim_out, fi, 4)
            Wm_v = torch.empty((dim_out, fi, 2), dtype=Wst.dtype, device=Wst.device)
            Wm_v[:, :, 0] = Wst_v[:, :, 0] + Wst_v[:, :, 2]  # real L+R
            Wm_v[:, :, 1] = Wst_v[:, :, 1] + Wst_v[:, :, 3]  # imag L+R
            mo_feat[1].weight.data.copy_(Wm_v.reshape(dim_out, 2 * fi))
            mo_feat[1].bias.data.copy_(st_feat[1].bias.data)

        # Fold MaskEstimator final linear for each band
        for st_me, mo_me in zip(stereo_model.mask_estimators, mono_model.mask_estimators):
            for b_idx, (st_mlp, mo_mlp) in enumerate(zip(st_me.to_freqs, mo_me.to_freqs)):
                # The first element is the MLP (Sequential of Linear/.../Linear); GLU is at index 1
                st_linears = [m for m in st_mlp[0] if isinstance(m, nn.Linear)]
                mo_linears = [m for m in mo_mlp[0] if isinstance(m, nn.Linear)]
                st_last = st_linears[-1]
                mo_last = mo_linears[-1]

                fi = int(mono_model.num_freqs_per_band[b_idx].item())

                # Stereo last linear: out_features = 8fi; Mono: out_features = 4fi (pre-GLU)
                Wst = st_last.weight.data  # (8fi, hidden)
                bst = st_last.bias.data    # (8fi,)
                hidden = Wst.shape[1]

                # Split into GLU halves (A,B), each 4fi
                Wst_v = Wst.view(2, 4 * fi, hidden)
                bst_v = bst.view(2, 4 * fi)

                def fold_rows(W_part):
                    # (4fi, hidden) -> (2fi, hidden) by summing L/R in each (real, imag)
                    Wp = W_part.view(fi, 4, hidden)  # [real_L, imag_L, real_R, imag_R]
                    out = torch.empty(fi, 2, hidden, dtype=W_part.dtype, device=W_part.device)
                    out[:, 0] = Wp[:, 0] + Wp[:, 2]  # real
                    out[:, 1] = Wp[:, 1] + Wp[:, 3]  # imag
                    return out.view(2 * fi, hidden)

                def fold_bias(b_part):
                    bp = b_part.view(fi, 4)
                    out = torch.stack([bp[:, 0] + bp[:, 2], bp[:, 1] + bp[:, 3]], dim=-1)
                    return out.view(2 * fi)

                W_A_new = fold_rows(Wst_v[0])
                W_B_new = fold_rows(Wst_v[1])
                bA_new = fold_bias(bst_v[0])
                bB_new = fold_bias(bst_v[1])

                W_new = torch.cat([W_A_new, W_B_new], dim=0)  # (4fi, hidden)
                b_new = torch.cat([bA_new, bB_new], dim=0)    # (4fi,)

                mo_last.weight.data.copy_(W_new)
                mo_last.bias.data.copy_(b_new)

    return mono_model


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()

    # Load config
    with open(project_path + "/configs/config_vocals_mel_band_roformer.yaml") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    # Build and load a temporary stereo model strictly from checkpoint
    stereo_cfg = dict(config.model)
    stereo_cfg['stereo'] = True
    stereo_model = MelBandRoformer(**stereo_cfg)
    state = torch.load(model_path, map_location=torch.device('cpu'))
    state = _unwrap_state_dict(state)
    stereo_model.load_state_dict(state, strict=False)
    stereo_model.eval()

    # Build mono model and convert stereo weights into mono
    mono_cfg = dict(config.model)
    mono_cfg['stereo'] = False
    base_mono_model = MelBandRoformer(**mono_cfg)
    base_mono_model = convert_stereo_to_mono_weights(stereo_model, base_mono_model).eval()

    # Wrap mono model for single-channel inference
    mel_band_roformer = MelBandRoformer_Modified(base_mono_model, custom_stft, custom_istft, NFFT, MAX_SIGNAL_LENGTH, SAMPLE_RATE)

    # Mono input for export
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)

    torch.onnx.export(
        mel_band_roformer,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del mel_band_roformer, base_mono_model, stereo_model, audio, custom_stft, custom_istft
    gc.collect()
print('\nExport done!\n\nStart to run MelBandRoformer by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4
session_opts.inter_op_num_threads = 0
session_opts.intra_op_num_threads = 0
session_opts.enable_cpu_mem_arena = True
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A0 = ort_session_A.get_inputs()[0].name
out_name_A0 = ort_session_A.get_outputs()[0].name

# Load the input audio
print(f"\nTest Input Audio: {test_noisy_audio}")
audio = np.array(AudioSegment.from_file(test_noisy_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(SAMPLE_RATE, audio_len)
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int) & (KEEP_ORIGINAL_SAMPLE_RATE):
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
inv_audio_len = float(100.0 / aligned_len)


if SAMPLE_RATE != 44100 and not KEEP_ORIGINAL_SAMPLE_RATE:
    SAMPLE_RATE = 44100
    audio_len = int(audio_len * SAMPLE_RATE_SCALE)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Run inference in parallel over segments
print("\nRunning the MelBandRoformer by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, audio))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.3f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:audio_len]
end_time = time.time()
print(f"Complete: 100.00%")

# Save the denoised wav (mono)
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
