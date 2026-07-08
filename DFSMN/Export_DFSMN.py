import gc
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from STFT_Process import STFT_Process                                                      # The custom STFT/ISTFT can be exported in ONNX format.

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, metadata_path_for_model, stamp_export_metadata


model_path            = "/home/DakeQQ/Downloads/speech_dfsmn_ans_psm_48k_causal"         # The DFSMN download path.
parent_path           = Path(__file__).resolve().parent                              # The folder that contains this script.
onnx_model_A          = str(parent_path / "DFSMN_ONNX" / "DFSMN.onnx")              # The exported onnx model path.
onnx_model_Metadata   = str(metadata_path_for_model(onnx_model_A))                   # The metadata carrier onnx model path.


DYNAMIC_AXES          = False                   # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
OPSET                 = 20
MODEL_SAMPLE_RATE     = 48000                   # DFSMN runs at 48kHz internally.
IN_SAMPLE_RATE        = 48000                   # [8000, 16000, 22500, 24000, 44000, 48000]; input audio sample rate.
OUT_SAMPLE_RATE       = 48000                   # [8000, 16000, 22500, 24000, 44000, 48000]; output audio sample rate.
INPUT_AUDIO_LENGTH    = 96000                   # The maximum input audio length in IN_SAMPLE_RATE samples.
WINDOW_TYPE           = 'hamming'               # STFT analysis / Kaldi fbank window (symmetric hamming, matches torch.hamming_window(periodic=False))
ISTFT_WINDOW_TYPE     = 'hamming_periodic'      # ISTFT synthesis window: the original post-process uses librosa.istft(window='hamming') == periodic hamming
N_MELS                = 120                     # Number of Mel bands to generate in the Mel-spectrogram
NFFT_STFT             = 1920                    # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH         = 1920                    # Length of windowing, edit it carefully.
HOP_LENGTH            = 960                     # Number of samples between successive frames in the STFT
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH        = INPUT_AUDIO_LENGTH

IN_AUDIO_DTYPE        = 'INT16'                 # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE       = 'INT16'                 # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
INV_INT16             = float(1.0 / 32768.0)

MODEL_AUDIO_LENGTH    = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE))
OUTPUT_AUDIO_LENGTH   = INPUT_AUDIO_LENGTH if DYNAMIC_AXES else int(round(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE))
INPUT_TO_MODEL_SCALE  = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
MODEL_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
INPUT_TO_OUTPUT_SCALE = float(OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
BATCH_WINDOW_SECONDS  = 1.5                     # When the configured input audio length is >= this many seconds, the audio is folded into fixed-length windows and batch-processed together to accelerate inference.
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH  # Per-window model-rate length, rounded UP to a HOP multiple. center=False needs (W-NFFT)%HOP==0 -> holds since W%HOP==0 and NFFT_STFT%HOP==0.
USE_BATCH_FOLD        = False                   # If true, batch-fold always enabled (requires DYNAMIC_AXES=False + IN==MODEL==OUT rate + INPUT_AUDIO_LENGTH >= BATCH_WINDOW_SECONDS*IN_SAMPLE_RATE).
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_WINDOW_LENGTH - 1) // FOLD_WINDOW_LENGTH) * FOLD_WINDOW_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH  # Static ONNX input length rounded up to whole windows; the tail is padded OUTSIDE the model (numpy) by the windowing loop.

# ---- Kaldi-fbank (feature extractor) parameters — must match the original pipeline exactly ----
KALDI_FRAME_LENGTH    = 1920                    # 40 ms @ 48 kHz analysis frame (samples)
KALDI_HOP_LENGTH      = 960                     # 20 ms @ 48 kHz frame shift (samples)
KALDI_NFFT            = 2048                    # next power of two of the 40 ms frame (Kaldi round_to_power_of_two)
PREEMPH_COEFF         = 0.97                    # Kaldi pre-emphasis coefficient

# Frame count after the mask STFT (center=False / snip-edges, computed in the 48 kHz domain).
# In fold mode this is the PER-WINDOW frame count (also sizes the ISTFT COLA bound per window).
STFT_SIGNAL_LENGTH    = ((FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else MODEL_AUDIO_LENGTH) - NFFT_STFT) // HOP_LENGTH + 1   # frames (center=False)
MAX_SIGNAL_LENGTH     = 2048 if DYNAMIC_AXES else STFT_SIGNAL_LENGTH        # ISTFT static frame bound (center=False)


class DFSMN(torch.nn.Module):
    def __init__(self, dfsmn, stft_model, istft_model, nfft_stft, n_mels, in_sample_rate, out_sample_rate, use_batch_fold=False, fold_window=0):
        super(DFSMN, self).__init__()
        self.istft_model = istft_model
        self.inv_int16 = torch.tensor([INV_INT16], dtype=torch.float16 if OUT_AUDIO_DTYPE == 'F16' else torch.float32)
        self.nfft_stft = nfft_stft
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.use_batch_fold = use_batch_fold          # Fold long audio into fixed windows and batch-process them together
        self.fold_window = fold_window                # Per-window length (model-rate samples) used when folding
        self.hop_len = KALDI_HOP_LENGTH
        self.kaldi_bins = KALDI_NFFT // 2 + 1                 # one-sided 2048-pt FFT bins (1025)
        self.log_eps = float(torch.finfo(torch.float32).eps)  # Kaldi log floor (numeric_limits<float>::epsilon)

        # ---- Kaldi log-mel-fbank feature extractor, folded into ONE Conv1d ----
        # Exactly reproduces torchaudio.compliance.kaldi.fbank(dither=0, frame_length=40ms,
        # frame_shift=20ms, num_mel_bins=120, window_type='hamming', sample_frequency=48000):
        #   per-frame DC removal -> 0.97 pre-emphasis -> symmetric hamming -> 2048-pt rfft power.
        # All of those per-frame steps are linear in the frame samples, so they collapse into a
        # single (2*1025, 1, 1920) convolution kernel (snip-edges == stride=hop, no padding).
        n = KALDI_FRAME_LENGTH
        win = torch.hamming_window(n, periodic=False, alpha=0.54, beta=0.46, dtype=torch.float64)
        dc_matrix = torch.eye(n, dtype=torch.float64) - (1.0 / n)              # subtract per-frame mean
        preemph = torch.eye(n, dtype=torch.float64)                           # y[i] = x[i] - c*x[i-1]
        rng = torch.arange(1, n)
        preemph[rng, rng - 1] = -PREEMPH_COEFF
        preemph[0, 0] = 1.0 - PREEMPH_COEFF                                   # replicate-pad first sample
        t = torch.arange(n, dtype=torch.float64).unsqueeze(0)                 # (1, n)
        freq = torch.arange(self.kaldi_bins, dtype=torch.float64).unsqueeze(1)  # (F, 1)
        omega = (2.0 * torch.pi / KALDI_NFFT) * freq * t                      # (F, n) 2048-pt DFT angles
        cos_win = torch.cos(omega) * win.unsqueeze(0)
        sin_win = -torch.sin(omega) * win.unsqueeze(0)
        preemph_dc = preemph @ dc_matrix                                      # (n, n)
        weight_real = cos_win @ preemph_dc                                    # (F, n)
        weight_imag = sin_win @ preemph_dc                                    # (F, n)
        fbank_kernel = torch.cat([weight_real, weight_imag], dim=0).unsqueeze(1).float()  # (2*kaldi_bins, 1, n)

        # ---- Fuse the Kaldi-fbank analysis and the mask-STFT into ONE Conv1d ----
        # Both are stride-960, length-1920, un-padded convolutions of the SAME audio, so their output
        # channels simply concatenate: [fbank_real | fbank_imag | stft_real | stft_imag]. A single conv
        # (+ one Split) replaces the two separate passes; each output channel is computed independently,
        # so the per-channel result is bit-identical to the un-fused version.
        self.stft_bins = nfft_stft // 2 + 1                   # one-sided 1920-pt mask-STFT bins (961)
        stft_kernel = stft_model.stft_kernel                  # (2*stft_bins, 1, n), built by STFT_Process
        assert stft_model.hop_len == self.hop_len and stft_kernel.shape[-1] == n, \
            "Fusion requires the fbank and mask-STFT convolutions to share stride and kernel length."
        self.register_buffer('analysis_conv_weight', torch.cat([fbank_kernel, stft_kernel.float()], dim=0))

        # Kaldi triangular mel filterbank (num_mel_bins, padded_window_size//2 + 1), zero-padded right.
        mel_banks, _ = torchaudio.compliance.kaldi.get_mel_banks(
            n_mels, KALDI_NFFT, 48000.0, 20.0, 0.0, 100.0, -500.0, 1.0)        # (120, 1024)
        mel_banks = torch.nn.functional.pad(mel_banks, (0, 1), mode='constant', value=0.0)  # (120, 1025)
        self.register_buffer('mel_banks', mel_banks.unsqueeze(0).float())     # (1, 120, 1025)

        # Pre-fuse the DfsmnAns mask network into channels-first convolution buffers.
        self._build_dfsmn_buffers(dfsmn)

    def _build_dfsmn_buffers(self, dfsmn):
        """Pre-fuse the DfsmnAns mask network (linear1 -> ReLU -> fsmn_depth x UniDeepFsmn ->
        linear2 -> Sigmoid) into channels-first convolution weights so the whole mask path
        runs as 1x1 / depthwise Conv1d with no per-layer unsqueeze/permute/squeeze (the
        original UniDeepFsmn.forward emits four such shape ops per layer, plus a transpose on
        the network input and output). Each pointwise AffineTransform becomes one Conv1d with
        its bias folded in; every causal FSMN memory convolution also absorbs its inner
        residual (out = p1 + conv(p1)) by adding 1 to the current-frame tap, so only the outer
        residual add survives per layer."""
        self.register_buffer('lin1_w', dfsmn.linear1.linear.weight.detach().unsqueeze(-1).contiguous())   # (256, 120, 1)
        self.register_buffer('lin1_b', dfsmn.linear1.linear.bias.detach().contiguous())                    # (256,)
        self.register_buffer('lin2_w', dfsmn.linear2.linear.weight.detach().unsqueeze(-1).contiguous())   # (961, 256, 1)
        self.register_buffer('lin2_b', dfsmn.linear2.linear.bias.detach().contiguous())                    # (961,)
        uf0 = dfsmn.deepfsmn[0]
        self.fsmn_depth = len(dfsmn.deepfsmn)
        self.fsmn_groups = uf0.output_dim
        self.register_buffer('fsmn_pad', torch.zeros((1, self.fsmn_groups, uf0.lorder - 1), dtype=torch.float32))
        self._uf_lin_w, self._uf_lin_b, self._uf_proj_w, self._uf_conv_w = [], [], [], []
        for i, uf in enumerate(dfsmn.deepfsmn):
            conv_w = uf.conv1.weight.detach().squeeze(-1).clone()   # (256, 1, lorder)
            conv_w[:, 0, -1] += 1.0                                 # fold inner residual p1 + conv(p1)
            self.register_buffer(f'uf_lin_w_{i}', uf.linear.weight.detach().unsqueeze(-1).contiguous())    # (256, 256, 1)
            self.register_buffer(f'uf_lin_b_{i}', uf.linear.bias.detach().contiguous())                     # (256,)
            self.register_buffer(f'uf_proj_w_{i}', uf.project.weight.detach().unsqueeze(-1).contiguous())   # (256, 256, 1)
            self.register_buffer(f'uf_conv_w_{i}', conv_w.contiguous())                                     # (256, 1, lorder)
            self._uf_lin_w.append(getattr(self, f'uf_lin_w_{i}'))
            self._uf_lin_b.append(getattr(self, f'uf_lin_b_{i}'))
            self._uf_proj_w.append(getattr(self, f'uf_proj_w_{i}'))
            self._uf_conv_w.append(getattr(self, f'uf_conv_w_{i}'))

    def forward(self, audio):
        audio = audio.float()  # int16 scale (matches original sf.read * 32768); no /32768, no global DC removal
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
        # One fused analysis conv -> [fbank_real | fbank_imag | stft_real | stft_imag]
        # (center=False; symmetric-hamming analysis for both the Kaldi fbank and the mask STFT,
        #  periodic-hamming synthesis happens later in the ISTFT).
        if self.use_batch_fold:
            # Input length is already a whole number of windows (tail padded OUTSIDE the model in
            # numpy by the windowing loop), so fold (1, 1, num_window*W) -> (num_window, 1, W) and
            # run the whole conv-only graph batched. No padding op inside the graph.
            audio = audio.reshape(-1, 1, self.fold_window)
        analysis = torch.nn.functional.conv1d(audio, self.analysis_conv_weight, stride=self.hop_len)
        real_fb, imag_fb, real_part, imag_part = torch.split(
            analysis, [self.kaldi_bins, self.kaldi_bins, self.stft_bins, self.stft_bins], dim=1)
        # Kaldi log-mel-fbank features kept channels-first (1, 120, frames) so the inlined mask
        # network below runs as 1x1 / depthwise convolutions without any per-frame transpose.
        power = real_fb * real_fb + imag_fb * imag_fb
        x = torch.matmul(self.mel_banks, power).clamp(min=self.log_eps).log()      # (1, 120, frames)

        # ---- Inlined DfsmnAns mask network (channels-first) ----
        # linear1 -> ReLU -> fsmn_depth x UniDeepFsmn -> linear2 -> Sigmoid, expanded to leaf
        # Conv1d ops on the pre-fused buffers built in __init__. The pointwise affines run as
        # 1x1 convolutions and the causal FSMN memory runs as one depthwise Conv1d whose weight
        # already carries the inner residual, so the feature stays in (1, channels, frames).
        x = F.relu(F.conv1d(x, self.lin1_w, self.lin1_b))
        for i in range(self.fsmn_depth):
            f1 = F.relu(F.conv1d(x, self._uf_lin_w[i], self._uf_lin_b[i]))
            p1 = F.conv1d(f1, self._uf_proj_w[i], None)
            mem = F.conv1d(torch.cat((self.fsmn_pad.expand(p1.shape[0], -1, -1), p1), dim=2), self._uf_conv_w[i], None, groups=self.fsmn_groups)
            x = x + mem
        mask = torch.sigmoid(F.conv1d(x, self.lin2_w, self.lin2_b))                # (1, 961, frames)
        # ---- end inlined mask network ----

        # Apply the mask to the complex STFT and reconstruct.
        real_part = real_part * mask
        imag_part = imag_part * mask
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
    inference_script = Path(__file__).resolve().with_name('Inference_DFSMN_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


print('Export start ...')
Path(onnx_model_A).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH,hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=False, pad_mode='constant').eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=ISTFT_WINDOW_TYPE, center_pad=False, pad_mode='constant').eval()
    dfsmn = pipeline(
        Tasks.acoustic_noise_suppression,
        model=model_path,
        device='cpu'
    ).model
    dfsmn = DFSMN(dfsmn, custom_stft, custom_istft, NFFT_STFT, N_MELS, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, use_batch_fold=USE_BATCH_FOLD, fold_window=FOLD_WINDOW_LENGTH)
    if "32" in IN_AUDIO_DTYPE:
        IN_TORCH_DTYPE = torch.float32
    elif "int" in IN_AUDIO_DTYPE.lower():
        IN_TORCH_DTYPE = torch.int16
    else:
        IN_TORCH_DTYPE = torch.float16
    audio = torch.ones((1, 1, EXPORT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
    torch.onnx.export(
        dfsmn,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'out_audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del dfsmn
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
model_metadata = build_audio_metadata_from_globals(
    globals(), producer=Path(__file__).name, model_name="DFSMN", task="denoise", model_family="dfsmn",
    max_dynamic_audio_seconds=6, normalize_audio_default=False, input_channels=1, output_channels=1,
    num_audio_inputs=1, feature_kind="kaldi_fbank_stft", center_pad=False, pad_mode="constant",
    extra={
        "n_mels": N_MELS, "nfft_stft": NFFT_STFT, "kaldi_nfft": KALDI_NFFT,
        "kaldi_frame_length": KALDI_FRAME_LENGTH, "kaldi_hop_length": KALDI_HOP_LENGTH,
        "preemph_coeff": PREEMPH_COEFF, "istft_window_type": ISTFT_WINDOW_TYPE,
    },
)
stamp_export_metadata(onnx_model_A, model_metadata, OPSET)
print(f"Metadata saved to: {onnx_model_Metadata}")
print('\nExport done!')
_run_inference_demo()
