#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STFT / ISTFT  —  ONNX-exportable Short-Time Fourier Transform.

This script:
  1. Builds a PyTorch model that performs STFT or ISTFT using Conv1d /
     ConvTranspose1d (no torch.stft / torch.istft at runtime).
  2. Exports the model to ONNX with optional dynamic axes.
  3. Validates the ONNX graph against torch.stft / torch.istft.
  4. Runs a round-trip (STFT → ISTFT) reconstruction test.
"""

import torch
import numpy as np
import onnxruntime as ort
from onnxslim import slim

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Configuration
# ═════════════════════════════════════════════════════════════════════════════

# -- Export settings --------------------------------------------------------
DYNAMIC_AXES = True          # True  → ONNX graph accepts variable-length audio
                             # False → fixed to INPUT_AUDIO_LENGTH
OPSET = 17                   # ONNX Runtime export setting

# -- FFT / framing parameters ----------------------------------------------
NFFT         = 400           # FFT size (number of frequency bins before folding)
WIN_LENGTH   = 400           # Analysis window length in samples (≤ NFFT)
HOP_LENGTH   = 160           # Hop (stride) between successive frames
WINDOW_TYPE  = 'hann'        # Window function: bartlett | blackman | hamming | hann | kaiser

# -- Padding ---------------------------------------------------------------
CENTER_PAD   = True          # True  → pad signal so frame centres align with sample indices
                             # False → no padding, first frame starts at sample 0
PAD_MODE     = 'constant'    # Padding style when CENTER_PAD is True: 'reflect' | 'constant'

# -- Audio dimensions (used for dummy tensors during export) ----------------
INPUT_AUDIO_LENGTH = 16000   # Length of the test / dummy waveform (samples)
MAX_SIGNAL_LENGTH  = 2048    # Upper-bound hint for frame-axis dim

# -- Model variants to export ----------------------------------------------
#    stft_A  / istft_A  →  real-only STFT  /  magnitude+phase ISTFT
#    stft_B  / istft_B  →  real+imag STFT  /  real+imag       ISTFT
STFT_TYPE  = "stft_B"
ISTFT_TYPE = "istft_B"

# -- Derived export paths ---------------------------------------------------
export_path_stft  = f"{STFT_TYPE}.onnx"
export_path_istft = f"{ISTFT_TYPE}.onnx"


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Derived constants & helpers
# ═════════════════════════════════════════════════════════════════════════════

# Clamp parameters so they never exceed input length or each other.
NFFT       = min(NFFT, INPUT_AUDIO_LENGTH)
WIN_LENGTH = min(WIN_LENGTH, NFFT)
HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)
HALF_NFFT  = NFFT // 2       # Number of positive-frequency bins (excluding DC & Nyquist)

# Number of STFT frames produced for INPUT_AUDIO_LENGTH samples.
if CENTER_PAD:
    # Center padding adds HALF_NFFT zeros on each side → one extra frame.
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
else:
    STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - NFFT) // HOP_LENGTH + 1

# -- Window function registry ----------------------------------------------
WINDOW_FUNCTIONS = {
    'bartlett': lambda L: torch.bartlett_window(L, periodic=True),
    'blackman': lambda L: torch.blackman_window(L, periodic=True),
    'hamming':  lambda L: torch.hamming_window(L,  periodic=True),
    'hann':     lambda L: torch.hann_window(L,     periodic=True),
    'kaiser':   lambda L: torch.kaiser_window(L,   periodic=True, beta=12.0)
}
DEFAULT_WINDOW_FN = lambda L: torch.hann_window(L, periodic=True)


def create_padded_window(win_length: int, n_fft: int, window_type: str) -> torch.Tensor:
    """
    Create a window of length *n_fft*, center-padding or cropping as needed
    so it matches the behavior of ``torch.stft`` (which always zero-pads a
    shorter window to n_fft internally).
    """
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    win = win_fn(win_length).float()

    if win_length == n_fft:
        return win

    if win_length < n_fft:
        # Zero-pad symmetrically to reach n_fft.
        pad_total = n_fft - win_length
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.cat([torch.zeros(pad_left), win, torch.zeros(pad_right)])

    # win_length > n_fft: centre-crop.
    start = (win_length - n_fft) // 2
    return win[start : start + n_fft]


def get_raw_window(win_length: int, window_type: str) -> torch.Tensor:
    """Return the raw (un-padded) window — used by ``torch.stft`` for reference tests."""
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    return win_fn(win_length).float()


# Pre-compute the padded window once for reuse.
WINDOW = create_padded_window(WIN_LENGTH, NFFT, WINDOW_TYPE)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  STFT / ISTFT Model
# ═════════════════════════════════════════════════════════════════════════════

class STFT_Process(torch.nn.Module):
    """
    Conv1d-based STFT / ConvTranspose1d-based ISTFT that exports cleanly to ONNX.

    Variants
    --------
    stft_A   → single Conv1d producing **real** part only.
    stft_B   → single Conv1d producing **real + imag** (split after convolution).
    istft_A  → accepts (magnitude, phase), reconstructs via ConvTranspose1d.
    istft_B  → accepts (real, imag),       reconstructs via ConvTranspose1d.
    """

    def __init__(
        self,
        model_type: str,
        n_fft: int       = NFFT,
        win_length: int  = WIN_LENGTH,
        hop_len: int     = HOP_LENGTH,
        max_frames: int  = MAX_SIGNAL_LENGTH,
        window_type: str = WINDOW_TYPE,
        center_pad: bool = True,
        pad_mode: str    = PAD_MODE
    ):
        super().__init__()

        self.model_type = model_type
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.half_n_fft = n_fft // 2
        self.center_pad = center_pad
        self.pad_mode   = pad_mode
        self.max_frames = max_frames

        F_bins = self.half_n_fft + 1          # Number of one-sided frequency bins
        window = create_padded_window(win_length, n_fft, window_type)

        self.register_buffer('ones', torch.ones(1, 1, max_frames, dtype=torch.float32))

        expected_len = torch.zeros(max_frames, dtype=torch.int32)
        for i in range(max_frames):
            expected_len[i] = self.n_fft + self.hop_len * (i - 1)
        if center_pad:
            expected_len = expected_len - self.half_n_fft
        self.register_buffer('expected_len', expected_len)

        # Pre-allocated zero buffer for constant-pad centre mode.
        if self.center_pad and self.pad_mode != 'reflect':
            self.register_buffer('padding_zero', torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32))

        # ── Build STFT convolution kernels ────────────────────────────────
        if model_type in ('stft_A', 'stft_B'):
            self._build_stft_kernels(n_fft, F_bins, window, model_type)

        # ── Build ISTFT transpose-convolution kernels ─────────────────────
        if model_type in ('istft_A', 'istft_B'):
            self._build_istft_kernels(n_fft, F_bins, window)

    # --------------------------------------------------------------------- #
    #  Kernel construction helpers                                          #
    # --------------------------------------------------------------------- #

    def _build_stft_kernels(self, n_fft, F_bins, window, model_type):
        """
        Pre-compute the DFT basis windowed by *window*, stored as Conv1d weights.

        For stft_A only the cosine (real) part is kept.
        For stft_B both cosine and sine parts are concatenated into one kernel
        so a single Conv1d + Split replaces two separate convolutions.
        """
        omega_factor = 2.0 * torch.pi / n_fft
        t = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)   # (1, n_fft)
        f = torch.arange(F_bins, dtype=torch.float32).unsqueeze(1)  # (F, 1)
        omega = omega_factor * f * t                                # (F, n_fft)

        windowed_cos = ( torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        windowed_sin = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

        if model_type == 'stft_A':
            self.register_buffer('stft_kernel', windowed_cos)
        else:
            # Interleave [cos ; sin] → one Conv → then Split along channel dim.
            self.register_buffer('stft_kernel', torch.cat([windowed_cos, windowed_sin], dim=0))

    def _build_istft_kernels(self, n_fft, F_bins, window):
        """
        Pre-compute the inverse-DFT basis (windowed) for ConvTranspose1d,
        plus a window² kernel used for overlap-add normalisation.
        """
        omega_factor = 2.0 * torch.pi / n_fft
        k = torch.arange(F_bins, dtype=torch.float32).unsqueeze(1)  # (F, 1)
        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)   # (1, n_fft)
        omega = omega_factor * k * n                                # (F, n_fft)

        cos_basis = torch.cos(omega)
        sin_basis = torch.sin(omega)

        # One-sided spectrum → double non-DC/Nyquist bins to recover full energy.
        scale = 2.0 * torch.ones(F_bins, 1)
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[F_bins - 1] = 1.0

        inv_n     = 1.0 / n_fft
        ifft_real = (scale *  cos_basis * inv_n) * window.unsqueeze(0)
        ifft_imag = (scale * -sin_basis * inv_n) * window.unsqueeze(0)

        # Single transposed-conv kernel: [real ; imag] × 1 group.
        self.register_buffer('inverse_kernel', torch.cat([ifft_real, ifft_imag], dim=0).unsqueeze(1))

        # Window² kernel for overlap-add COLA normalisation.
        win_sq_kernel = window.square().reshape(1, 1, -1)

        inv_win_sum = 1.0 / torch.nn.functional.conv_transpose1d(self.ones, win_sq_kernel, stride=self.hop_len)
        max_n_frames = self.expected_len[-1].long()
        self.register_buffer('inv_win_sum', inv_win_sum[..., :max_n_frames].clamp(max=65504.0).half())

    # --------------------------------------------------------------------- #
    #  Forward dispatcher                                                   #
    # --------------------------------------------------------------------- #

    def forward(self, *args):
        dispatch = {
            'stft_A':  self.stft_A_forward,
            'stft_B':  self.stft_B_forward,
            'istft_A': self.istft_A_forward,
            'istft_B': self.istft_B_forward
        }
        fn = dispatch.get(self.model_type)
        if fn is None:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return fn(*args)

    # --------------------------------------------------------------------- #
    #  Padding  (concat-based — friendlier than F.pad on non-CPU providers) #
    # --------------------------------------------------------------------- #

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Center-pad the input waveform so the first STFT frame is centred on
        sample 0 (matching ``torch.stft(center=True)``).

        Two modes:
        - reflect : mirror the edges (Slice + Flip + Concat — avoids ONNX Pad op).
        - constant: zero-pad using a pre-allocated buffer.
        """
        if not self.center_pad:
            return x

        if self.pad_mode == 'reflect':
            left  = x[..., 1: self.half_n_fft + 1].flip(2)
            right = x[..., -(self.half_n_fft + 1): -1].flip(2)
            return torch.cat([left, x, right], dim=2)

        # Constant (zero) padding.
        if x.shape[0] != 1:
            padding_zero = torch.cat([self.padding_zero] * x.shape[0], dim=0)
        else:
            padding_zero = self.padding_zero
        return torch.cat([padding_zero, x, padding_zero], dim=2)

    # --------------------------------------------------------------------- #
    #  STFT variants                                                        #
    # --------------------------------------------------------------------- #

    def stft_A_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STFT producing real part only (cosine projection)."""
        return torch.nn.functional.conv1d(self._pad_input(x), self.stft_kernel, stride=self.hop_len)

    def stft_B_forward(self, x: torch.Tensor):
        """STFT producing (real, imag) via a single Conv1d + channel Split."""
        out = torch.nn.functional.conv1d(self._pad_input(x), self.stft_kernel, stride=self.hop_len)
        return torch.split(out, self.half_n_fft + 1, dim=1)

    # --------------------------------------------------------------------- #
    #  ISTFT core + variants                                                #
    # --------------------------------------------------------------------- #

    def _istft_core(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """
        Shared inverse-STFT logic:
          1. Concatenate real & imag channels.
          2. ConvTranspose1d (overlap-add synthesis).
          3. Normalize by the squared-window overlap sum (COLA condition).
          4. Remove center padding if applicable.
        """
        inp = torch.cat((real, imag), dim=1)

        # Synthesize waveform via transposed convolution (overlap-add).
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_kernel, stride=self.hop_len)

        # Overlap-add normalization: divide by sum of squared windows.
        n_frames     = real.shape[-1]
        expected_len = self.expected_len[n_frames].long()
        if self.center_pad:
            # Strip the center padding that was added during the forward STFT.
            slice_start = self.half_n_fft
        else:
            slice_start = 0
        inv = inv[..., slice_start: expected_len] * self.inv_win_sum[..., slice_start: expected_len].float()

        return inv

    def istft_A_forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """ISTFT from polar form (magnitude + phase)."""
        return self._istft_core(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

    def istft_B_forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """ISTFT from rectangular form (real + imag)."""
        return self._istft_core(real, imag)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Test / validation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _torch_istft_safe(complex_spec: torch.Tensor):
    """
    Call ``torch.istft`` and return (audio_numpy, True).
    Returns (None, False) if COLA constraint is not satisfied.
    """
    try:
        audio = torch.istft(
            complex_spec,
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=get_raw_window(WIN_LENGTH, WINDOW_TYPE),
            center=CENTER_PAD
        ).squeeze().numpy()
        return audio, True
    except RuntimeError:
        return None, False


# -- STFT tests ----------------------------------------------------------- #

def test_onnx_stft_A(x: torch.Tensor, center_pad: bool = True):
    """Compare ONNX STFT-A output (real only) against ``torch.stft``."""
    torch_out = torch.view_as_real(torch.stft(
        x.squeeze(0),
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        return_complex=True,
        window=get_raw_window(WIN_LENGTH, WINDOW_TYPE),
        pad_mode=PAD_MODE,
        center=center_pad
    ))
    pt_real = torch_out[..., 0].squeeze().numpy()

    sess     = ort.InferenceSession(export_path_stft)
    ort_real = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].squeeze()

    print("STFT Result (A) [ONNX vs torch.stft]: mean |Δ| =", np.abs(pt_real - ort_real[:, :pt_real.shape[-1]]).mean())


def test_onnx_stft_B(x: torch.Tensor, center_pad: bool = True):
    """Compare ONNX STFT-B output (real + imag) against ``torch.stft``."""
    torch_out = torch.view_as_real(torch.stft(
        x.squeeze(0),
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        return_complex=True,
        window=get_raw_window(WIN_LENGTH, WINDOW_TYPE),
        pad_mode=PAD_MODE,
        center=center_pad
    ))
    pt_r = torch_out[..., 0].squeeze().numpy()
    pt_i = torch_out[..., 1].squeeze().numpy()

    sess          = ort.InferenceSession(export_path_stft)
    ort_r, ort_i  = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})

    diff = 0.5 * (np.abs(pt_r - ort_r.squeeze()[:, :pt_r.shape[-1]]).mean() + np.abs(pt_i - ort_i.squeeze()[:, :pt_r.shape[-1]]).mean())
    print("STFT Result (B) [ONNX vs torch.stft]: mean |Δ| =", diff)


# -- ISTFT tests ---------------------------------------------------------- #

def test_onnx_istft_A(mag: torch.Tensor, phase: torch.Tensor):
    """Validate ONNX ISTFT-A (magnitude + phase) against the PyTorch module and torch.istft."""
    sess      = ort.InferenceSession(export_path_istft)
    ort_audio = sess.run(None, {
        sess.get_inputs()[0].name: mag.numpy(),
        sess.get_inputs()[1].name: phase.numpy()
    })[0].squeeze()

    # Compare against torch.istft (may fail if COLA is not satisfied).
    pt_audio, ok = _torch_istft_safe(torch.polar(mag, phase))
    if ok:
        min_len = min(len(pt_audio), len(ort_audio))
        print("ISTFT Result (A) [ONNX vs torch.istft]: mean |Δ| =", np.abs(pt_audio[:min_len] - ort_audio[:min_len]).mean())
    else:
        print("ISTFT Result (A): torch.istft comparison skipped "
              "(COLA not met with center=False + zero-edge window)")


def test_onnx_istft_B(real: torch.Tensor, imag: torch.Tensor):
    """Validate ONNX ISTFT-B (real + imag) against the PyTorch module and torch.istft."""
    sess      = ort.InferenceSession(export_path_istft)
    ort_audio = sess.run(None, {
        sess.get_inputs()[0].name: real.numpy(),
        sess.get_inputs()[1].name: imag.numpy()
    })[0].squeeze()

    # Compare against torch.istft (may fail if COLA is not satisfied).
    pt_audio, ok = _torch_istft_safe(torch.complex(real, imag))
    if ok:
        min_len = min(len(pt_audio), len(ort_audio))
        print("ISTFT Result (B) [ONNX vs torch.istft]: mean |Δ| =", np.abs(pt_audio[:min_len] - ort_audio[:min_len]).mean())
    else:
        print("ISTFT Result (B): torch.istft comparison skipped "
              "(COLA not met with center=False + zero-edge window)")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Export & verification entry-point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    with torch.inference_mode():
        print(
            f"\nConfig  NFFT={NFFT}, WIN_LEN={WIN_LENGTH}, HOP={HOP_LENGTH}, "
            f"CENTER={CENTER_PAD}, FRAMES={STFT_SIGNAL_LENGTH}"
        )

        # ── 5a. Export STFT to ONNX ──────────────────────────────────────
        stft_model  = STFT_Process(STFT_TYPE, center_pad=CENTER_PAD).eval()
        dummy_audio = torch.randn(1, 1, INPUT_AUDIO_LENGTH, dtype=torch.float32)

        dyn_axes_stft = {'input_audio': {2: 'audio_len'}}
        if STFT_TYPE == 'stft_A':
            out_names = ['real']
            dyn_axes_stft['real'] = {2: 'signal_len'}
        else:
            out_names = ['real', 'imag']
            dyn_axes_stft['real'] = {2: 'signal_len'}
            dyn_axes_stft['imag'] = {2: 'signal_len'}

        torch.onnx.export(
            stft_model,
            (dummy_audio,),
            export_path_stft,
            input_names=['input_audio'],
            output_names=out_names,
            dynamic_axes=dyn_axes_stft if DYNAMIC_AXES else None,
            opset_version=OPSET,
            dynamo=False
        )

        slim(
            model=export_path_stft,
            output_model=export_path_stft,
            no_shape_infer=False,
            save_as_external_data=False,
            verbose=False
        )

        # ── 5b. Export ISTFT to ONNX ─────────────────────────────────────
        istft_model = STFT_Process(ISTFT_TYPE, center_pad=CENTER_PAD).eval()

        if ISTFT_TYPE == 'istft_A':
            dummy_mag   = torch.randn(1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH)
            dummy_phase = torch.randn_like(dummy_mag)
            dummy_inp   = (dummy_mag, dummy_phase)
            in_names    = ['magnitude', 'phase']
            dyn_axes_istft = {
                'magnitude':    {2: 'signal_len'},
                'phase':        {2: 'signal_len'},
                'output_audio': {2: 'audio_len'}
            }
        else:
            dummy_real = torch.randn(1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH)
            dummy_imag = torch.randn_like(dummy_real)
            dummy_inp  = (dummy_real, dummy_imag)
            in_names   = ['real', 'imag']
            dyn_axes_istft = {
                'real':         {2: 'signal_len'},
                'imag':         {2: 'signal_len'},
                'output_audio': {2: 'audio_len'}
            }

        torch.onnx.export(
            istft_model,
            dummy_inp,
            export_path_istft,
            input_names=in_names,
            output_names=['output_audio'],
            dynamic_axes=dyn_axes_istft if DYNAMIC_AXES else None,
            opset_version=OPSET,
            dynamo=False
        )

        slim(
            model=export_path_istft,
            output_model=export_path_istft,
            no_shape_infer=False,
            save_as_external_data=False,
            verbose=False
        )

        # ── 5c. Validate STFT against torch.stft ────────────────────────
        print("\nTesting Custom STFT against torch.stft …")
        if STFT_TYPE == 'stft_A':
            test_onnx_stft_A(dummy_audio, center_pad=CENTER_PAD)
        else:
            test_onnx_stft_B(dummy_audio, center_pad=CENTER_PAD)

        # ── 5d. Validate ISTFT ───────────────────────────────────────────
        print("\nTesting Custom ISTFT …")
        if ISTFT_TYPE == 'istft_A':
            test_onnx_istft_A(*dummy_inp)
        else:
            test_onnx_istft_B(*dummy_inp)

        # ── 5e. Round-trip test: STFT → ISTFT ────────────────────────────
        if STFT_TYPE == 'stft_B':
            print("\n--- Round-trip STFT → ISTFT ---")
            stft_sess  = ort.InferenceSession(export_path_stft)
            istft_sess = ort.InferenceSession(export_path_istft)

            ort_r, ort_i = stft_sess.run(None, {'input_audio': dummy_audio.numpy()})

            if ISTFT_TYPE == 'istft_A':
                # Convert rectangular → polar for the A variant.
                mag   = np.sqrt(ort_r ** 2 + ort_i ** 2)
                phase = np.arctan2(ort_i, ort_r)
                recon = istft_sess.run(None, {'magnitude': mag, 'phase': phase})[0]
            else:
                recon = istft_sess.run(None, {'real': ort_r, 'imag': ort_i})[0]

            recon = recon.squeeze()
            orig  = dummy_audio.squeeze().numpy()

            min_len = min(len(orig), len(recon))
            skip    = NFFT if not CENTER_PAD else 0  # Edge frames may be inaccurate without centre pad.
            if min_len > 2 * skip:
                s, e = skip, min_len - skip
                print(f"Round-trip mean |Δ| (skipping {skip} edge samples) =", np.abs(orig[s:e] - recon[s:e]).mean())


if __name__ == "__main__":
    main()