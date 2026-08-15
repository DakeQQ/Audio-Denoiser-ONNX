"""Static packed STFT/ISTFT transforms used by the GAP-URGENet exporter."""

import torch

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Configuration
# ═════════════════════════════════════════════════════════════════════════════

# Defaults preserve the standalone helper's previous constructor contract.
_DEFAULT_NFFT = 400
_DEFAULT_WIN_LENGTH = 400
_DEFAULT_HOP_LENGTH = 160
_DEFAULT_MAX_FRAMES = 101
_DEFAULT_WINDOW_TYPE = "hann"
_DEFAULT_CENTER_PAD = True
_DEFAULT_PAD_MODE = "constant"


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Window helpers
# ═════════════════════════════════════════════════════════════════════════════

# -- Window function registry ----------------------------------------------
WINDOW_FUNCTIONS = {
    'bartlett':  lambda L: torch.bartlett_window(L, periodic=True),
    'blackman':  lambda L: torch.blackman_window(L, periodic=True),
    'hamming':   lambda L: torch.hamming_window(L,  periodic=True),
    'hann':      lambda L: torch.hann_window(L,     periodic=True),
    'hann_sqrt': lambda L: torch.hann_window(L,     periodic=True).pow(0.5),
    'povey':     lambda L: torch.hann_window(L,     periodic=False).pow(0.85),
    'kaiser':    lambda L: torch.kaiser_window(L,   periodic=True, beta=12.0)
}
DEFAULT_WINDOW_FN = lambda L: torch.hann_window(L, periodic=True)


def create_padded_window(win_length: int, n_fft: int, window_type: str) -> torch.Tensor:
    """Create a window of length *n_fft*, center-padding or cropping as needed."""
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    win = win_fn(win_length).float()

    if win_length == n_fft:
        return win
    if win_length < n_fft:
        pad_total = n_fft - win_length
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.cat([torch.zeros(pad_left), win, torch.zeros(pad_right)])
    start = (win_length - n_fft) // 2
    return win[start : start + n_fft]


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Optimized STFT / ISTFT Models (Static Graph)
# ═════════════════════════════════════════════════════════════════════════════

class STFT_Process(torch.nn.Module):
    """
    Static-graph Conv1d STFT / ConvTranspose1d ISTFT for ONNX export.

    All constants precomputed in __init__() as registered buffers.
    Forward path is pure tensor ops — no dispatch, no branching, no shape queries.

    Variants
    --------
    stft_A   → Conv1d producing real part only.
    stft_B   → Conv1d producing real + imag (split after convolution).
    istft_A  → (magnitude, phase) → ConvTranspose1d reconstruction.
    istft_B  → (real, imag) → ConvTranspose1d reconstruction.
    """

    def __init__(
        self,
        model_type: str,
        n_fft: int = _DEFAULT_NFFT,
        win_length: int = _DEFAULT_WIN_LENGTH,
        hop_len: int = _DEFAULT_HOP_LENGTH,
        max_frames: int = _DEFAULT_MAX_FRAMES,
        window_type: str = _DEFAULT_WINDOW_TYPE,
        center_pad: bool = _DEFAULT_CENTER_PAD,
        pad_mode: str = _DEFAULT_PAD_MODE,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
        static_norm: bool = False,
        istft_trim_left: int = 0,
        istft_trim_right: int = 0,
        static_batch: int | None = None,
        persistent_buffers: bool = True,
    ):
        super().__init__()

        self.model_type = model_type
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.half_n_fft = n_fft // 2
        self.n_frames   = max_frames
        if static_batch is not None and static_batch < 1:
            raise ValueError("static_batch must be positive when specified.")
        self.static_batch = static_batch
        self._persistent_buffers = persistent_buffers

        f_bins = self.half_n_fft + 1
        window = create_padded_window(win_length, n_fft, window_type)

        if istft_trim_left < 0 or istft_trim_right < 0:
            raise ValueError("ISTFT trim values must be non-negative.")

        # ── Precompute static output slice bounds for ISTFT ───────────────
        raw_len = n_fft + hop_len * (max_frames - 1)
        if center_pad:
            out_start = self.half_n_fft
            out_end = raw_len - self.half_n_fft
        else:
            out_start = 0
            out_end = raw_len
        self._out_start = out_start + istft_trim_left
        self._out_end = out_end - istft_trim_right
        if self._out_start >= self._out_end:
            raise ValueError(
                "ISTFT trim removes the complete reconstructed waveform: "
                f"raw_length={raw_len}, start={self._out_start}, end={self._out_end}."
            )

        # ── Bind forward to the correct variant (no dispatch overhead) ────
        if model_type == 'stft_A':
            self.forward = self._stft_A_forward
        elif model_type == 'stft_B':
            self.forward = self._stft_B_forward
        elif model_type == 'istft_A':
            self.forward = self._istft_A_forward
        elif model_type == 'istft_B':
            self.forward = self._istft_B_forward
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # ── STFT: constant zero-padding buffer ────────────────────────────
        if model_type in ('stft_A', 'stft_B'):
            self._build_stft_kernels(n_fft, f_bins, window, model_type, input_scale)
            if center_pad and pad_mode == 'constant':
                self.register_buffer(
                    'padding_zero',
                    torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32),
                    persistent=self._persistent_buffers,
                )
            self._center_pad = center_pad
            self._pad_mode   = pad_mode

        # ── ISTFT: inverse kernel + pre-sliced normalization ──────────────
        if model_type in ('istft_A', 'istft_B'):
            self._build_istft_kernels(
                n_fft,
                f_bins,
                window,
                hop_len,
                max_frames,
                static_norm,
                output_scale,
            )

    def _build_stft_kernels(self, n_fft, f_bins, window, model_type, input_scale):
        """Precompute windowed DFT basis as Conv1d kernel weights."""
        omega_factor = 2.0 * torch.pi / n_fft
        t = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        f = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        omega = omega_factor * f * t

        scaled_window = window * input_scale
        windowed_cos = ( torch.cos(omega) * scaled_window.unsqueeze(0)).unsqueeze(1)
        windowed_sin = (-torch.sin(omega) * scaled_window.unsqueeze(0)).unsqueeze(1)

        if model_type == 'stft_A':
            self.register_buffer('stft_kernel', windowed_cos, persistent=self._persistent_buffers)
        else:
            self.register_buffer(
                'stft_kernel',
                torch.cat([windowed_cos, windowed_sin], dim=0),
                persistent=self._persistent_buffers,
            )

    def _build_istft_kernels(self, n_fft, f_bins, window, hop_len, n_frames, static_norm, output_scale):
        """Precompute inverse-DFT kernel and window² kernel for COLA normalization."""
        omega_factor = 2.0 * torch.pi / n_fft
        k = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        omega = omega_factor * k * n

        cos_basis = torch.cos(omega)
        sin_basis = torch.sin(omega)

        scale = 2.0 * torch.ones(f_bins, 1)
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[f_bins - 1] = 1.0

        inv_n     = 1.0 / n_fft
        ifft_real = (scale *  cos_basis * inv_n) * window.unsqueeze(0)
        ifft_imag = (scale * -sin_basis * inv_n) * window.unsqueeze(0)

        self.register_buffer(
            'inverse_kernel',
            torch.cat([ifft_real, ifft_imag], dim=0).unsqueeze(1),
            persistent=self._persistent_buffers,
        )

        self.static_norm = static_norm
        win_sq_kernel = window.square().reshape(1, 1, -1)
        if static_norm:
            win_sum = torch.nn.functional.conv_transpose1d(
                torch.ones(1, 1, n_frames),
                win_sq_kernel,
                stride=hop_len,
            )
            win_sum = win_sum[..., self._out_start:self._out_end].contiguous()
            output_length = self._out_end - self._out_start
            self.static_output_length = output_length
            self.static_output_periods = output_length // hop_len
            periodic = output_length % hop_len == 0
            if periodic:
                periods = win_sum.reshape(-1, hop_len)
                periodic = torch.equal(periods, periods[0:1].expand_as(periods))
            self.periodic_static_norm = periodic
            self.register_buffer(
                'win_sum',
                win_sum[..., :hop_len].reshape(1, 1, 1, hop_len) if periodic else win_sum,
                persistent=self._persistent_buffers,
            )
            self.output_scale = output_scale
        else:
            self.register_buffer('win_sq_kernel', win_sq_kernel, persistent=self._persistent_buffers)
            self.output_scale = output_scale

    # --------------------------------------------------------------------- #
    #  STFT forward variants (no branching, static tensor ops only)         #
    # --------------------------------------------------------------------- #

    def _stft_A_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STFT producing real part only (cosine projection)."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left  = x[..., 1: self.half_n_fft + 1].flip(2)
                right = x[..., -(self.half_n_fft + 1): -1].flip(2)
                x = torch.cat([left, x, right], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero = torch.cat([self.padding_zero] * x.shape[0], dim=0)
                else:
                    padding_zero = self.padding_zero
                x = torch.cat([padding_zero, x, padding_zero], dim=2)
        return torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)

    def _stft_B_forward(self, x: torch.Tensor):
        """STFT producing (real, imag) via a single Conv1d + channel Split."""
        out = self._stft_B_packed_forward(x)
        return torch.split(out, self.half_n_fft + 1, dim=1)

    def _stft_B_packed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STFT with real/imaginary channels kept packed as ``(B,2F,T)``."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left  = x[..., 1: self.half_n_fft + 1].flip(2)
                right = x[..., -(self.half_n_fft + 1): -1].flip(2)
                x = torch.cat([left, x, right], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero = torch.cat([self.padding_zero] * x.shape[0], dim=0)
                else:
                    padding_zero = self.padding_zero
                x = torch.cat([padding_zero, x, padding_zero], dim=2)
        return torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)

    # --------------------------------------------------------------------- #
    #  ISTFT forward variants (static slicing, no Shape/Gather ops)         #
    # --------------------------------------------------------------------- #

    def _istft_B_forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """ISTFT from rectangular form. Dynamic-length compatible."""
        return self._istft_B_packed_forward(torch.cat((real, imag), dim=1))

    def _istft_B_packed_forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ISTFT from packed real/imaginary channels ``(B,2F,T)``."""
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_kernel, stride=self.hop_len)
        if self.static_norm:
            inv = inv[..., self._out_start:self._out_end]
            if self.periodic_static_norm:
                batch = -1 if self.static_batch is None else self.static_batch
                inv = inv.reshape(batch, 1, self.static_output_periods, self.hop_len)
                inv = (inv / self.win_sum).reshape(batch, 1, self.static_output_length)
            else:
                inv = inv / self.win_sum
            return inv if self.output_scale == 1.0 else inv * self.output_scale
        # Compute COLA normalization dynamically based on input n_frames.
        ones = torch.ones(1, 1, inp.shape[2], dtype=inp.dtype, device=inp.device)
        win_sum = torch.nn.functional.conv_transpose1d(ones, self.win_sq_kernel, stride=self.hop_len)
        inv = inv[..., self._out_start:self._out_end] / win_sum[..., self._out_start:self._out_end]
        return inv * self.output_scale

    def _istft_A_forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """ISTFT from polar form. Dynamic-length compatible."""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        inp = torch.cat((real, imag), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_kernel, stride=self.hop_len)
        if self.static_norm:
            inv = inv[..., self._out_start:self._out_end]
            if self.periodic_static_norm:
                batch = -1 if self.static_batch is None else self.static_batch
                inv = inv.reshape(batch, 1, self.static_output_periods, self.hop_len)
                inv = (inv / self.win_sum).reshape(batch, 1, self.static_output_length)
            else:
                inv = inv / self.win_sum
            return inv if self.output_scale == 1.0 else inv * self.output_scale
        # Compute COLA normalization dynamically based on input n_frames.
        ones = torch.ones(1, 1, magnitude.shape[2], dtype=magnitude.dtype, device=magnitude.device)
        win_sum = torch.nn.functional.conv_transpose1d(ones, self.win_sq_kernel, stride=self.hop_len)
        inv = inv[..., self._out_start:self._out_end] / win_sum[..., self._out_start:self._out_end]
        return inv * self.output_scale
