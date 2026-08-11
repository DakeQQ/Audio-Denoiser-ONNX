"""Standalone DPDFNet checkpoint exporter with a static one-frame ONNX boundary."""

from __future__ import annotations

import gc
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Callable, Final, Iterable, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from onnx.compose import add_prefix
from torch import Tensor

EncoderStreamingState = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
DfDecoderStreamingState = Tuple[Tensor, Tensor]
StreamingState = Tuple[Tensor, Tensor, EncoderStreamingState, Tensor, DfDecoderStreamingState, Tensor, Tensor]

try:
    from STFT_Process import STFT_Process
except ModuleNotFoundError:
    from DPDFNet.STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate audio_onnx_metadata.py")
from audio_onnx_metadata import (
    build_audio_metadata_from_globals,
    build_model_metadata,
    export_metadata_carrier,
    metadata_path_for_model,
)


parent_path = Path(__file__).resolve().parent

DPDFNET_16KHZ_DPRNN_BLOCKS: Final[dict[str, int]] = {
    "baseline": 0,
    "dpdfnet2": 2,
    "dpdfnet4": 4,
    "dpdfnet8": 8,
}


# User settings.
model_path      = str(Path.home() / "Downloads" / "DPDFNet")                # The DPDFNet download path.
checkpoint_path = str(Path(model_path) / "checkpoints" / "dpdfnet2.pth")    # The checkpoint to export.  [dpdfnet2, dpdfnet4, dpdfnet8]
onnx_model_A    = parent_path / "DPDFNet_ONNX" / "DPDFNet.onnx"             # The exported model path.
IN_SAMPLE_RATE  = 16000                                                     # [8000, 16000, 22500, 24000, 44000, 48000]
OUT_SAMPLE_RATE = 16000                                                     # [8000, 16000, 22500, 24000, 44000, 48000]
INPUT_AUDIO_LENGTH = 16000                                                  # Static input capacity in input-rate samples.
IN_AUDIO_DTYPE  = "F32"                                                     # ["F16", "F32", "INT16"]
OUT_AUDIO_DTYPE = "F32"                                                     # ["F16", "F32", "INT16"]

# Fixed DPDFNet model and ONNX export parameters.
OPSET               = 20
MODEL_SAMPLE_RATE   = 16000
NFFT                = 320
WINDOW_LENGTH       = 320
HOP_LENGTH          = 160
DPDFNET_OUTPUT_DELAY = WINDOW_LENGTH * 2
INV_INT16           = float(1.0 / 32768.0)
FUSE_GROUPED_LINEAR = True
FUSE_GROUPED_CONVOLUTIONS  = True
FUSE_SUBPIXEL_CONVOLUTIONS = True
FUSE_BATCH_NORMALIZATION   = True
EXPORT_SEQUENCE_FIRST   = True
FLOAT_VALIDATION_RTOL   = 1e-4
FLOAT_VALIDATION_ATOL   = 1e-4
INTEGER_VALIDATION_ATOL = 1

# Derived export dimensions.
MODEL_AUDIO_LENGTH  = math.ceil(INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
OUTPUT_AUDIO_LENGTH = math.ceil(INPUT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)
STFT_INPUT_LENGTH   = MODEL_AUDIO_LENGTH + WINDOW_LENGTH
STATIC_STFT_FRAMES  = STFT_INPUT_LENGTH // HOP_LENGTH + 1


checkpoint_model_name = lambda path: Path(path).expanduser().stem.lower()


def resolve_checkpoint_config(path: str | Path) -> tuple[str, int]:
    """Return the supported 16 kHz model name and DPRNN depth for a checkpoint path."""

    model_name = checkpoint_model_name(path)
    try:
        return model_name, DPDFNET_16KHZ_DPRNN_BLOCKS[model_name]
    except KeyError as error:
        supported = ", ".join(DPDFNET_16KHZ_DPRNN_BLOCKS)
        raise ValueError(
            f"Unsupported DPDFNet checkpoint name {model_name!r}. "
            f"This exporter supports the 16 kHz checkpoints: {supported}."
        ) from error


MODEL_NAME, DPRNN_NUM_BLOCKS = resolve_checkpoint_config(checkpoint_path)


def audio_torch_dtype(name: str) -> torch.dtype:
    """Map the public audio dtype setting to the ONNX-export dummy tensor dtype."""

    dtypes = {
        "F16": torch.float16,
        "F32": torch.float32,
        "INT16": torch.int16,
    }
    try:
        return dtypes[name.upper()]
    except KeyError as error:
        raise ValueError(f"Unsupported audio dtype {name!r}; use one of {sorted(dtypes)}.") from error


IN_TORCH_DTYPE = audio_torch_dtype(IN_AUDIO_DTYPE)
OUT_TORCH_DTYPE = audio_torch_dtype(OUT_AUDIO_DTYPE)


def as_complex(tensor: Tensor) -> Tensor:
    if torch.is_complex(tensor):
        return tensor
    if tensor.shape[-1] != 2:
        raise ValueError(f"The last dimension must be real/imaginary pairs, got {tuple(tensor.shape)}.")
    return torch.view_as_complex(tensor if tensor.stride(-1) == 1 else tensor.contiguous())


def as_real(tensor: Tensor) -> Tensor:
    return torch.view_as_real(tensor) if torch.is_complex(tensor) else tensor


def get_mag(tensor: Tensor) -> Tensor:
    if tensor.shape[-1] != 2:
        raise ValueError(f"Expected real/imaginary pairs, got {tuple(tensor.shape)}.")
    return tensor.square().sum(dim=-1).sqrt()


def get_pow(tensor: Tensor) -> Tensor:
    if tensor.shape[-1] != 2:
        raise ValueError(f"Expected real/imaginary pairs, got {tuple(tensor.shape)}.")
    return tensor.square().sum(dim=-1)


def to_db(tensor: Tensor) -> Tensor:
    return 10.0 * torch.log10(tensor + 1e-10)


def vorbis_window(window_length: int) -> Tensor:
    window = np.zeros(window_length, dtype=np.float32)
    half_length = window_length / 2
    for index in range(window_length):
        sine = np.sin(0.5 * np.pi * (index + 0.5) / half_length)
        window[index] = np.sin(0.5 * np.pi * sine * sine)
    return torch.from_numpy(window)


def get_wnorm(window_length: int, hop_length: int) -> float:
    return 1.0 / (window_length ** 2 / (2 * hop_length))


def erb_filter_banks(
    n_filters: int = 32,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: int = 0,
    high_freq: Optional[int] = None,
    min_nb_freqs: int = 2,
) -> np.ndarray:
    """Build the published DPDFNet rectangular ERB filter bank."""

    def frequency_to_erb(frequency):
        return 9.265 * np.log1p(frequency / (24.7 * 9.265))

    def erb_to_frequency(erb):
        return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)

    high_freq = high_freq if high_freq is not None else fs // 2
    if high_freq > fs // 2 or not 0 <= low_freq < high_freq:
        raise ValueError("ERB filter-bank frequency limits are invalid.")
    nyquist = fs / 2
    frequency_width = fs / nfft
    erb_low = frequency_to_erb(0.0)
    erb_high = frequency_to_erb(nyquist)
    step = (erb_high - erb_low) / n_filters
    bins = np.zeros(n_filters + 1, dtype=np.int32)
    for index in range(n_filters + 1):
        bins[index] = int(round(erb_to_frequency(erb_low + index * step) / frequency_width))
    bins[-1] = nfft // 2 + 1

    filter_bank = np.zeros((n_filters, nfft // 2 + 1), dtype=np.float32)
    frequency_overflow = 0
    for index in range(n_filters):
        start, end = bins[index] + frequency_overflow, bins[index + 1]
        if end - start < min_nb_freqs:
            frequency_overflow = min_nb_freqs - (end - start)
            end = min(end + frequency_overflow, nfft // 2 + 1)
        else:
            frequency_overflow = 0
        filter_bank[index, start:end] = 1.0
    if not np.all(filter_bank.sum(axis=1) > 0.0):
        raise ValueError("DPDFNet ERB filter bank contains an empty band.")
    return np.abs(filter_bank)


class CyclicBuffer(nn.Module):
    """A state-vector-backed fixed-size FIFO with a single-frame input contract."""

    def __init__(self, shape: list[int], time_steps: int, delay_frames: int = 0, time_dim: int = 2):
        super().__init__()
        if time_steps < 1 or delay_frames < 0:
            raise ValueError("CyclicBuffer requires time_steps >= 1 and delay_frames >= 0.")
        self.time_steps = int(time_steps)
        self.delay_frames = int(delay_frames)
        self.time_dim = int(time_dim)
        self.capacity = self.time_steps + self.delay_frames
        shape = list(shape)
        if self.time_dim < 0:
            self.time_dim += len(shape)
        if not 0 <= self.time_dim < len(shape):
            raise ValueError(f"time_dim={time_dim} is outside a {len(shape)}-rank tensor.")
        shape[self.time_dim] = self.capacity
        dimensions = list(range(len(shape)))
        self.perm = [self.time_dim] + [dimension for dimension in dimensions if dimension != self.time_dim]
        self.inv_perm = [self.perm.index(dimension) for dimension in dimensions]
        self.shape_tf = tuple(shape[dimension] for dimension in self.perm)
        self._state_size = math.prod(self.shape_tf)

    def state_size(self) -> int:
        return self._state_size

    def initial_state(
        self,
        state: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if state is None:
            return torch.zeros(self._state_size, dtype=dtype, device=device)
        state = state.reshape(-1)
        if state.numel() != self._state_size:
            raise ValueError(f"CyclicBuffer state must contain {self._state_size} values, got {state.numel()}.")
        return state

    def forward(self, tensor: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        if tensor.dim() != len(self.inv_perm):
            raise ValueError(f"CyclicBuffer expected rank {len(self.inv_perm)}, got {tensor.dim()}.")
        if tensor.shape[self.time_dim] != 1:
            raise ValueError("CyclicBuffer only accepts a single frame on its time dimension.")
        time_first = tensor.permute(self.perm)
        if state is None:
            buffer = time_first.new_zeros(self.shape_tf)
        else:
            end = offset + self._state_size
            flat_buffer = state[offset:end]
            if flat_buffer.numel() != self._state_size:
                raise ValueError("DPDFNet state does not contain the requested CyclicBuffer region.")
            buffer = flat_buffer.view(self.shape_tf)
        next_buffer = torch.cat((buffer[1:], time_first), dim=0)
        output = next_buffer[:self.time_steps].permute(self.inv_perm)
        if state is None:
            return output
        return output, next_buffer.reshape(-1), offset + self._state_size


class GRUCellInternalState(nn.Module):
    def __init__(self, input_size: int, units: int, batch_size: int = 1):
        super().__init__()
        self.units = units
        self.batch_size = batch_size
        self.grucell = nn.GRUCell(input_size=input_size, hidden_size=units)

    def state_size(self) -> int:
        return self.batch_size * self.units

    def initial_state(
        self,
        state: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if state is None:
            return torch.zeros(self.state_size(), dtype=dtype, device=device)
        state = state.reshape(-1)
        if state.numel() != self.state_size():
            raise ValueError(f"GRU state must contain {self.state_size()} values, got {state.numel()}.")
        return state

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        if state is None:
            return self.grucell(inputs, inputs.new_zeros(inputs.shape[0], self.units))
        if inputs.shape[0] != self.batch_size:
            raise ValueError(f"GRU expects batch {self.batch_size}, got {inputs.shape[0]}.")
        end = offset + self.state_size()
        hidden = state[offset:end]
        if hidden.numel() != self.state_size():
            raise ValueError("DPDFNet state does not contain the requested GRU region.")
        output = self.grucell(inputs, hidden.view(self.batch_size, self.units))
        return output, output.reshape(-1), end


class FixedSequenceEMA(nn.Module):
    """Evaluate a fixed-length exponential moving average without a temporal Loop."""

    def __init__(self, alpha: float, frame_count: int):
        super().__init__()
        if not 0.0 <= alpha < 1.0 or frame_count < 1:
            raise ValueError("FixedSequenceEMA requires 0 <= alpha < 1 and frame_count >= 1.")
        indices = torch.arange(frame_count, dtype=torch.float32)
        lag = indices[:, None] - indices[None, :]
        coefficients = torch.where(
            lag >= 0,
            (1.0 - alpha) * torch.pow(torch.tensor(alpha, dtype=torch.float32), lag),
            torch.zeros_like(lag),
        )
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("initial_decay", torch.pow(torch.tensor(alpha, dtype=torch.float32), indices + 1.0))
        self.frame_count = frame_count

    def forward(self, inputs: Tensor, initial_state: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.frame_count:
            raise ValueError(
                f"FixedSequenceEMA expects [batch, {self.frame_count}, features], got {tuple(inputs.shape)}."
            )
        initial_state = initial_state.reshape(1, 1, -1)
        weighted_inputs = torch.bmm(self.coefficients.unsqueeze(0), inputs)
        return weighted_inputs + self.initial_decay.view(1, -1, 1) * initial_state


class GRUCellSequence(nn.Module):
    """Lossless fixed-batch sequence equivalent of one or more GRUCell layers."""

    def __init__(self, cells: Iterable[GRUCellInternalState], batch_size: int):
        super().__init__()
        cells = tuple(cells)
        if not cells or batch_size < 1:
            raise ValueError("GRUCellSequence requires at least one GRUCell and a positive batch size.")
        input_size = cells[0].grucell.input_size
        hidden_size = cells[0].grucell.hidden_size
        if any(
            cell.grucell.input_size != input_size
            or cell.grucell.hidden_size != hidden_size
            for cell in cells
        ):
            raise ValueError("All GRUCellSequence layers must have identical input and hidden sizes.")
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=len(cells),
            batch_first=True,
        )
        with torch.no_grad():
            for layer_index, cell in enumerate(cells):
                self.gru.__getattr__(f"weight_ih_l{layer_index}").copy_(cell.grucell.weight_ih)
                self.gru.__getattr__(f"weight_hh_l{layer_index}").copy_(cell.grucell.weight_hh)
                self.gru.__getattr__(f"bias_ih_l{layer_index}").copy_(cell.grucell.bias_ih)
                self.gru.__getattr__(f"bias_hh_l{layer_index}").copy_(cell.grucell.bias_hh)
        self.batch_size = batch_size
        self.register_buffer(
            "initial_state",
            torch.zeros(len(cells), batch_size, hidden_size),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[0] != self.batch_size:
            raise ValueError(
                f"GRUCellSequence expects [batch={self.batch_size}, time, features], got {tuple(inputs.shape)}."
            )
        outputs, _ = self.gru(inputs, self.initial_state)
        return outputs


class DPRNNBlock(nn.Module):
    """The published DPDFNet dual-path RNN block in its static single-frame form."""

    def __init__(self, num_feat: int, hidden_dim: int, stateful: bool = True):
        super().__init__()
        self.num_feat = num_feat
        self.hidden_dim = hidden_dim
        self.stateful = stateful
        self.intra_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_intra = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_intra = nn.LayerNorm(hidden_dim)
        self.inter_gru = GRUCellInternalState(hidden_dim, hidden_dim, batch_size=num_feat)
        self.fc_inter = nn.Linear(hidden_dim, hidden_dim)
        self.ln_inter = nn.LayerNorm(hidden_dim)

    def state_size(self) -> int:
        return self.inter_gru.state_size()

    def initial_state(
        self,
        state: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"DPRNN state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return self.inter_gru.initial_state(device=device, dtype=dtype)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        batch, channels, frames, frequencies = inputs.shape
        if channels != self.hidden_dim:
            raise ValueError(f"DPRNN expected {self.hidden_dim} channels, got {channels}.")
        intra = inputs.permute(0, 2, 3, 1).reshape(batch * frames, frequencies, channels)
        intra, _ = self.intra_gru(intra)
        intra = self.ln_intra(self.fc_intra(intra))
        intra = intra.reshape(batch, frames, frequencies, channels).permute(0, 3, 1, 2)
        residual = inputs + intra

        inter = residual.permute(0, 3, 2, 1).reshape(batch * frequencies * frames, channels)
        if state is None:
            inter = self.inter_gru(inter)
            state_out = None
        else:
            inter, state_out, offset = self.inter_gru(inter, state=state, offset=offset)
        inter = self.ln_inter(self.fc_inter(inter))
        inter = inter.reshape(batch, frequencies, frames, channels).permute(0, 3, 2, 1)
        output = residual + inter
        if state is None:
            return output
        return output, state_out, offset


class DPRNN(nn.Module):
    def __init__(self, num_feat: int, ch_in: int, hidden_dim: int, ch_out: int, num_blocks: int = 6, stateful: bool = True):
        super().__init__()
        self.input_proj = nn.Identity() if ch_in == hidden_dim else nn.Conv2d(ch_in, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            DPRNNBlock(num_feat=num_feat, hidden_dim=hidden_dim, stateful=stateful)
            for _ in range(num_blocks)
        )
        self.output_proj = nn.Identity() if hidden_dim == ch_out else nn.Conv2d(hidden_dim, ch_out, kernel_size=1)

    def state_size(self) -> int:
        return sum(block.state_size() for block in self.blocks)

    def initial_state(
        self,
        state: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"DPRNN state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        states = [block.initial_state(device=device, dtype=dtype) for block in self.blocks]
        return torch.cat(states, dim=0) if states else torch.zeros(0, dtype=dtype, device=device)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        output = self.input_proj(inputs)
        state_outputs: List[Tensor] = []
        for block in self.blocks:
            if state is None:
                output = block(output)
            else:
                output, block_state, offset = block(output, state=state, offset=offset)
                state_outputs.append(block_state)
        output = self.output_proj(output)
        if state is None:
            return output
        state_out = torch.cat(state_outputs, dim=0) if state_outputs else state.new_zeros(0)
        return output, state_out, offset


class CheckpointStftState(nn.Module):
    """Retain DPDFNet's checkpoint window key without direct Fourier operator calls."""

    def __init__(self, n_fft: int, win_length: int, hop_length: int, window: Tensor):
        super().__init__()
        if window.shape[0] != win_length:
            raise ValueError("STFT window length does not match the checkpoint transform state.")
        self.n_fft = n_fft
        self.win_len = win_length
        self.hop = hop_length
        self.register_buffer("w", window)


class CheckpointIstftState(nn.Module):
    """Retain DPDFNet's checkpoint synthesis-window keys until custom transforms are built."""

    def __init__(self, n_fft: int, win_length: int, hop_length: int, window: Tensor):
        super().__init__()
        if window.shape[0] != win_length:
            raise ValueError("ISTFT window length does not match the checkpoint transform state.")
        self.n_fft_inv = n_fft
        self.win_len_inv = win_length
        self.hop_inv = hop_length
        self.register_buffer("w_inv", window)


class Mask(nn.Module):
    def __init__(self, erb_inv_fb: Tensor, post_filter: bool = False, eps: float = 1e-12):
        super().__init__()
        self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.post_filter = post_filter
        self.eps = eps
        self.spec_buffer = CyclicBuffer(
            shape=[1, 1, 1, erb_inv_fb.shape[1], 2],
            time_steps=1,
            delay_frames=2,
            time_dim=2,
        )

    def state_size(self) -> int:
        return self.spec_buffer.state_size()

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Mask state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return self.spec_buffer.initial_state(device=device, dtype=dtype)

    def forward(self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None, state: Optional[Tensor] = None, offset: int = 0):
        if atten_lim is not None:
            mask = mask.clamp(min=(10.0 ** (-atten_lim / 20.0)).view(-1, 1, 1, 1))
        mask = mask.matmul(self.erb_inv_fb)
        if not spec.is_complex():
            mask = mask.unsqueeze(4)
        if state is None:
            return self.spec_buffer(spec) * mask
        buffered_spec, state_out, offset = self.spec_buffer(spec, state=state, offset=offset)
        return buffered_spec * mask, state_out, offset


class ErbNorm(nn.Module):
    def __init__(self, num_feat: int, alpha: float, eps: float = 1e-12, stateful: bool = False, dynamic_var: bool = False):
        super().__init__()
        self.num_feat = num_feat
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.dynamic_var = dynamic_var
        self.register_buffer("mu0", self._initial_mean(), persistent=False)

    def _initial_mean(self) -> Tensor:
        step = (-90.0 - -60.0) / (self.num_feat - 1)
        return (-60.0 + torch.arange(self.num_feat) * step).reshape(1, 1, self.num_feat)

    def state_size(self) -> int:
        return self.num_feat

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"ERB state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return self.mu0.reshape(-1).clone()

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        if inputs.ndim != 3:
            raise ValueError("ERB normalization expects [batch, frame, band].")
        if state is None:
            mean = self.alpha * self.mu0 + (1.0 - self.alpha) * inputs
            return (inputs - mean) / 40.0
        end = offset + self.state_size()
        previous = state[offset:end]
        if previous.numel() != self.state_size():
            raise ValueError("DPDFNet state does not contain the requested ERB normalization region.")
        mean = self.alpha * previous.view(1, 1, self.num_feat) + (1.0 - self.alpha) * inputs
        return (inputs - mean) / 40.0, mean.reshape(-1), end


class SpecNorm(nn.Module):
    def __init__(self, num_feat: int, alpha: float, eps: float = 1e-12, stateful: bool = False):
        super().__init__()
        self.num_feat = num_feat
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.register_buffer("s0", self._initial_power(), persistent=False)

    def _initial_power(self) -> Tensor:
        step = (0.0001 - 0.001) / (self.num_feat - 1)
        return (0.001 + torch.arange(self.num_feat) * step).reshape(1, 1, self.num_feat)

    def state_size(self) -> int:
        return self.num_feat

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Spectrum state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return self.s0.reshape(-1).clone()

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        if inputs.ndim != 4:
            raise ValueError("Spectrum normalization expects [batch, frame, bin, real_imag].")
        magnitude = get_mag(inputs)
        if state is None:
            power = self.alpha * self.s0 + (1.0 - self.alpha) * magnitude
            denominator = (power + self.eps).sqrt()
            return torch.stack((inputs[..., 0] / denominator, inputs[..., 1] / denominator), dim=-1)
        end = offset + self.state_size()
        previous = state[offset:end]
        if previous.numel() != self.state_size():
            raise ValueError("DPDFNet state does not contain the requested spectrum normalization region.")
        power = self.alpha * previous.view(1, 1, self.num_feat) + (1.0 - self.alpha) * magnitude
        denominator = (power + self.eps).sqrt()
        normalized = torch.stack((inputs[..., 0] / denominator, inputs[..., 1] / denominator), dim=-1)
        return normalized, power.reshape(-1), end


class Conv2DPointWiseAsLinear(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, bias: bool = True):
        super().__init__()
        self.in_channel = input_channel
        self.output_channel = output_channel
        self.cnn_fc = nn.Linear(input_channel, output_channel, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, frames, frequencies = inputs.shape
        output = self.cnn_fc(inputs.permute(0, 2, 3, 1).reshape(batch * frames * frequencies, channels))
        return output.reshape(batch, frames, frequencies, self.output_channel).permute(0, 3, 1, 2)


class GroupedConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding="valid", dilation=(1, 1), bias=True, groups=1):
        super().__init__()
        if in_ch % groups or out_ch % groups:
            raise ValueError("Grouped convolution channels must be divisible by groups.")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.groups = groups
        self.convs = nn.ModuleList(
            nn.Conv2d(in_ch // groups, out_ch // groups, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(groups)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.groups == 1:
            return self.convs[0](inputs)
        return torch.cat(
            tuple(convolution(split) for convolution, split in zip(self.convs, torch.chunk(inputs, self.groups, dim=1))),
            dim=1,
        )


class Conv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Iterable[int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        point_wise_type: str = "cnn",
    ):
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        frequency_padding = kernel_size[1] // 2 + dilation - 1 if fpad else 0
        layers: List[nn.Module] = []
        if kernel_size[0] > 1:
            layers.append(nn.Identity())
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1 or max(kernel_size) == 1:
            separable = False
        if groups > 1 and not (in_ch == out_ch == groups):
            layers.append(
                GroupedConv2D(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=(0, frequency_padding),
                    stride=(1, fstride),
                    dilation=(1, dilation),
                    groups=groups,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=(0, frequency_padding),
                    stride=(1, fstride),
                    dilation=(1, dilation),
                    groups=groups,
                    bias=bias,
                )
            )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False) if point_wise_type == "cnn" else Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        point_wise_type: str = "cnn",
    ):
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        frequency_padding = kernel_size[1] // 2 if fpad else 0
        layers: List[nn.Module] = []
        if kernel_size[0] > 1:
            layers.append(nn.ConstantPad2d((0, 0, kernel_size[0] - 1, 0), 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(kernel_size[0] - 1, frequency_padding + dilation - 1),
                output_padding=(0, frequency_padding),
                stride=(1, fstride),
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False) if point_wise_type == "cnn" else Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class SubPixelConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride=2, padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        super().__init__()
        if fstride <= 1:
            raise ValueError("Sub-pixel convolution requires a frequency stride greater than one.")
        self.fstride = fstride
        self.out_channels = out_channels
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=padding, dilation=dilation, groups=groups)
            for _ in range(fstride)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        output = torch.cat(tuple(convolution(inputs) for convolution in self.convs), dim=1)
        batch, _, frames, frequencies = output.shape
        return output.reshape(batch, self.fstride, self.out_channels, frames, frequencies).permute(0, 2, 3, 4, 1).reshape(batch, self.out_channels, frames, frequencies * self.fstride)


class FusedSubPixelConv2D(nn.Module):
    """Export-only one-Conv representation of independent sub-pixel branches."""

    def __init__(self, source: SubPixelConv2D):
        super().__init__()
        if len(source.convs) != source.fstride:
            raise ValueError("Sub-pixel branch count does not match its frequency stride.")
        reference = source.convs[0]
        if any(
            branch.in_channels != reference.in_channels
            or branch.out_channels != reference.out_channels
            or branch.kernel_size != reference.kernel_size
            or branch.stride != reference.stride
            or branch.padding != reference.padding
            or branch.dilation != reference.dilation
            or branch.groups != reference.groups
            or (branch.bias is None) != (reference.bias is None)
            for branch in source.convs[1:]
        ):
            raise ValueError("Sub-pixel branches must have identical convolution layouts for export fusion.")
        if reference.out_channels != source.out_channels or source.out_channels % reference.groups:
            raise ValueError("Sub-pixel output channels are incompatible with grouped export fusion.")
        self.fstride = source.fstride
        self.out_channels = source.out_channels
        self.groups = reference.groups
        self.out_channels_per_group = source.out_channels // reference.groups
        self.convolution = nn.Conv2d(
            reference.in_channels,
            source.out_channels * source.fstride,
            kernel_size=reference.kernel_size,
            stride=reference.stride,
            padding=reference.padding,
            dilation=reference.dilation,
            groups=reference.groups,
            bias=reference.bias is not None,
            device=reference.weight.device,
            dtype=reference.weight.dtype,
        )
        with torch.no_grad():
            branch_weights = tuple(
                branch.weight.reshape(
                    self.groups,
                    self.out_channels_per_group,
                    *branch.weight.shape[1:],
                )
                for branch in source.convs
            )
            packed_weight = torch.stack(branch_weights, dim=2).reshape_as(self.convolution.weight)
            self.convolution.weight.copy_(packed_weight)
            if self.convolution.bias is not None:
                branch_biases = tuple(
                    branch.bias.reshape(self.groups, self.out_channels_per_group)
                    for branch in source.convs
                )
                packed_bias = torch.stack(branch_biases, dim=2).reshape_as(self.convolution.bias)
                self.convolution.bias.copy_(packed_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.convolution(inputs)
        batch, _, frames, frequencies = output.shape
        return output.reshape(
            batch,
            self.groups,
            self.out_channels_per_group,
            self.fstride,
            frames,
            frequencies,
        ).permute(0, 1, 2, 4, 5, 3).reshape(
            batch,
            self.out_channels,
            frames,
            frequencies * self.fstride,
        )


class SubPixelConv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        point_wise_type: str = "cnn",
    ):
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        frequency_padding = kernel_size[1] // 2 if fpad else 0
        layers: List[nn.Module] = []
        if kernel_size[0] > 1:
            layers.append(nn.ConstantPad2d((0, 0, kernel_size[0] - 1, 0), 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(
            SubPixelConv2D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=(0, frequency_padding + dilation - 1),
                fstride=fstride,
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False) if point_wise_type == "cnn" else Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class GroupedLinearEinsum(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        if input_size % groups or hidden_size % groups:
            raise ValueError("Grouped linear dimensions must be divisible by groups.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        self.ws = input_size // groups
        self.weight = nn.Parameter(torch.zeros(groups, input_size // groups, hidden_size // groups))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        leading = inputs.shape[:-1]
        output = inputs.reshape(-1, self.groups, self.ws)
        output = (output.unsqueeze(-2) @ self.weight).squeeze(-2)
        return output.reshape(leading + torch.Size((self.hidden_size,))) + self.bias


class GroupedLinear(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = False):
        super().__init__()
        if input_size % groups or hidden_size % groups:
            raise ValueError("Grouped linear dimensions must be divisible by groups.")
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.shuffle = shuffle if groups > 1 else False
        self.layers = nn.ModuleList(nn.Linear(self.input_size, self.hidden_size) for _ in range(groups))

    def forward(self, inputs: Tensor) -> Tensor:
        output = torch.cat(tuple(layer(split) for layer, split in zip(self.layers, torch.split(inputs, self.input_size, dim=-1))), dim=-1)
        if not self.shuffle:
            return output
        original_shape = output.shape
        return output.view(-1, self.hidden_size, self.groups).transpose(-1, -2).reshape(original_shape)


def convert_grouped_linear_to_einsum(module: nn.Module) -> int:
    """Fold independent grouped Linear modules into static batched MatMul layouts."""

    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, GroupedLinear):
            if child.shuffle:
                raise ValueError(f"Cannot fuse shuffled GroupedLinear {name!r}.")
            fused = GroupedLinearEinsum(child.input_size * child.groups, child.hidden_size * child.groups, child.groups)
            with torch.no_grad():
                fused.weight.copy_(torch.stack(tuple(layer.weight.T for layer in child.layers), dim=0))
                fused.bias.copy_(torch.cat(tuple(layer.bias for layer in child.layers), dim=0))
            setattr(module, name, fused)
            replacements += 1
        else:
            replacements += convert_grouped_linear_to_einsum(child)
    return replacements


class SqueezedGRU_S(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        gru_skip_op: Optional[Callable[..., nn.Module]] = None,
        linear_act_layer: Callable[..., nn.Module] = nn.Identity,
        group_linear_layer: Callable[..., nn.Module] = GroupedLinearEinsum,
        stateful: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(group_linear_layer(input_size, hidden_size, linear_groups), linear_act_layer())
        self.stateful = stateful
        self.gru = nn.ModuleList(GRUCellInternalState(hidden_size, hidden_size) for _ in range(num_layers))
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        self.linear_out = (
            nn.Sequential(group_linear_layer(hidden_size, output_size, linear_groups), linear_act_layer())
            if output_size is not None
            else nn.Identity()
        )

    def state_size(self) -> int:
        return sum(gru_cell.state_size() for gru_cell in self.gru)

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Squeezed GRU state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        states = [gru_cell.initial_state(device=device, dtype=dtype) for gru_cell in self.gru]
        return torch.cat(states, dim=0) if states else torch.zeros(0, dtype=dtype, device=device)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        output = self.linear_in(inputs)
        state_outputs: List[Tensor] = []
        for gru_cell in self.gru:
            if state is None:
                output = gru_cell(output)
            else:
                output, gru_state, offset = gru_cell(output, state=state, offset=offset)
                state_outputs.append(gru_state)
        output = self.linear_out(output)
        if self.gru_skip is not None:
            output = output + self.gru_skip(inputs)
        if state is None:
            return output
        state_out = torch.cat(state_outputs, dim=0) if state_outputs else state.new_zeros(0)
        return output, state_out, offset


class MultiFrameModule(nn.Module):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0, real: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.real = real
        self.pad = nn.ConstantPad3d((0, 0, 0, 0, frame_size - 1 - lookahead, lookahead), 0.0) if real else nn.ConstantPad2d((0, 0, frame_size - 1 - lookahead, lookahead), 0.0)
        self.need_unfold = frame_size > 1
        self.lookahead = lookahead


def df_real(spec: Tensor, coefs: Tensor) -> Tensor:
    real = (spec[..., 0] * coefs[..., 0]).sum(dim=2) - (spec[..., 1] * coefs[..., 1]).sum(dim=2)
    imaginary = (spec[..., 0] * coefs[..., 1]).sum(dim=2) + (spec[..., 1] * coefs[..., 0]).sum(dim=2)
    return torch.stack((real, imaginary), dim=-1)


class DF(MultiFrameModule):
    def __init__(self, freq_bins: int, num_freqs: int, frame_size: int, lookahead: int = 0, conj: bool = False):
        super().__init__(num_freqs, frame_size, lookahead)
        self.conj = conj
        self.spec_buffer = CyclicBuffer([1, 1, 1, freq_bins, 2], time_steps=frame_size, delay_frames=0, time_dim=2)
        self.coefs_buffer = CyclicBuffer([1, 5, 1, num_freqs, 2], time_steps=1, delay_frames=2, time_dim=2)

    def state_size(self) -> int:
        return self.coefs_buffer.state_size() + self.spec_buffer.state_size()

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Deep-filter state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return torch.cat((self.coefs_buffer.initial_state(device=device, dtype=dtype), self.spec_buffer.initial_state(device=device, dtype=dtype)), dim=0)

    def forward(self, spec: Tensor, coefs: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        state_outputs: List[Tensor] = []
        if state is None:
            buffered_coefs = self.coefs_buffer(coefs)
        else:
            buffered_coefs, coefs_state, offset = self.coefs_buffer(coefs, state=state, offset=offset)
            state_outputs.append(coefs_state)
        buffered_coefs = buffered_coefs.permute(0, 2, 1, 3, 4)
        if state is None:
            buffered_spec = self.spec_buffer(spec)
        else:
            buffered_spec, spec_state, offset = self.spec_buffer(spec, state=state, offset=offset)
            state_outputs.append(spec_state)
        filtered = df_real(buffered_spec[..., :self.num_freqs, :], buffered_coefs).unsqueeze(1)
        output = torch.cat((filtered, buffered_spec[:, :, 2:3, self.num_freqs:]), dim=3)
        if state is None:
            return output
        return output, torch.cat(state_outputs, dim=0), offset


class Add(nn.Module):
    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return first + second


class Concat(nn.Module):
    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return torch.cat((first, second), dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
        nb_erb: int,
        nb_df: int,
        conv_ch: int,
        conv_kernel_inp: Tuple[int, int],
        conv_kernel: Tuple[int, int],
        enc_concat: bool,
        emb_hidden_dim: int,
        enc_lin_groups: int,
        emb_num_layers: int,
        lin_groups: int,
        emb_gru_skip_enc: str = "none",
        stateful: bool = False,
        group_linear_type: str = "einsum",
        point_wise_type: str = "cnn",
        separable_first_conv: bool = True,
        lsnr_min: float = -15.0,
        lsnr_max: float = 35.0,
        dprnn_num_blocks: int = 0,
    ):
        super().__init__()
        if nb_erb % 4:
            raise ValueError("erb_bins must be divisible by four.")
        if conv_kernel_inp[0] > 1:
            self.erb_conv0_buffer = CyclicBuffer([1, 1, 1, nb_erb], time_steps=conv_kernel_inp[0], time_dim=2)
            self.df_conv0_buffer = CyclicBuffer([1, 2, 1, nb_df], time_steps=conv_kernel_inp[0], time_dim=2)
        else:
            self.erb_conv0_buffer = nn.Identity()
            self.df_conv0_buffer = nn.Identity()
        self.erb_conv0 = Conv2dNormAct(1, conv_ch, conv_kernel_inp, bias=False, separable=separable_first_conv, point_wise_type=point_wise_type)
        conv_layer = partial(Conv2dNormAct, in_ch=conv_ch, out_ch=conv_ch, kernel_size=conv_kernel, bias=False, separable=True, point_wise_type=point_wise_type)
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(2, conv_ch, conv_kernel_inp, bias=False, separable=separable_first_conv, point_wise_type=point_wise_type)
        self.df_conv1 = conv_layer(fstride=2)
        self.dprnn_erb = DPRNN(nb_erb // 4, conv_ch, conv_ch, conv_ch, dprnn_num_blocks, stateful) if dprnn_num_blocks else nn.Identity()
        self.dprnn_df = DPRNN(nb_df // 2, conv_ch, conv_ch, conv_ch, dprnn_num_blocks, stateful) if dprnn_num_blocks else nn.Identity()
        self.erb_bins = nb_erb
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4
        group_linear = GroupedLinearEinsum if group_linear_type == "einsum" else GroupedLinear
        self.df_fc_emb = nn.Sequential(group_linear(conv_ch * nb_df // 2, self.emb_in_dim, enc_lin_groups), nn.ReLU(inplace=True))
        if enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        if emb_gru_skip_enc == "none":
            skip_op = None
        elif emb_gru_skip_enc == "identity":
            if self.emb_in_dim != self.emb_out_dim:
                raise ValueError("Encoder GRU identity skip dimensions do not match.")
            skip_op = nn.Identity
        elif emb_gru_skip_enc == "groupedlinear":
            skip_op = partial(group_linear, input_size=self.emb_out_dim, hidden_size=self.emb_out_dim, groups=lin_groups)
        else:
            raise ValueError(f"Unsupported encoder GRU skip: {emb_gru_skip_enc}.")
        self.emb_gru = SqueezedGRU_S(self.emb_in_dim, self.emb_dim, output_size=self.emb_out_dim, num_layers=1, gru_skip_op=skip_op, linear_groups=lin_groups, linear_act_layer=partial(nn.ReLU, inplace=True), group_linear_layer=group_linear, stateful=stateful)
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def state_size(self) -> int:
        return sum(module.state_size() for module in (self.erb_conv0_buffer, self.dprnn_erb, self.df_conv0_buffer, self.dprnn_df, self.emb_gru) if hasattr(module, "state_size"))

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Encoder state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        states = [module.initial_state(device=device, dtype=dtype) for module in (self.erb_conv0_buffer, self.dprnn_erb, self.df_conv0_buffer, self.dprnn_df, self.emb_gru) if hasattr(module, "initial_state")]
        return torch.cat(states, dim=0) if states else torch.zeros(0, dtype=dtype, device=device)

    def initial_stream_state(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> EncoderStreamingState:
        empty = torch.zeros(0, dtype=dtype, device=device)
        return (
            self.erb_conv0_buffer.initial_state(device=device, dtype=dtype)
            if hasattr(self.erb_conv0_buffer, "initial_state")
            else empty,
            self.dprnn_erb.initial_state(device=device, dtype=dtype)
            if hasattr(self.dprnn_erb, "initial_state")
            else empty,
            self.df_conv0_buffer.initial_state(device=device, dtype=dtype)
            if hasattr(self.df_conv0_buffer, "initial_state")
            else empty,
            self.dprnn_df.initial_state(device=device, dtype=dtype)
            if hasattr(self.dprnn_df, "initial_state")
            else empty,
            self.emb_gru.initial_state(device=device, dtype=dtype),
        )

    def forward(self, feat_erb: Tensor, feat_spec: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        states: List[Tensor] = []
        if state is None or not hasattr(self.erb_conv0_buffer, "state_size"):
            buffered_erb = self.erb_conv0_buffer(feat_erb)
        else:
            buffered_erb, erb_state, offset = self.erb_conv0_buffer(feat_erb, state=state, offset=offset)
            states.append(erb_state)
        e0 = self.erb_conv0(buffered_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        if state is None or not hasattr(self.dprnn_erb, "state_size"):
            erb_dprnn = self.dprnn_erb(e3)
        else:
            erb_dprnn, dprnn_erb_state, offset = self.dprnn_erb(e3, state=state, offset=offset)
            states.append(dprnn_erb_state)
        if state is None or not hasattr(self.df_conv0_buffer, "state_size"):
            buffered_spec = self.df_conv0_buffer(feat_spec)
        else:
            buffered_spec, spec_state, offset = self.df_conv0_buffer(feat_spec, state=state, offset=offset)
            states.append(spec_state)
        c0 = self.df_conv0(buffered_spec)
        c1 = self.df_conv1(c0)
        if state is None or not hasattr(self.dprnn_df, "state_size"):
            df_dprnn = self.dprnn_df(c1)
        else:
            df_dprnn, dprnn_df_state, offset = self.dprnn_df(c1, state=state, offset=offset)
            states.append(dprnn_df_state)
        cemb = self.df_fc_emb(df_dprnn.permute(0, 2, 3, 1).flatten(1))
        emb = self.combine(erb_dprnn.permute(0, 2, 3, 1).flatten(1), cemb)
        if state is None:
            emb = self.emb_gru(emb)
        else:
            emb, emb_state, offset = self.emb_gru(emb, state=state, offset=offset)
            states.append(emb_state)
        lsnr = self.lsnr_fc(emb).squeeze(-1) * self.lsnr_scale + self.lsnr_offset
        if state is None:
            return e0, e1, e2, e3, emb, c0, lsnr
        return e0, e1, e2, e3, emb, c0, lsnr, torch.cat(states, dim=0) if states else state.new_zeros(0), offset

    def forward_stream(
        self,
        feat_erb: Tensor,
        feat_spec: Tensor,
        state: EncoderStreamingState,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, EncoderStreamingState]:
        """Run the private export path without repacking independent encoder states."""

        erb_buffer_state, erb_dprnn_state, spec_buffer_state, df_dprnn_state, emb_state = state
        if hasattr(self.erb_conv0_buffer, "state_size"):
            buffered_erb, erb_buffer_state, _ = self.erb_conv0_buffer(
                feat_erb,
                state=erb_buffer_state,
                offset=0,
            )
        else:
            buffered_erb = self.erb_conv0_buffer(feat_erb)
        e0 = self.erb_conv0(buffered_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        if hasattr(self.dprnn_erb, "state_size"):
            erb_dprnn, erb_dprnn_state, _ = self.dprnn_erb(e3, state=erb_dprnn_state, offset=0)
        else:
            erb_dprnn = self.dprnn_erb(e3)
        if hasattr(self.df_conv0_buffer, "state_size"):
            buffered_spec, spec_buffer_state, _ = self.df_conv0_buffer(
                feat_spec,
                state=spec_buffer_state,
                offset=0,
            )
        else:
            buffered_spec = self.df_conv0_buffer(feat_spec)
        c0 = self.df_conv0(buffered_spec)
        c1 = self.df_conv1(c0)
        if hasattr(self.dprnn_df, "state_size"):
            df_dprnn, df_dprnn_state, _ = self.dprnn_df(c1, state=df_dprnn_state, offset=0)
        else:
            df_dprnn = self.dprnn_df(c1)
        cemb = self.df_fc_emb(df_dprnn.permute(0, 2, 3, 1).flatten(1))
        emb = self.combine(erb_dprnn.permute(0, 2, 3, 1).flatten(1), cemb)
        emb, emb_state, _ = self.emb_gru(emb, state=emb_state, offset=0)
        lsnr = self.lsnr_fc(emb).squeeze(-1) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr, (
            erb_buffer_state,
            erb_dprnn_state,
            spec_buffer_state,
            df_dprnn_state,
            emb_state,
        )


class ErbDecoder(nn.Module):
    def __init__(self, nb_erb: int, conv_ch: int, conv_kernel: Tuple[int, int], convt_kernel: Tuple[int, int], emb_num_layers: int, emb_hidden_dim: int, lin_groups: int, emb_gru_skip: str = "none", stateful: bool = False, group_linear_type: str = "einsum", upsample_conv_type: str = "transpose", point_wise_type: str = "cnn"):
        super().__init__()
        if nb_erb % 8:
            raise ValueError("erb_bins must be divisible by eight.")
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4
        group_linear = GroupedLinearEinsum if group_linear_type == "einsum" else GroupedLinear
        if emb_gru_skip == "none":
            skip_op = None
        elif emb_gru_skip == "identity":
            if self.emb_in_dim != self.emb_out_dim:
                raise ValueError("ERB decoder GRU identity skip dimensions do not match.")
            skip_op = nn.Identity
        elif emb_gru_skip == "groupedlinear":
            skip_op = partial(group_linear, input_size=self.emb_in_dim, hidden_size=self.emb_out_dim, groups=lin_groups)
        else:
            raise ValueError(f"Unsupported ERB decoder GRU skip: {emb_gru_skip}.")
        self.emb_gru = SqueezedGRU_S(self.emb_in_dim, self.emb_dim, output_size=self.emb_out_dim, num_layers=emb_num_layers, gru_skip_op=skip_op, linear_groups=lin_groups, linear_act_layer=partial(nn.ReLU, inplace=True), group_linear_layer=group_linear, stateful=stateful)
        upsample_layer = ConvTranspose2dNormAct if upsample_conv_type == "transpose" else SubPixelConv2dNormAct
        tconv = partial(upsample_layer, kernel_size=convt_kernel, bias=False, separable=True, point_wise_type=point_wise_type)
        conv = partial(Conv2dNormAct, bias=False, separable=True, point_wise_type=point_wise_type)
        self.conv3p = conv(conv_ch, conv_ch, kernel_size=1)
        self.convt3 = conv(conv_ch, conv_ch, kernel_size=conv_kernel)
        self.conv2p = conv(conv_ch, conv_ch, kernel_size=1)
        self.convt2 = tconv(conv_ch, conv_ch, fstride=2)
        self.conv1p = conv(conv_ch, conv_ch, kernel_size=1)
        self.convt1 = tconv(conv_ch, conv_ch, fstride=2)
        self.conv0p = conv(conv_ch, conv_ch, kernel_size=1)
        self.conv0_out = conv(conv_ch, 1, kernel_size=conv_kernel, activation_layer=nn.Sigmoid)

    def state_size(self) -> int:
        return self.emb_gru.state_size()

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"ERB decoder state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return self.emb_gru.initial_state(device=device, dtype=dtype)

    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        batch, _, frames, frequency_div_8 = e3.shape
        if state is None:
            emb = self.emb_gru(emb)
            emb_state = None
        else:
            emb, emb_state, offset = self.emb_gru(emb, state=state, offset=offset)
        emb = emb.view(batch, frames, frequency_div_8, -1).permute(0, 3, 1, 2)
        e3 = self.convt3(self.conv3p(e3) + emb)
        e2 = self.convt2(self.conv2p(e2) + e3)
        e1 = self.convt1(self.conv1p(e1) + e2)
        mask = self.conv0_out(self.conv0p(e0) + e1)
        if state is None:
            return mask
        return mask, emb_state if emb_state is not None else state.new_zeros(0), offset


class DfOutputReshapeMF(nn.Module):
    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        shape = list(coefs.shape)
        shape[-1] = -1
        shape.append(2)
        return coefs.view(shape).permute(0, 3, 1, 2, 4)


class DfDecoder(nn.Module):
    def __init__(self, nb_erb: int, nb_df: int, conv_ch: int, df_hidden_dim: int, emb_hidden_dim: int, df_order: int, df_num_layers: int, df_pathway_kernel_size_t: int, lin_groups: int, df_gru_skip: str = "groupedlinear", stateful: bool = False, group_linear_type: str = "einsum", point_wise_type: str = "cnn"):
        super().__init__()
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = df_hidden_dim
        self.df_n_hidden = df_hidden_dim
        self.df_n_layers = df_num_layers
        self.df_order = df_order
        self.df_bins = nb_df
        self.df_out_ch = df_order * 2
        self.df_convp_buffer = CyclicBuffer([1, conv_ch, 1, nb_df], time_steps=df_pathway_kernel_size_t, time_dim=2)
        self.df_convp = Conv2dNormAct(conv_ch, self.df_out_ch, kernel_size=(df_pathway_kernel_size_t, 1), fstride=1, bias=False, separable=True, point_wise_type=point_wise_type)
        group_linear = GroupedLinearEinsum if group_linear_type == "einsum" else GroupedLinear
        self.df_gru = SqueezedGRU_S(self.emb_in_dim, self.emb_dim, num_layers=df_num_layers, gru_skip_op=None, linear_act_layer=partial(nn.ReLU, inplace=True), group_linear_layer=group_linear, stateful=stateful)
        if df_gru_skip == "none":
            self.df_skip = None
        elif df_gru_skip == "identity":
            if emb_hidden_dim != df_hidden_dim:
                raise ValueError("DF decoder identity skip dimensions do not match.")
            self.df_skip = nn.Identity()
        elif df_gru_skip == "groupedlinear":
            self.df_skip = group_linear(self.emb_in_dim, self.emb_dim, groups=lin_groups)
        else:
            raise ValueError(f"Unsupported DF decoder GRU skip: {df_gru_skip}.")
        self.df_out = nn.Sequential(group_linear(self.df_n_hidden, self.df_bins * self.df_out_ch, groups=lin_groups), nn.Tanh())

    def state_size(self) -> int:
        return self.df_gru.state_size() + self.df_convp_buffer.state_size()

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"DF decoder state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return torch.cat((self.df_gru.initial_state(device=device, dtype=dtype), self.df_convp_buffer.initial_state(device=device, dtype=dtype)), dim=0)

    def initial_stream_state(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> DfDecoderStreamingState:
        return (
            self.df_gru.initial_state(device=device, dtype=dtype),
            self.df_convp_buffer.initial_state(device=device, dtype=dtype),
        )

    def forward(self, emb: Tensor, c0: Tensor, state: Optional[Tensor] = None, offset: int = 0):
        batch = emb.shape[0]
        states: List[Tensor] = []
        if state is None:
            coefficients = self.df_gru(emb)
        else:
            coefficients, gru_state, offset = self.df_gru(emb, state=state, offset=offset)
            states.append(gru_state)
        if self.df_skip is not None:
            coefficients = coefficients + self.df_skip(emb)
        if state is None:
            buffered_c0 = self.df_convp_buffer(c0)
        else:
            buffered_c0, conv_state, offset = self.df_convp_buffer(c0, state=state, offset=offset)
            states.append(conv_state)
        c0 = self.df_convp(buffered_c0).permute(0, 2, 3, 1)
        frames = c0.shape[1]
        coefficients = self.df_out(coefficients)
        if coefficients.dim() == 2:
            if frames != 1:
                raise ValueError("2D DF coefficients are only valid for one frame.")
            coefficients = coefficients.unsqueeze(1)
        coefficients = coefficients.view(batch, frames, self.df_bins, self.df_out_ch) + c0
        if state is None:
            return coefficients
        return coefficients, torch.cat(states, dim=0) if states else state.new_zeros(0), offset

    def forward_stream(
        self,
        emb: Tensor,
        c0: Tensor,
        state: DfDecoderStreamingState,
    ) -> Tuple[Tensor, DfDecoderStreamingState]:
        """Run the private export path without repacking its unrelated state partitions."""

        gru_state, conv_state = state
        batch = emb.shape[0]
        coefficients, gru_state, _ = self.df_gru(emb, state=gru_state, offset=0)
        if self.df_skip is not None:
            coefficients = coefficients + self.df_skip(emb)
        buffered_c0, conv_state, _ = self.df_convp_buffer(c0, state=conv_state, offset=0)
        c0 = self.df_convp(buffered_c0).permute(0, 2, 3, 1)
        frames = c0.shape[1]
        coefficients = self.df_out(coefficients)
        if coefficients.dim() == 2:
            if frames != 1:
                raise ValueError("2D DF coefficients are only valid for one frame.")
            coefficients = coefficients.unsqueeze(1)
        coefficients = coefficients.view(batch, frames, self.df_bins, self.df_out_ch) + c0
        return coefficients, (gru_state, conv_state)


class DPDFNet(nn.Module):
    """Published 16 kHz DPDFNet architecture with the explicit streaming-state forward path."""

    def __init__(
        self,
        n_fft: int = 320,
        win_length: float = 0.02,
        hop_length: float = 0.01,
        samplerate: int = 16000,
        freq_df: int = 4800,
        nb_erb: int = 32,
        min_nb_freqs: int = 1,
        erb_to_db: bool = True,
        alpha_norm: float = 0.98,
        conv_ch: int = 64,
        conv_kernel_inp: Tuple[int, int] = (3, 3),
        conv_kernel: Tuple[int, int] = (1, 3),
        convt_kernel: Tuple[int, int] = (1, 3),
        enc_gru_dim: int = 256,
        erb_dec_gru_dim: int = 256,
        df_dec_gru_dim: int = 256,
        enc_lin_groups: int = 32,
        emb_gru_skip_enc: str = "none",
        lin_groups: int = 16,
        df_order: int = 5,
        df_pathway_kernel_size_t: int = 5,
        df_gru_skip: str = "groupedlinear",
        df_lookahead: int = 2,
        conv_lookahead: int = 2,
        enc_concat: bool = True,
        emb_num_layers: int = 2,
        df_num_layers: int = 2,
        stateful: bool = False,
        mask_method: str = "before_df",
        erb_dynamic_var: bool = False,
        norm_stateful: bool = False,
        upsample_conv_type: str = "subpixel",
        group_linear_type: str = "loop",
        point_wise_type: str = "cnn",
        separable_first_conv: bool = True,
        lsnr_min: float = -15.0,
        lsnr_max: float = 35.0,
        dprnn_num_blocks: int = 0,
    ):
        super().__init__()
        if upsample_conv_type not in {"transpose", "subpixel"}:
            raise ValueError(f"Unsupported upsample type: {upsample_conv_type}.")
        if group_linear_type not in {"einsum", "loop"}:
            raise ValueError(f"Unsupported grouped linear type: {group_linear_type}.")
        if point_wise_type not in {"cnn", "linear"}:
            raise ValueError(f"Unsupported pointwise type: {point_wise_type}.")
        self.mask_method = mask_method
        self.nb_erb = nb_erb
        self.erb_to_db = erb_to_db
        filter_bank = torch.tensor(erb_filter_banks(n_filters=nb_erb, nfft=n_fft, fs=samplerate, min_nb_freqs=min_nb_freqs), dtype=torch.float32)
        inverse_filter_bank = filter_bank.clone().t()
        filter_bank = filter_bank / filter_bank.sum(-1, keepdim=True)
        self.register_buffer("erb_fb", filter_bank.t())
        self.register_buffer("erb_inv_fb", inverse_filter_bank.t())
        window_size = int(win_length * samplerate)
        hop_size = int(hop_length * samplerate)
        self.n_fft = n_fft
        self.window_length = window_size
        self.hop_length = hop_size
        self.sample_rate = samplerate
        self.stft = CheckpointStftState(n_fft, window_size, hop_size, vorbis_window(window_size))
        self.istft = CheckpointIstftState(n_fft, window_size, hop_size, vorbis_window(window_size))
        self.istft_norm = CheckpointIstftState(n_fft, window_size, hop_size, vorbis_window(window_size))
        self.wnorm = get_wnorm(window_size, hop_size)
        if nb_erb % 8:
            raise ValueError("erb_bins must be divisible by eight.")
        self.df_lookahead = df_lookahead
        self.freq_bins = n_fft // 2 + 1
        self.nb_df = int((freq_df / (samplerate // 2)) * self.freq_bins)
        self.erb_bins = nb_erb
        self.pad_feat = nn.ConstantPad2d((0, 0, -conv_lookahead, conv_lookahead), 0.0) if conv_lookahead > 0 else nn.Identity()
        self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -df_lookahead, df_lookahead), 0.0) if df_lookahead > 0 else nn.Identity()
        self.enc = Encoder(nb_erb, self.nb_df, conv_ch, conv_kernel_inp, conv_kernel, enc_concat, enc_gru_dim, enc_lin_groups, emb_num_layers - 1, lin_groups, emb_gru_skip_enc, stateful, group_linear_type, point_wise_type, separable_first_conv, lsnr_min, lsnr_max, dprnn_num_blocks)
        self.erb_dec = ErbDecoder(nb_erb, conv_ch, conv_kernel, convt_kernel, emb_num_layers, erb_dec_gru_dim, lin_groups, emb_gru_skip_enc, stateful, group_linear_type, upsample_conv_type, point_wise_type)
        self.mask = Mask(self.erb_inv_fb)
        self.df_order = df_order
        self.df_op = DF(self.freq_bins, self.nb_df, df_order, self.df_lookahead)
        self.df_dec = DfDecoder(nb_erb, self.nb_df, conv_ch, df_dec_gru_dim, erb_dec_gru_dim, df_order, df_num_layers, df_pathway_kernel_size_t, lin_groups, df_gru_skip, stateful, group_linear_type, "cnn")
        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.nb_df)
        self.erb_norm = ErbNorm(nb_erb, alpha_norm, dynamic_var=erb_dynamic_var, stateful=norm_stateful)
        self.spec_norm = SpecNorm(self.nb_df, alpha_norm, stateful=norm_stateful)
        self._export_prepared = False

    def state_size(self) -> int:
        return self.erb_norm.state_size() + self.spec_norm.state_size() + self.enc.state_size() + self.erb_dec.state_size() + self.df_dec.state_size() + self.mask.state_size() + self.df_op.state_size()

    def initial_state(self, state: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"DPDFNet state must contain {self.state_size()} values, got {state.numel()}.")
            return state
        return torch.cat((self.erb_norm.initial_state(device=device, dtype=dtype), self.spec_norm.initial_state(device=device, dtype=dtype), self.enc.initial_state(device=device, dtype=dtype), self.erb_dec.initial_state(device=device, dtype=dtype), self.df_dec.initial_state(device=device, dtype=dtype), self.mask.initial_state(device=device, dtype=dtype), self.df_op.initial_state(device=device, dtype=dtype)), dim=0)

    def initial_stream_state(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> StreamingState:
        """Return the private export-state partitions in legacy flat-state order."""

        return (
            self.erb_norm.initial_state(device=device, dtype=dtype),
            self.spec_norm.initial_state(device=device, dtype=dtype),
            self.enc.initial_stream_state(device=device, dtype=dtype),
            self.erb_dec.initial_state(device=device, dtype=dtype),
            self.df_dec.initial_stream_state(device=device, dtype=dtype),
            self.mask.initial_state(device=device, dtype=dtype),
            self.df_op.initial_state(device=device, dtype=dtype),
        )

    def forward(self, spec: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        expected_state_size = self.state_size()
        if state is None:
            state = self.initial_state(device=spec.device, dtype=torch.float32)
        elif state.ndim != 1 or state.numel() != expected_state_size or state.device != spec.device or state.dtype != torch.float32:
            raise ValueError("DPDFNet requires a float32 one-dimensional state tensor matching the model state size.")
        state_outputs: List[Tensor] = []
        offset = 0
        spec, feat_erb, feat_spec, feature_state, offset = self._feature_extraction(spec, state=state, offset=offset)
        state_outputs.append(feature_state)
        feat_spec = feat_spec.permute(0, 3, 1, 2)
        e0, e1, e2, e3, emb, c0, _lsnr, encoder_state, offset = self.enc(feat_erb, feat_spec, state=state, offset=offset)
        state_outputs.append(encoder_state)
        mask, erb_decoder_state, offset = self.erb_dec(emb, e3, e2, e1, e0, state=state, offset=offset)
        state_outputs.append(erb_decoder_state)
        coefs, df_decoder_state, offset = self.df_dec(emb, c0, state=state, offset=offset)
        state_outputs.append(df_decoder_state)
        coefs = self.df_out_transform(coefs)
        if self.mask_method != "before_df":
            raise ValueError("The standalone exporter supports the published before_df masking mode only.")
        masked_spec, mask_state, offset = self.mask(spec, mask, state=state, offset=offset)
        state_outputs.append(mask_state)
        enhanced, df_state, offset = self.df_op(masked_spec, coefs, state=state, offset=offset)
        state_outputs.append(df_state)
        if offset != expected_state_size:
            raise RuntimeError(f"DPDFNet consumed {offset} state values, expected {expected_state_size}.")
        return enhanced.squeeze(1), torch.cat(state_outputs, dim=0)

    def forward_stream(self, spec: Tensor, state: StreamingState) -> Tuple[Tensor, StreamingState]:
        """Run one frame without materializing the legacy flat state between frames."""

        if len(state) != 7:
            raise ValueError(f"DPDFNet streaming state must contain 7 partitions, got {len(state)}.")
        erb_norm_state, spec_norm_state, encoder_state, erb_decoder_state, df_decoder_state, mask_state, df_state = state
        spec, feat_erb, feat_spec, erb_norm_state, spec_norm_state = self._feature_extraction_stream(
            spec,
            erb_norm_state,
            spec_norm_state,
        )
        feat_spec = feat_spec.permute(0, 3, 1, 2)
        e0, e1, e2, e3, emb, c0, _lsnr, encoder_state = self.enc.forward_stream(
            feat_erb,
            feat_spec,
            encoder_state,
        )
        mask, erb_decoder_state, erb_decoder_offset = self.erb_dec(
            emb,
            e3,
            e2,
            e1,
            e0,
            state=erb_decoder_state,
            offset=0,
        )
        coefs, df_decoder_state = self.df_dec.forward_stream(emb, c0, df_decoder_state)
        coefs = self.df_out_transform(coefs)
        if self.mask_method != "before_df":
            raise ValueError("The standalone exporter supports the published before_df masking mode only.")
        masked_spec, mask_state, mask_offset = self.mask(spec, mask, state=mask_state, offset=0)
        enhanced, df_state, df_offset = self.df_op(masked_spec, coefs, state=df_state, offset=0)
        return enhanced.squeeze(1), (
            erb_norm_state,
            spec_norm_state,
            encoder_state,
            erb_decoder_state,
            df_decoder_state,
            mask_state,
            df_state,
        )

    def _feature_extraction(self, spec: Tensor, state: Tensor, offset: int):
        feat_erb = get_pow(spec) @ self.erb_fb
        if self.erb_to_db:
            feat_erb = to_db(feat_erb)
        feat_spec = spec[..., :self.nb_df, :]
        feat_erb, erb_state, offset = self.erb_norm(feat_erb, state=state, offset=offset)
        feat_spec, spec_state, offset = self.spec_norm(feat_spec, state=state, offset=offset)
        return as_real(spec.unsqueeze(1)), feat_erb.unsqueeze(1), feat_spec, torch.cat((erb_state, spec_state), dim=0), offset

    def _feature_extraction_stream(self, spec: Tensor, erb_state: Tensor, spec_state: Tensor):
        feat_erb = get_pow(spec) @ self.erb_fb
        if self.erb_to_db:
            feat_erb = to_db(feat_erb)
        feat_spec = spec[..., :self.nb_df, :]
        feat_erb, erb_state, erb_offset = self.erb_norm(feat_erb, state=erb_state, offset=0)
        feat_spec, spec_state, spec_offset = self.spec_norm(feat_spec, state=spec_state, offset=0)
        return as_real(spec.unsqueeze(1)), feat_erb.unsqueeze(1), feat_spec, erb_state, spec_state

    def prepare_for_export_(self):
        if self._export_prepared:
            return 0, 0, 0, 0
        grouped_linear = convert_grouped_linear_to_einsum(self) if FUSE_GROUPED_LINEAR else 0
        grouped_convolution = fuse_grouped_convolutions_for_export_(self) if FUSE_GROUPED_CONVOLUTIONS else 0
        subpixel_convolution = fuse_subpixel_convolutions_for_export_(self) if FUSE_SUBPIXEL_CONVOLUTIONS else 0
        batch_norm = fuse_batch_norms_for_export_(self) if FUSE_BATCH_NORMALIZATION else 0
        self._export_prepared = True
        return grouped_linear, grouped_convolution, subpixel_convolution, batch_norm


class DPDFNet_CUSTOM(nn.Module):
    """ONNX wrapper that keeps DPDFNet's native spectrum-and-state API explicit."""

    def __init__(self, dpdfnet: DPDFNet):
        super().__init__()
        self.dpdfnet = dpdfnet
        self.register_buffer("wnorm", torch.tensor(float(dpdfnet.wnorm), dtype=torch.float32))
        self.register_buffer("inv_wnorm", torch.tensor(1.0 / float(dpdfnet.wnorm), dtype=torch.float32))

    def forward(self, noisy_spec: Tensor, state_in: Tensor) -> Tuple[Tensor, Tensor]:
        denoised_spec, state_out = self.dpdfnet(noisy_spec * self.wnorm, state_in)
        return denoised_spec * self.inv_wnorm, state_out

    def forward_stream(self, noisy_spec: Tensor, state_in: StreamingState) -> Tuple[Tensor, StreamingState]:
        denoised_spec, state_out = self.dpdfnet.forward_stream(noisy_spec * self.wnorm, state_in)
        return denoised_spec * self.inv_wnorm, state_out


class FixedTimePrefix(nn.Module):
    """Prepend a static zero history to the time axis of a fixed batch-one tensor."""

    def __init__(self, prefix_shape: tuple[int, ...]):
        super().__init__()
        if len(prefix_shape) < 3 or prefix_shape[0] != 1:
            raise ValueError("FixedTimePrefix requires a batch-one prefix with a time dimension.")
        self.register_buffer("prefix", torch.zeros(prefix_shape, dtype=torch.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat((self.prefix, inputs), dim=2)


class FixedTimeDelay(nn.Module):
    """Emit a fixed-length delayed sequence from a static zero-prefixed history."""

    def __init__(self, prefix_shape: tuple[int, ...], frame_count: int):
        super().__init__()
        self.prefix = FixedTimePrefix(prefix_shape)
        self.frame_count = frame_count

    def forward(self, inputs: Tensor) -> Tensor:
        return self.prefix(inputs).narrow(2, 0, self.frame_count)


class FixedTimeHistory(nn.Module):
    """Materialize a causal fixed-width time window without a per-frame buffer Loop."""

    def __init__(self, prefix_shape: tuple[int, ...], frame_count: int, window_size: int):
        super().__init__()
        if window_size < 1:
            raise ValueError("FixedTimeHistory requires a positive window size.")
        self.prefix = FixedTimePrefix(prefix_shape)
        self.frame_count = frame_count
        self.window_size = window_size

    def forward(self, inputs: Tensor) -> Tensor:
        padded = self.prefix(inputs)
        frames = tuple(
            padded.narrow(2, offset, self.frame_count)
            for offset in range(self.window_size)
        )
        return torch.stack(frames, dim=3)


class SqueezedGRUSequence(nn.Module):
    """Run a checkpoint SqueezedGRU_S across all static frames with one native GRU."""

    def __init__(self, source: SqueezedGRU_S, batch_size: int = 1):
        super().__init__()
        self.linear_in = source.linear_in
        self.gru = GRUCellSequence(source.gru, batch_size=batch_size)
        self.linear_out = source.linear_out
        self.gru_skip = source.gru_skip

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.linear_in(inputs)
        output = self.gru(output)
        output = self.linear_out(output)
        if self.gru_skip is not None:
            output = output + self.gru_skip(inputs)
        return output


class DPRNNSequence(nn.Module):
    """Run DPDFNet DPRNN blocks over the whole static frame sequence at once."""

    def __init__(self, source: DPRNN, frame_count: int, batch_size: int = 1):
        super().__init__()
        if not source.blocks:
            raise ValueError("DPRNNSequence requires at least one DPRNN block.")
        self.input_proj = source.input_proj
        self.output_proj = source.output_proj
        self.blocks = source.blocks
        self.inter_grus = nn.ModuleList(
            GRUCellSequence((block.inter_gru,), batch_size=batch_size * block.num_feat)
            for block in source.blocks
        )
        self.frame_count = frame_count
        self.batch_size = batch_size
        self.num_feat = source.blocks[0].num_feat
        self.hidden_dim = source.blocks[0].hidden_dim

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.input_proj(inputs)
        for block, inter_gru in zip(self.blocks, self.inter_grus):
            intra = output.permute(0, 2, 3, 1).reshape(
                self.batch_size * self.frame_count,
                self.num_feat,
                self.hidden_dim,
            )
            intra, _ = block.intra_gru(intra)
            intra = block.ln_intra(block.fc_intra(intra))
            intra = intra.reshape(
                self.batch_size,
                self.frame_count,
                self.num_feat,
                self.hidden_dim,
            ).permute(0, 3, 1, 2)
            residual = output + intra

            inter = residual.permute(0, 3, 2, 1).reshape(
                self.batch_size * self.num_feat,
                self.frame_count,
                self.hidden_dim,
            )
            inter = inter_gru(inter)
            inter = block.ln_inter(block.fc_inter(inter))
            inter = inter.reshape(
                self.batch_size,
                self.num_feat,
                self.frame_count,
                self.hidden_dim,
            ).permute(0, 3, 2, 1)
            output = residual + inter
        return self.output_proj(output)


class DPDFNet_SEQUENCE(nn.Module):
    """Static batch-one DPDFNet sequence path without an ONNX Loop over frames."""

    def __init__(self, spectral_model: DPDFNet_CUSTOM, frame_count: int):
        super().__init__()
        if frame_count < 1:
            raise ValueError("DPDFNet_SEQUENCE requires a positive frame count.")
        self.dpdfnet = spectral_model.dpdfnet
        self.frame_count = frame_count
        self.freq_bins = self.dpdfnet.freq_bins
        self.nb_df = self.dpdfnet.nb_df
        self.nb_erb = self.dpdfnet.nb_erb
        self.register_buffer("wnorm", spectral_model.wnorm.detach().clone())
        self.register_buffer("inv_wnorm", spectral_model.inv_wnorm.detach().clone())

        encoder = self.dpdfnet.enc
        df_decoder = self.dpdfnet.df_dec
        if not isinstance(encoder.erb_conv0_buffer, CyclicBuffer) or not isinstance(
            encoder.df_conv0_buffer, CyclicBuffer
        ):
            raise ValueError("DPDFNet_SEQUENCE requires checkpoint causal encoder buffers.")
        if not isinstance(df_decoder.df_convp_buffer, CyclicBuffer):
            raise ValueError("DPDFNet_SEQUENCE requires the checkpoint DF convolution buffer.")
        if not isinstance(self.dpdfnet.mask.spec_buffer, CyclicBuffer):
            raise ValueError("DPDFNet_SEQUENCE requires the checkpoint mask buffer.")
        if not isinstance(self.dpdfnet.df_op.coefs_buffer, CyclicBuffer) or not isinstance(
            self.dpdfnet.df_op.spec_buffer, CyclicBuffer
        ):
            raise ValueError("DPDFNet_SEQUENCE requires checkpoint deep-filter buffers.")

        self.erb_ema = FixedSequenceEMA(self.dpdfnet.erb_norm.alpha, frame_count)
        self.spec_ema = FixedSequenceEMA(self.dpdfnet.spec_norm.alpha, frame_count)
        self.erb_conv_prefix = FixedTimePrefix(
            (1, 1, encoder.erb_conv0_buffer.time_steps - 1, self.nb_erb)
        )
        self.df_conv_prefix = FixedTimePrefix(
            (1, 2, encoder.df_conv0_buffer.time_steps - 1, self.nb_df)
        )
        self.erb_dprnn = DPRNNSequence(encoder.dprnn_erb, frame_count)
        self.df_dprnn = DPRNNSequence(encoder.dprnn_df, frame_count)
        self.encoder_emb_gru = SqueezedGRUSequence(encoder.emb_gru)
        self.erb_decoder_gru = SqueezedGRUSequence(self.dpdfnet.erb_dec.emb_gru)
        self.df_decoder_gru = SqueezedGRUSequence(df_decoder.df_gru)
        self.df_convp_prefix = FixedTimePrefix(
            (1, df_decoder.df_convp_buffer.shape_tf[2], df_decoder.df_convp_buffer.time_steps - 1, self.nb_df)
        )
        self.mask_delay = FixedTimeDelay(
            (1, 1, self.dpdfnet.mask.spec_buffer.capacity - 1, self.freq_bins, 2),
            frame_count,
        )
        self.coefs_delay = FixedTimeDelay(
            (1, self.dpdfnet.df_order, self.dpdfnet.df_op.coefs_buffer.capacity - 1, self.nb_df, 2),
            frame_count,
        )
        self.spec_history = FixedTimeHistory(
            (1, 1, self.dpdfnet.df_op.spec_buffer.time_steps - 1, self.freq_bins, 2),
            frame_count,
            self.dpdfnet.df_op.spec_buffer.time_steps,
        )

    def forward(self, noisy_frames: Tensor) -> Tensor:
        spec = noisy_frames * self.wnorm
        feat_erb = get_pow(spec) @ self.dpdfnet.erb_fb
        if self.dpdfnet.erb_to_db:
            feat_erb = to_db(feat_erb)
        raw_feat_spec = spec[..., :self.nb_df, :]

        erb_mean = self.erb_ema(feat_erb, self.dpdfnet.erb_norm.mu0)
        feat_erb = (feat_erb - erb_mean) / 40.0
        spec_power = self.spec_ema(get_mag(raw_feat_spec), self.dpdfnet.spec_norm.s0)
        spec_denominator = (spec_power + self.dpdfnet.spec_norm.eps).sqrt()
        feat_spec = torch.stack(
            (
                raw_feat_spec[..., 0] / spec_denominator,
                raw_feat_spec[..., 1] / spec_denominator,
            ),
            dim=-1,
        )

        spec = spec.unsqueeze(1)
        feat_erb = feat_erb.unsqueeze(1)
        feat_spec = feat_spec.permute(0, 3, 1, 2)
        encoder = self.dpdfnet.enc
        buffered_erb = self.erb_conv_prefix(feat_erb)
        e0 = encoder.erb_conv0(buffered_erb)
        e1 = encoder.erb_conv1(e0)
        e2 = encoder.erb_conv2(e1)
        e3 = encoder.erb_conv3(e2)
        erb_dprnn = self.erb_dprnn(e3)
        buffered_spec = self.df_conv_prefix(feat_spec)
        c0 = encoder.df_conv0(buffered_spec)
        c1 = encoder.df_conv1(c0)
        df_dprnn = self.df_dprnn(c1)
        cemb = encoder.df_fc_emb(df_dprnn.permute(0, 2, 3, 1).flatten(2))
        emb = encoder.combine(erb_dprnn.permute(0, 2, 3, 1).flatten(2), cemb)
        emb = self.encoder_emb_gru(emb)

        erb_decoder = self.dpdfnet.erb_dec
        decoded_emb = self.erb_decoder_gru(emb)
        decoded_emb = decoded_emb.reshape(1, self.frame_count, e3.shape[-1], -1).permute(0, 3, 1, 2)
        d3 = erb_decoder.convt3(erb_decoder.conv3p(e3) + decoded_emb)
        d2 = erb_decoder.convt2(erb_decoder.conv2p(e2) + d3)
        d1 = erb_decoder.convt1(erb_decoder.conv1p(e1) + d2)
        mask = erb_decoder.conv0_out(erb_decoder.conv0p(e0) + d1)

        df_decoder = self.dpdfnet.df_dec
        coefficients = self.df_decoder_gru(emb)
        if df_decoder.df_skip is not None:
            coefficients = coefficients + df_decoder.df_skip(emb)
        df_conv = df_decoder.df_convp(self.df_convp_prefix(c0)).permute(0, 2, 3, 1)
        coefficients = df_decoder.df_out(coefficients).reshape(
            1,
            self.frame_count,
            self.nb_df,
            df_decoder.df_out_ch,
        ) + df_conv
        coefficients = self.dpdfnet.df_out_transform(coefficients)

        if self.dpdfnet.mask_method != "before_df":
            raise ValueError("DPDFNet_SEQUENCE supports the published before_df masking mode only.")
        mask = mask.matmul(self.dpdfnet.erb_inv_fb).unsqueeze(4)
        masked_spec = self.mask_delay(spec) * mask
        delayed_coefficients = self.coefs_delay(coefficients).permute(0, 2, 1, 3, 4)
        spec_history = self.spec_history(masked_spec).squeeze(1)
        filtered = df_real(spec_history[..., :self.nb_df, :], delayed_coefficients)
        enhanced = torch.cat(
            (filtered, spec_history[:, :, self.dpdfnet.df_order // 2, self.nb_df:]),
            dim=2,
        )
        return enhanced * self.inv_wnorm


class DPDFNet_AUDIO(nn.Module):
    """Fixed-length waveform wrapper around DPDFNet's recurrent spectrum model."""

    def __init__(
        self,
        spectral_model: DPDFNet_CUSTOM,
        stft_model: STFT_Process,
        istft_model: STFT_Process,
        *,
        in_sample_rate: int,
        out_sample_rate: int,
        model_sample_rate: int,
        model_audio_length: int,
        output_audio_length: int,
        frame_count: int,
        input_is_integer: bool,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if min(in_sample_rate, out_sample_rate, model_sample_rate, model_audio_length, output_audio_length, frame_count) < 1:
            raise ValueError("DPDFNet audio export dimensions and sample rates must be positive.")
        self.spectral_model = spectral_model
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.freq_bins = spectral_model.dpdfnet.freq_bins
        self.model_audio_length = model_audio_length
        self.output_audio_length = output_audio_length
        self.frame_count = frame_count
        self.input_is_integer = input_is_integer
        self.output_is_integer = output_dtype == torch.int16
        self.output_is_float32 = output_dtype == torch.float32
        self.resample_before_centering = in_sample_rate > model_sample_rate
        self.resample_after_centering = in_sample_rate < model_sample_rate
        self.output_resample_before_pcm = out_sample_rate < model_sample_rate
        self.output_resample_after_pcm = out_sample_rate > model_sample_rate
        self.aligned_istft_length = model_audio_length + WINDOW_LENGTH - DPDFNET_OUTPUT_DELAY
        self.output_alignment_padding = max(0, model_audio_length - self.aligned_istft_length)
        initial_stream_state = spectral_model.dpdfnet.initial_stream_state(dtype=torch.float32)
        self.register_buffer("initial_erb_norm_state", initial_stream_state[0])
        self.register_buffer("initial_spec_norm_state", initial_stream_state[1])
        self.register_buffer("initial_encoder_erb_buffer_state", initial_stream_state[2][0])
        self.register_buffer("initial_encoder_erb_dprnn_state", initial_stream_state[2][1])
        self.register_buffer("initial_encoder_spec_buffer_state", initial_stream_state[2][2])
        self.register_buffer("initial_encoder_df_dprnn_state", initial_stream_state[2][3])
        self.register_buffer("initial_encoder_emb_state", initial_stream_state[2][4])
        self.register_buffer("initial_erb_decoder_state", initial_stream_state[3])
        self.register_buffer("initial_df_decoder_gru_state", initial_stream_state[4][0])
        self.register_buffer("initial_df_convp_state", initial_stream_state[4][1])
        self.register_buffer("initial_mask_state", initial_stream_state[5])
        self.register_buffer("initial_df_state", initial_stream_state[6])
        self.register_buffer("analysis_padding", torch.zeros((1, 1, WINDOW_LENGTH), dtype=torch.float32))

    @staticmethod
    def _resample(audio: Tensor, output_length: int) -> Tensor:
        return torch.nn.functional.interpolate(
            audio,
            size=output_length,
            mode="linear",
            align_corners=False,
        )

    def _initial_stream_state(self) -> StreamingState:
        return (
            self.initial_erb_norm_state,
            self.initial_spec_norm_state,
            (
                self.initial_encoder_erb_buffer_state,
                self.initial_encoder_erb_dprnn_state,
                self.initial_encoder_spec_buffer_state,
                self.initial_encoder_df_dprnn_state,
                self.initial_encoder_emb_state,
            ),
            self.initial_erb_decoder_state,
            (self.initial_df_decoder_gru_state, self.initial_df_convp_state),
            self.initial_mask_state,
            self.initial_df_state,
        )

    def analysis(self, audio: Tensor) -> Tensor:
        audio = audio.float()
        if self.resample_before_centering:
            audio = self._resample(audio, self.model_audio_length)
        if self.input_is_integer:
            audio = audio * INV_INT16
        if self.resample_after_centering:
            audio = self._resample(audio, self.model_audio_length)

        analysis_audio = torch.cat((audio, self.analysis_padding), dim=-1)
        packed_spectrum = self.stft_model._stft_B_packed_forward(analysis_audio)
        return packed_spectrum.reshape(1, 2, self.freq_bins, self.frame_count).permute(0, 3, 2, 1)

    def synthesis(self, denoised_frames: Tensor) -> Tensor:
        packed_denoised_spectrum = denoised_frames.permute(0, 3, 2, 1).reshape(
            1,
            self.freq_bins * 2,
            self.frame_count,
        )
        audio = self.istft_model._istft_B_packed_forward(packed_denoised_spectrum)
        audio = audio[..., DPDFNET_OUTPUT_DELAY :]
        if self.output_alignment_padding:
            audio = torch.nn.functional.pad(audio, (0, self.output_alignment_padding))
        elif self.aligned_istft_length > self.model_audio_length:
            audio = audio[..., : self.model_audio_length]

        if self.output_resample_before_pcm:
            audio = self._resample(audio, self.output_audio_length)
        if self.output_is_integer:
            audio = audio * 32767.0
        if self.output_resample_after_pcm:
            audio = self._resample(audio, self.output_audio_length)
        if self.output_is_integer:
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        if self.output_is_float32:
            return audio
        return audio.to(torch.float16)

    def forward(self, audio: Tensor) -> Tensor:
        noisy_frames = self.analysis(audio)
        state = self._initial_stream_state()
        denoised_frames: List[Tensor] = []
        for frame_index in range(self.frame_count):
            noisy_frame = noisy_frames[:, frame_index : frame_index + 1]
            denoised_frame, state = self.spectral_model.forward_stream(noisy_frame, state)
            denoised_frames.append(denoised_frame)
        return self.synthesis(torch.cat(denoised_frames, dim=1))


class DPDFNet_AUDIO_SEQUENCE(nn.Module):
    """Static waveform wrapper whose recurrent core is evaluated as one frame sequence."""

    def __init__(self, audio_model: DPDFNet_AUDIO):
        super().__init__()
        self.audio_model = audio_model
        self.spectral_sequence = DPDFNet_SEQUENCE(
            audio_model.spectral_model,
            audio_model.frame_count,
        )

    def forward(self, audio: Tensor) -> Tensor:
        noisy_frames = self.audio_model.analysis(audio)
        return self.audio_model.synthesis(self.spectral_sequence(noisy_frames))


class DPDFNet_AUDIO_ANALYSIS(nn.Module):
    """Export the waveform-to-static-frame boundary without tracing the recurrent core."""

    def __init__(self, audio_model: DPDFNet_AUDIO):
        super().__init__()
        self.audio_model = audio_model

    def forward(self, audio: Tensor) -> Tensor:
        return self.audio_model.analysis(audio)


class DPDFNet_AUDIO_SYNTHESIS(nn.Module):
    """Export the static-frame-to-waveform boundary without tracing the recurrent core."""

    def __init__(self, audio_model: DPDFNet_AUDIO):
        super().__init__()
        self.audio_model = audio_model

    def forward(self, denoised_frames: Tensor) -> Tensor:
        return self.audio_model.synthesis(denoised_frames)


def build_audio_transforms(model: DPDFNet) -> tuple[STFT_Process, STFT_Process]:
    """Build static Conv STFT/ISTFT models matching the checkpoint's Vorbis framing."""

    if (model.n_fft, model.window_length, model.hop_length) != (NFFT, WINDOW_LENGTH, HOP_LENGTH):
        raise ValueError(
            "STFT_Process constants must match the embedded DPDFNet framing: "
            f"model={(model.n_fft, model.window_length, model.hop_length)}, "
            f"export={(NFFT, WINDOW_LENGTH, HOP_LENGTH)}."
        )
    stft_model = STFT_Process(
        model_type="stft_B",
        n_fft=model.n_fft,
        win_length=model.window_length,
        hop_len=model.hop_length,
        max_frames=STATIC_STFT_FRAMES,
        window_type="vorbis",
        center_pad=True,
        pad_mode="reflect",
    ).eval()
    istft_model = STFT_Process(
        model_type="istft_B",
        n_fft=model.n_fft,
        win_length=model.window_length,
        hop_len=model.hop_length,
        max_frames=STATIC_STFT_FRAMES,
        window_type="vorbis",
        center_pad=True,
        pad_mode="reflect",
        static_norm=True,
    ).eval()
    if istft_model._out_end - istft_model._out_start != STFT_INPUT_LENGTH:
        raise RuntimeError("Static ISTFT output length does not match the configured STFT input length.")
    return stft_model, istft_model


def fuse_grouped_convolutions_for_export_(module: nn.Module) -> int:
    """Replace independent group branches with native grouped Conv2d kernels."""

    fused_count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, GroupedConv2D):
            if child.groups == 1:
                setattr(module, name, child.convs[0])
                fused_count += 1
                continue
            reference = child.convs[0]
            fused = nn.Conv2d(
                child.in_ch,
                child.out_ch,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias,
                device=reference.weight.device,
                dtype=reference.weight.dtype,
            )
            with torch.no_grad():
                fused.weight.copy_(torch.cat(tuple(convolution.weight for convolution in child.convs), dim=0))
                if fused.bias is not None:
                    fused.bias.copy_(torch.cat(tuple(convolution.bias for convolution in child.convs), dim=0))
            setattr(module, name, fused.eval())
            fused_count += 1
        else:
            fused_count += fuse_grouped_convolutions_for_export_(child)
    return fused_count


def fuse_subpixel_convolutions_for_export_(module: nn.Module) -> int:
    """Pack parallel sub-pixel branches into one grouped convolution for export."""

    fused_count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SubPixelConv2D):
            setattr(module, name, FusedSubPixelConv2D(child).eval())
            fused_count += 1
        else:
            fused_count += fuse_subpixel_convolutions_for_export_(child)
    return fused_count


def _fuse_conv_batch_norm_(convolution: nn.Module, batch_norm: nn.BatchNorm2d) -> None:
    if not isinstance(convolution, (nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError(f"Cannot fold BatchNorm2d into {type(convolution).__name__}.")
    if batch_norm.running_mean is None or batch_norm.running_var is None:
        raise ValueError("BatchNorm2d needs running statistics for export fusion.")
    with torch.no_grad():
        scale = (batch_norm.weight if batch_norm.weight is not None else torch.ones_like(batch_norm.running_mean)) / torch.sqrt(batch_norm.running_var + batch_norm.eps)
        batch_bias = batch_norm.bias if batch_norm.bias is not None else torch.zeros_like(batch_norm.running_mean)
        convolution_bias = convolution.bias if convolution.bias is not None else torch.zeros_like(batch_norm.running_mean)
        if isinstance(convolution, nn.Conv2d):
            weight = convolution.weight * scale.reshape(-1, 1, 1, 1)
        else:
            groups = convolution.groups
            output_per_group = convolution.out_channels // groups
            input_per_group = convolution.in_channels // groups
            weight = convolution.weight.reshape(groups, input_per_group, output_per_group, *convolution.weight.shape[2:])
            weight = (weight * scale.reshape(groups, 1, output_per_group, 1, 1)).reshape_as(convolution.weight)
        convolution.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        convolution.bias = nn.Parameter(((convolution_bias - batch_norm.running_mean) * scale + batch_bias).contiguous(), requires_grad=False)


def fuse_batch_norms_for_export_(model: nn.Module) -> int:
    expected = sum(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    fused_count = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        for index, child in enumerate(module):
            if not isinstance(child, nn.BatchNorm2d):
                continue
            if index == 0:
                raise ValueError("BatchNorm2d cannot be folded without a preceding convolution.")
            _fuse_conv_batch_norm_(module[index - 1], child)
            module[index] = nn.Identity()
            fused_count += 1
    if fused_count != expected:
        raise RuntimeError(f"Folded {fused_count} of {expected} DPDFNet BatchNorm2d layers.")
    return fused_count


def correct_state_dict(state_dict: dict) -> dict:
    """Map the training checkpoint's recurrent keys to the exporter stateful modules."""

    streaming_state = {}
    for key, value in state_dict.items():
        if "inter_gru" in key:
            adjusted_key = key.replace("_l0", "").replace("inter_gru.", "inter_gru.grucell.")
        elif "gru.gru" in key:
            layer = key[-1]
            adjusted_key = key[:-3].replace(".gru.", f".gru.{layer}.grucell.")
        else:
            adjusted_key = key
        streaming_state[adjusted_key] = value
    return streaming_state


def load_checkpoint_state(path: str) -> dict:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DPDFNet checkpoint was not found: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"DPDFNet checkpoint must be a state dictionary, got {type(payload).__name__}.")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("DPDFNet checkpoint state_dict is not a dictionary.")
    return state_dict


def build_audio_metadata() -> dict:
    """Describe the single waveform graph with the shared audio metadata schema."""

    return build_audio_metadata_from_globals(
        globals(),
        producer=Path(__file__).name,
        model_name=MODEL_NAME,
        task="denoise",
        model_family="dpdfnet",
        max_dynamic_audio_seconds=30,
        normalize_audio_default=False,
        input_channels=1,
        output_channels=1,
        num_audio_inputs=1,
        feature_kind="stft",
        center_pad=True,
        pad_mode="reflect",
    )


def validation_audio() -> Tensor:
    """Produce representative static input without changing the public audio dtype contract."""

    torch.manual_seed(0)
    shape = (1, 1, INPUT_AUDIO_LENGTH)
    if IN_TORCH_DTYPE == torch.int16:
        return torch.randint(-32768, 32768, shape, dtype=torch.int16)
    return torch.randn(shape, dtype=torch.float32).to(IN_TORCH_DTYPE)


_COPY_OPERATORS: Final[frozenset[str]] = frozenset(
    {"Concat", "Pad", "Tile", "Expand", "Transpose", "ScatterND", "Reshape"}
)
_SHAPE_OPERATORS: Final[frozenset[str]] = frozenset(
    {"Shape", "Gather", "GatherElements", "GatherND", "Slice", "Split", "Unsqueeze", "Squeeze", "Reshape"}
)
_INDEXING_OPERATORS: Final[frozenset[str]] = frozenset(
    {"Gather", "GatherElements", "GatherND", "Slice", "Split", "ScatterND"}
)
_ELEMENTWISE_OPERATORS: Final[frozenset[str]] = frozenset(
    {"Add", "Sub", "Mul", "Div", "Pow", "Neg", "Where", "Clip", "Max", "Min"}
)
_TENSOR_ELEMENT_BYTES: Final[dict[int, int]] = {
    onnx.TensorProto.BOOL: 1,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.UINT16: 2,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.BFLOAT16: 2,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.UINT32: 4,
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.UINT64: 8,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.COMPLEX64: 8,
    onnx.TensorProto.COMPLEX128: 16,
}


def _value_shape_signature(value: onnx.ValueInfoProto) -> tuple[str, int, tuple[Union[int, str, None], ...]]:
    tensor_type = value.type.tensor_type
    dimensions: List[Union[int, str, None]] = []
    if tensor_type.HasField("shape"):
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param"):
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(None)
    return value.name, int(tensor_type.elem_type), tuple(dimensions)


def _model_interface_signature(model: onnx.ModelProto) -> tuple[tuple, tuple]:
    return (
        tuple(_value_shape_signature(value) for value in model.graph.input),
        tuple(_value_shape_signature(value) for value in model.graph.output),
    )


def repair_static_audio_output_shape_(model: onnx.ModelProto) -> bool:
    """Repair only the legacy exporter's symbolic final-audio dimensions."""

    expected_shape = (1, 1, OUTPUT_AUDIO_LENGTH)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError("Static audio shape repair requires exactly one graph input and output.")
    output = model.graph.output[0]
    tensor_type = output.type.tensor_type
    if output.name != "denoised_audio" or tensor_type.elem_type != _onnx_tensor_type(OUT_TORCH_DTYPE):
        raise RuntimeError("Static audio shape repair found an unexpected output name or dtype.")
    if not tensor_type.HasField("shape") or len(tensor_type.shape.dim) != len(expected_shape):
        raise RuntimeError("Static audio shape repair requires a rank-3 output tensor.")
    dimensions = tensor_type.shape.dim
    changed = False
    for dimension, expected in zip(dimensions, expected_shape):
        if dimension.HasField("dim_value"):
            if int(dimension.dim_value) != expected:
                raise RuntimeError(
                    f"Static audio shape repair refused conflicting output dimension "
                    f"{dimension.dim_value}; expected {expected}."
                )
        elif dimension.HasField("dim_param"):
            if not dimension.dim_param:
                raise RuntimeError("Static audio shape repair refused an empty symbolic dimension.")
            changed = True
        else:
            raise RuntimeError("Static audio shape repair refused an unspecified output dimension.")
    if changed:
        del dimensions[:]
        for expected in expected_shape:
            dimensions.add().dim_value = expected
    return changed


def _known_value_nbytes(value: Optional[onnx.ValueInfoProto]) -> int:
    if value is None:
        return 0
    _name, element_type, dimensions = _value_shape_signature(value)
    if element_type not in _TENSOR_ELEMENT_BYTES or not dimensions:
        return 0
    if any(not isinstance(dimension, int) or dimension < 1 for dimension in dimensions):
        return 0
    return math.prod(dimensions) * _TENSOR_ELEMENT_BYTES[element_type]


def _initializer_nbytes(initializer: onnx.TensorProto) -> int:
    if initializer.raw_data:
        return len(initializer.raw_data)
    element_bytes = _TENSOR_ELEMENT_BYTES.get(initializer.data_type, 0)
    return math.prod(initializer.dims) * element_bytes if element_bytes else 0


def _iter_graph_nodes(graph: onnx.GraphProto):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from _iter_graph_nodes(attribute.g)
            for nested_graph in attribute.graphs:
                yield from _iter_graph_nodes(nested_graph)


def graph_metrics(model: onnx.ModelProto) -> dict:
    """Collect graph-only export metrics without creating an on-disk diagnostic artifact."""

    try:
        shaped_model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
        shape_inference = "complete"
    except (onnx.shape_inference.InferenceError, ValueError) as error:
        shaped_model = model
        shape_inference = f"partial ({type(error).__name__})"
    value_infos = {
        value.name: value
        for value in (*shaped_model.graph.input, *shaped_model.graph.output, *shaped_model.graph.value_info)
    }
    known_nbytes = {name: _known_value_nbytes(value) for name, value in value_infos.items()}
    known_nbytes.update(
        {initializer.name: _initializer_nbytes(initializer) for initializer in shaped_model.graph.initializer}
    )
    remaining_consumers = Counter(
        value_name
        for node in shaped_model.graph.node
        for value_name in node.input
        if value_name
    )
    remaining_consumers.update(output.name for output in shaped_model.graph.output)
    live_values = {value.name for value in shaped_model.graph.input}
    live_bytes = sum(known_nbytes[value.name] for value in shaped_model.graph.input)
    peak_live_bytes = live_bytes
    static_traffic_bytes = 0
    for node in shaped_model.graph.node:
        static_traffic_bytes += sum(known_nbytes.get(input_name, 0) for input_name in node.input if input_name)
        for output_name in node.output:
            output_bytes = known_nbytes.get(output_name, 0)
            live_bytes += output_bytes
            static_traffic_bytes += output_bytes
            live_values.add(output_name)
        peak_live_bytes = max(peak_live_bytes, live_bytes)
        for input_name in node.input:
            if input_name in live_values:
                remaining_consumers[input_name] -= 1
                if remaining_consumers[input_name] == 0:
                    live_bytes -= known_nbytes.get(input_name, 0)
                    live_values.remove(input_name)
        all_nodes = tuple(_iter_graph_nodes(shaped_model.graph))
        histogram = Counter(node.op_type for node in all_nodes)
    custom_nodes = sorted(
        f"{node.domain or 'ai.onnx'}::{node.op_type}"
            for node in all_nodes
        if node.domain not in {"", "ai.onnx"} or node.op_type.startswith("ATen")
    )
    return {
            "node_count": len(all_nodes),
            "top_level_node_count": len(shaped_model.graph.node),
            "nested_node_count": len(all_nodes) - len(shaped_model.graph.node),
        "initializer_count": len(shaped_model.graph.initializer),
        "initializer_bytes": sum(_initializer_nbytes(initializer) for initializer in shaped_model.graph.initializer),
        "cast_count": histogram["Cast"],
        "shape_op_count": sum(histogram[operator] for operator in _SHAPE_OPERATORS),
        "layout_op_count": histogram["Transpose"] + histogram["Reshape"],
        "indexing_op_count": sum(histogram[operator] for operator in _INDEXING_OPERATORS),
        "elementwise_op_count": sum(histogram[operator] for operator in _ELEMENTWISE_OPERATORS),
        "copy_op_count": sum(histogram[operator] for operator in _COPY_OPERATORS),
        "estimated_static_traffic_bytes": static_traffic_bytes,
        "estimated_peak_live_activation_bytes": peak_live_bytes,
        "histogram": histogram,
        "interface": _model_interface_signature(model),
        "opsets": tuple((entry.domain or "ai.onnx", entry.version) for entry in model.opset_import),
        "custom_nodes": tuple(custom_nodes),
        "shape_inference": shape_inference,
    }


def print_graph_metrics(label: str, metrics: dict) -> None:
    histogram = ", ".join(f"{operator}:{count}" for operator, count in sorted(metrics["histogram"].items()))
    print(
        f"{label} graph metrics: nodes={metrics['node_count']} "
        f"(top-level={metrics['top_level_node_count']}, nested={metrics['nested_node_count']}); "
        f"initializers={metrics['initializer_count']} ({metrics['initializer_bytes']} bytes); "
        f"Cast={metrics['cast_count']}; shape={metrics['shape_op_count']}; "
        f"layout={metrics['layout_op_count']}; indexing={metrics['indexing_op_count']}; "
        f"elementwise={metrics['elementwise_op_count']}; copy-family={metrics['copy_op_count']}; "
        f"top-level-estimated-static-traffic={metrics['estimated_static_traffic_bytes']} bytes; "
        f"top-level-estimated-peak-live-activations={metrics['estimated_peak_live_activation_bytes']} bytes; "
        f"shape-inference={metrics['shape_inference']}."
    )
    print(f"{label} operator histogram: {histogram}")
    if metrics["custom_nodes"]:
        raise RuntimeError(f"Unexpected custom or fallback nodes in {label} graph: {metrics['custom_nodes']}")


def _metadata_properties(model: onnx.ModelProto) -> dict[str, str]:
    return {property_.key: property_.value for property_ in model.metadata_props}


def _onnx_tensor_type(dtype: torch.dtype) -> int:
    tensor_types = {
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.float32: onnx.TensorProto.FLOAT,
        torch.int16: onnx.TensorProto.INT16,
    }
    try:
        return tensor_types[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported public ONNX audio dtype: {dtype}.") from error


def _copy_value_info(value: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    copied = onnx.ValueInfoProto()
    copied.CopyFrom(value)
    return copied


def _require_export_tensor(
    value: onnx.ValueInfoProto,
    *,
    label: str,
    expected_name: str,
    expected_type: int,
    expected_shape: Optional[tuple[int, ...]],
) -> None:
    name, element_type, dimensions = _value_shape_signature(value)
    if name != expected_name or element_type != expected_type:
        raise RuntimeError(
            f"Unexpected {label} tensor: name={name!r}, type={element_type}; "
            f"expected name={expected_name!r}, type={expected_type}."
        )
    if expected_shape is not None and dimensions != expected_shape:
        raise RuntimeError(
            f"Unexpected {label} tensor shape: {dimensions}; expected {expected_shape}."
        )


def _shared_opsets(*models: onnx.ModelProto) -> list[onnx.OperatorSetIdProto]:
    imports: dict[str, onnx.OperatorSetIdProto] = {}
    for model in models:
        for entry in model.opset_import:
            previous = imports.get(entry.domain)
            if previous is not None and previous.version != entry.version:
                raise RuntimeError(
                    f"Cannot compose ONNX models with different {entry.domain or 'ai.onnx'} opsets: "
                    f"{previous.version} and {entry.version}."
                )
            if previous is None:
                copied = onnx.OperatorSetIdProto()
                copied.CopyFrom(entry)
                imports[entry.domain] = copied
    return list(imports.values())


def compose_static_audio_loop_model(
    analysis_model: onnx.ModelProto,
    core_model: onnx.ModelProto,
    synthesis_model: onnx.ModelProto,
    *,
    initial_state: Tensor,
    frame_count: int,
    freq_bins: int,
) -> onnx.ModelProto:
    """Build one static waveform graph whose DPDFNet recurrence is an ONNX Loop."""

    if frame_count < 1 or freq_bins < 1:
        raise ValueError("DPDFNet Loop export requires positive frame and frequency-bin counts.")
    initial_state = initial_state.detach().cpu().contiguous()
    if initial_state.dtype != torch.float32 or initial_state.ndim != 1:
        raise ValueError("DPDFNet Loop state must be a one-dimensional float32 tensor.")
    state_size = int(initial_state.numel())
    expected_frame_shape = (1, 1, freq_bins, 2)
    expected_frames_shape = (1, frame_count, freq_bins, 2)

    if len(analysis_model.graph.input) != 1 or len(analysis_model.graph.output) != 1:
        raise RuntimeError("DPDFNet analysis export must have one input and one output.")
    if len(core_model.graph.input) != 2 or len(core_model.graph.output) != 2:
        raise RuntimeError("DPDFNet recurrent-frame export must have two inputs and two outputs.")
    if len(synthesis_model.graph.input) != 1 or len(synthesis_model.graph.output) != 1:
        raise RuntimeError("DPDFNet synthesis export must have one input and one output.")
    _require_export_tensor(
        analysis_model.graph.input[0],
        label="analysis input",
        expected_name="noisy_audio",
        expected_type=_onnx_tensor_type(IN_TORCH_DTYPE),
        expected_shape=(1, 1, INPUT_AUDIO_LENGTH),
    )
    _require_export_tensor(
        analysis_model.graph.output[0],
        label="analysis output",
        expected_name="noisy_frames",
        expected_type=TensorProto.FLOAT,
        expected_shape=expected_frames_shape,
    )
    _require_export_tensor(
        core_model.graph.input[0],
        label="recurrent-frame input",
        expected_name="noisy_frame",
        expected_type=TensorProto.FLOAT,
        expected_shape=expected_frame_shape,
    )
    _require_export_tensor(
        core_model.graph.input[1],
        label="recurrent-state input",
        expected_name="state_in",
        expected_type=TensorProto.FLOAT,
        expected_shape=(state_size,),
    )
    _require_export_tensor(
        core_model.graph.output[0],
        label="recurrent-frame output",
        expected_name="denoised_frame",
        expected_type=TensorProto.FLOAT,
        expected_shape=expected_frame_shape,
    )
    _require_export_tensor(
        core_model.graph.output[1],
        label="recurrent-state output",
        expected_name="state_out",
        expected_type=TensorProto.FLOAT,
        expected_shape=(state_size,),
    )
    _require_export_tensor(
        synthesis_model.graph.input[0],
        label="synthesis input",
        expected_name="denoised_frames",
        expected_type=TensorProto.FLOAT,
        expected_shape=expected_frames_shape,
    )
    _require_export_tensor(
        synthesis_model.graph.output[0],
        label="synthesis output",
        expected_name="denoised_audio",
        expected_type=_onnx_tensor_type(OUT_TORCH_DTYPE),
        expected_shape=None,
    )

    analysis = add_prefix(
        analysis_model,
        "analysis/",
        rename_inputs=False,
        rename_outputs=True,
        inplace=False,
    )
    core = add_prefix(core_model, "loop_core/", inplace=False)
    synthesis = add_prefix(
        synthesis_model,
        "synthesis/",
        rename_inputs=False,
        rename_outputs=False,
        inplace=False,
    )
    analysis_input = analysis.graph.input[0]
    analysis_output = analysis.graph.output[0]
    core_frame_input, core_state_input = core.graph.input
    core_frame_output, core_state_output = core.graph.output
    synthesis_input = synthesis.graph.input[0]
    synthesis_output = synthesis.graph.output[0]

    loop_index = "dpdfnet_loop_index"
    loop_condition_in = "dpdfnet_loop_condition_in"
    loop_condition_out = "dpdfnet_loop_condition_out"
    loop_frame = "dpdfnet_loop_frame"
    loop_scan_frame = "dpdfnet_loop_scan_frame"
    loop_axis = "dpdfnet_loop_frame_axis"
    loop_scan_frames = "dpdfnet_loop_scan_frames"
    loop_state_out = "dpdfnet_loop_state_out"
    body = helper.make_graph(
        [
            helper.make_node(
                "Gather",
                [analysis_output.name, loop_index],
                [loop_frame],
                axis=1,
                name="dpdfnet_loop/gather_frame",
            ),
            helper.make_node(
                "Unsqueeze",
                [loop_frame, loop_axis],
                [core_frame_input.name],
                name="dpdfnet_loop/restore_frame_axis",
            ),
            *core.graph.node,
            helper.make_node(
                "Identity",
                [loop_condition_in],
                [loop_condition_out],
                name="dpdfnet_loop/continue",
            ),
            helper.make_node(
                "Squeeze",
                [core_frame_output.name, loop_axis],
                [loop_scan_frame],
                name="dpdfnet_loop/remove_frame_axis",
            ),
        ],
        "dpdfnet_recurrent_loop_body",
        [
            helper.make_tensor_value_info(loop_index, TensorProto.INT64, []),
            helper.make_tensor_value_info(loop_condition_in, TensorProto.BOOL, []),
            _copy_value_info(core_state_input),
        ],
        [
            helper.make_tensor_value_info(loop_condition_out, TensorProto.BOOL, []),
            _copy_value_info(core_state_output),
            helper.make_tensor_value_info(loop_scan_frame, TensorProto.FLOAT, (1, freq_bins, 2)),
        ],
        initializer=[
            *core.graph.initializer,
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=loop_axis),
        ],
        value_info=core.graph.value_info,
    )
    loop_trip_count = "dpdfnet_loop_trip_count"
    loop_initial_condition = "dpdfnet_loop_initial_condition"
    loop_initial_state = "dpdfnet_loop_initial_state"
    graph = helper.make_graph(
        [
            *analysis.graph.node,
            helper.make_node(
                "Loop",
                [loop_trip_count, loop_initial_condition, loop_initial_state],
                [loop_state_out, loop_scan_frames],
                body=body,
                name="dpdfnet_recurrent_loop",
            ),
            helper.make_node(
                "Transpose",
                [loop_scan_frames],
                [synthesis_input.name],
                perm=[1, 0, 2, 3],
                name="dpdfnet_loop/restore_batch_time",
            ),
            *synthesis.graph.node,
        ],
        "dpdfnet_static_audio_loop",
        [_copy_value_info(analysis_input)],
        [_copy_value_info(synthesis_output)],
        initializer=[
            *analysis.graph.initializer,
            numpy_helper.from_array(np.asarray(frame_count, dtype=np.int64), name=loop_trip_count),
            numpy_helper.from_array(np.asarray(True, dtype=np.bool_), name=loop_initial_condition),
            numpy_helper.from_array(initial_state.numpy(), name=loop_initial_state),
            *synthesis.graph.initializer,
        ],
        value_info=[
            _copy_value_info(analysis_output),
            helper.make_tensor_value_info(loop_scan_frames, TensorProto.FLOAT, (frame_count, 1, freq_bins, 2)),
            _copy_value_info(synthesis_input),
        ],
    )
    composed = helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=_shared_opsets(analysis, core, synthesis),
    )
    composed.ir_version = max(analysis.ir_version, core.ir_version, synthesis.ir_version)
    onnx.checker.check_model(composed)
    return composed


def export_static_audio_loop_model(
    audio_model: DPDFNet_AUDIO,
    noisy_audio: Tensor,
    raw_model_path: Path,
) -> None:
    """Export static waveform boundaries and a single recurrent ONNX Loop body."""

    raw_model_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path = raw_model_path.with_name("DPDFNet.analysis.onnx")
    core_path = raw_model_path.with_name("DPDFNet.core.onnx")
    synthesis_path = raw_model_path.with_name("DPDFNet.synthesis.onnx")
    initial_state = audio_model.spectral_model.dpdfnet.initial_state(dtype=torch.float32)
    noisy_frame = torch.zeros((1, 1, audio_model.freq_bins, 2), dtype=torch.float32)
    denoised_frames = torch.zeros(
        (1, audio_model.frame_count, audio_model.freq_bins, 2),
        dtype=torch.float32,
    )
    export_options = {
        "do_constant_folding": True,
        "dynamic_axes": None,
        "opset_version": OPSET,
        "dynamo": False,
    }
    torch.onnx.export(
        DPDFNet_AUDIO_ANALYSIS(audio_model).eval(),
        (noisy_audio,),
        str(analysis_path),
        input_names=["noisy_audio"],
        output_names=["noisy_frames"],
        **export_options,
    )
    torch.onnx.export(
        audio_model.spectral_model,
        (noisy_frame, initial_state),
        str(core_path),
        input_names=["noisy_frame", "state_in"],
        output_names=["denoised_frame", "state_out"],
        **export_options,
    )
    torch.onnx.export(
        DPDFNet_AUDIO_SYNTHESIS(audio_model).eval(),
        (denoised_frames,),
        str(synthesis_path),
        input_names=["denoised_frames"],
        output_names=["denoised_audio"],
        **export_options,
    )
    composed = compose_static_audio_loop_model(
        onnx.load(str(analysis_path), load_external_data=False),
        onnx.load(str(core_path), load_external_data=False),
        onnx.load(str(synthesis_path), load_external_data=False),
        initial_state=initial_state,
        frame_count=audio_model.frame_count,
        freq_bins=audio_model.freq_bins,
    )
    onnx.save_model(composed, str(raw_model_path), save_as_external_data=False)


def export_static_audio_sequence_model(
    audio_model: DPDFNet_AUDIO,
    noisy_audio: Tensor,
    raw_model_path: Path,
) -> DPDFNet_AUDIO_SEQUENCE:
    """Export the static sequence-first waveform model without an ONNX Loop."""

    sequence_audio_model = DPDFNet_AUDIO_SEQUENCE(audio_model).eval()
    torch.onnx.export(
        sequence_audio_model,
        (noisy_audio,),
        str(raw_model_path),
        input_names=["noisy_audio"],
        output_names=["denoised_audio"],
        do_constant_folding=True,
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False,
    )
    return sequence_audio_model


def _validate_raw_output_interface(raw_outputs: tuple) -> None:
    expected_shape = (1, 1, OUTPUT_AUDIO_LENGTH)
    if len(raw_outputs) != 1:
        raise RuntimeError(f"Unexpected raw exported output contract: {raw_outputs}.")
    output_name, output_type, dimensions = raw_outputs[0]
    if output_name != "denoised_audio" or output_type != _onnx_tensor_type(OUT_TORCH_DTYPE):
        raise RuntimeError(f"Unexpected raw exported output contract: {raw_outputs}.")
    if len(dimensions) != len(expected_shape):
        raise RuntimeError(f"Raw exported output rank is not static-audio compatible: {raw_outputs}.")
    for dimension, expected in zip(dimensions, expected_shape):
        if isinstance(dimension, int) and dimension != expected:
            raise RuntimeError(f"Raw exported output dimension conflicts with the static contract: {raw_outputs}.")
        if dimension is None:
            raise RuntimeError(f"Raw exported output has an unspecified dimension: {raw_outputs}.")


def _create_in_memory_session(model: onnx.ModelProto) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _report_numerical_difference(label: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise RuntimeError(
            f"{label} changed output contract: reference={reference.shape}/{reference.dtype}, "
            f"candidate={candidate.shape}/{candidate.dtype}."
        )
    if np.issubdtype(reference.dtype, np.integer) or reference.dtype == np.bool_:
        if not np.array_equal(reference, candidate):
            mismatch = reference != candidate
            mismatch_count = int(np.count_nonzero(mismatch))
            indices = np.argwhere(mismatch)[:3].tolist()
            difference = np.abs(
                reference.astype(np.int64, copy=False) - candidate.astype(np.int64, copy=False)
            )
            max_absolute = int(difference.max(initial=0))
            if reference.dtype == np.bool_ or max_absolute > INTEGER_VALIDATION_ATOL:
                raise RuntimeError(
                    f"{label} changed {mismatch_count} integer or boolean output values; "
                    f"max-abs={max_absolute}; first-indices={indices}."
                )
            print(
                f"{label}: {mismatch_count} integer values differ; max-abs={max_absolute}; "
                f"within {INTEGER_VALIDATION_ATOL}-LSB tolerance; first-indices={indices}."
            )
            return
        print(f"{label}: exact integer/bool equality.")
        return
    if not np.array_equal(np.isnan(reference), np.isnan(candidate)) or not np.array_equal(
        np.isinf(reference), np.isinf(candidate)
    ):
        raise RuntimeError(f"{label} changed NaN or Inf behavior.")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    reference_finite = reference[finite].astype(np.float64, copy=False)
    candidate_finite = candidate[finite].astype(np.float64, copy=False)
    absolute = np.abs(reference_finite - candidate_finite)
    relative = absolute / np.maximum(np.abs(reference_finite), np.finfo(np.float64).tiny)
    np.testing.assert_allclose(
        candidate,
        reference,
        rtol=FLOAT_VALIDATION_RTOL,
        atol=FLOAT_VALIDATION_ATOL,
        equal_nan=True,
    )
    denominator = np.linalg.norm(reference_finite) * np.linalg.norm(candidate_finite)
    cosine = float(np.dot(reference_finite, candidate_finite) / denominator) if denominator else 1.0
    print(
        f"{label}: max-abs={absolute.max(initial=0.0):.6g}; "
        f"mean-abs={absolute.mean() if absolute.size else 0.0:.6g}; "
        f"max-rel={relative.max(initial=0.0):.6g}; cosine={cosine:.9f}."
    )


def validate_export_models(
    raw_model: onnx.ModelProto,
    final_model: onnx.ModelProto,
    audio_model: nn.Module,
    noisy_audio: Tensor,
    model_metadata: dict,
) -> None:
    """Validate the metadata-free audio graph and source/raw/final numerics."""

    onnx.checker.check_model(raw_model)
    onnx.checker.check_model(final_model)
    embedded_runtime_keys = set(build_model_metadata(model_metadata)).intersection(
        _metadata_properties(final_model)
    )
    if embedded_runtime_keys:
        raise RuntimeError(
            "DPDFNet runtime metadata must be stored only in the Metadata.onnx sidecar; "
            f"found embedded keys: {sorted(embedded_runtime_keys)}."
        )
    raw_interface = _model_interface_signature(raw_model)
    final_interface = _model_interface_signature(final_model)
    raw_inputs, raw_outputs = raw_interface
    final_inputs, final_outputs = final_interface
    expected_input = (("noisy_audio", _onnx_tensor_type(IN_TORCH_DTYPE), (1, 1, INPUT_AUDIO_LENGTH)),)
    expected_output = (("denoised_audio", _onnx_tensor_type(OUT_TORCH_DTYPE), (1, 1, OUTPUT_AUDIO_LENGTH)),)
    if raw_inputs != expected_input:
        raise RuntimeError(f"Unexpected exported input contract: {raw_inputs}.")
    _validate_raw_output_interface(raw_outputs)
    if final_inputs != expected_input or final_outputs != expected_output:
        raise RuntimeError(f"Unexpected final exported graph contract: inputs={final_inputs}, outputs={final_outputs}.")
    input_values = {"noisy_audio": noisy_audio.detach().cpu().numpy()}
    with torch.inference_mode():
        torch_output = audio_model(noisy_audio).detach().cpu().numpy()
    raw_session = _create_in_memory_session(raw_model)
    raw_output = raw_session.run(None, input_values)[0]
    del raw_session
    gc.collect()
    final_session = _create_in_memory_session(final_model)
    final_output = final_session.run(None, input_values)[0]
    del final_session
    gc.collect()
    _report_numerical_difference("PyTorch vs raw ONNX", torch_output, raw_output)
    _report_numerical_difference("Raw ONNX vs final ONNX", raw_output, final_output)


def _write_final_model_atomically(model: onnx.ModelProto, final_path: Path) -> None:
    """Write the audio graph without exposing or retaining a staging path."""

    if any(initializer.data_location == onnx.TensorProto.EXTERNAL for initializer in model.graph.initializer):
        raise RuntimeError("DPDFNet export unexpectedly requires external ONNX data; refusing multi-file output.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{final_path.stem}.",
            suffix=".onnx",
            dir=final_path.parent,
            delete=False,
        ) as staging_file:
            staging_path = Path(staging_file.name)
            staging_file.write(model.SerializeToString())
        onnx.checker.check_model(str(staging_path))
        os.replace(staging_path, final_path)
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)


def _write_metadata_model_atomically(metadata: dict, model_path: Path) -> Path:
    """Write the validated runtime-metadata sidecar without embedding it in the audio graph."""

    metadata_path = metadata_path_for_model(model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    expected_metadata = build_model_metadata(metadata)
    with tempfile.TemporaryDirectory(prefix=f".{metadata_path.stem}.", dir=metadata_path.parent) as staging_dir:
        staging_path = Path(staging_dir) / metadata_path.name
        export_metadata_carrier(staging_path, metadata, OPSET)
        sidecar_model = onnx.load(str(staging_path), load_external_data=False)
        onnx.checker.check_model(sidecar_model)
        actual_metadata = _metadata_properties(sidecar_model)
        if actual_metadata != expected_metadata:
            raise RuntimeError(
                "DPDFNet metadata sidecar does not contain the expected runtime metadata: "
                f"expected={sorted(expected_metadata)}, actual={sorted(actual_metadata)}."
            )
        os.replace(staging_path, metadata_path)
    return metadata_path


def _remove_legacy_export_artifacts(final_path: Path) -> None:
    """Remove obsolete raw exports and prior optimized artifacts after export."""

    for artifact in (
        final_path.with_name(f"{final_path.stem}.raw.onnx"),
        Path(f"{final_path}.data"),
    ):
        artifact.unlink(missing_ok=True)
    legacy_directory = parent_path / "DPDFNet_Optimized"
    for artifact in (
        legacy_directory / final_path.name,
        legacy_directory / f"{final_path.stem}_Metadata.onnx",
        legacy_directory / f"{final_path.name}.data",
    ):
        artifact.unlink(missing_ok=True)
    try:
        legacy_directory.rmdir()
    except OSError:
        pass


def main() -> None:
    print("Export start ...")
    final_model_path = Path(onnx_model_A).expanduser().resolve()
    export_dir = final_model_path.parent
    export_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dpdfnet_onnx_export_") as temp_dir:
        raw_model_path = Path(temp_dir) / "DPDFNet.raw.onnx"
        with torch.inference_mode():
            dpdfnet = DPDFNet(
                n_fft=NFFT,
                win_length=WINDOW_LENGTH / MODEL_SAMPLE_RATE,
                hop_length=HOP_LENGTH / MODEL_SAMPLE_RATE,
                samplerate=MODEL_SAMPLE_RATE,
                conv_kernel_inp=(3, 3),
                conv_ch=64,
                enc_gru_dim=256,
                erb_dec_gru_dim=256,
                df_dec_gru_dim=256,
                enc_lin_groups=32,
                lin_groups=16,
                upsample_conv_type="subpixel",
                group_linear_type="loop",
                point_wise_type="cnn",
                separable_first_conv=True,
                dprnn_num_blocks=DPRNN_NUM_BLOCKS,
            ).eval()
            dpdfnet.load_state_dict(correct_state_dict(load_checkpoint_state(checkpoint_path)), strict=True)
            grouped_linear_count, grouped_conv_count, subpixel_conv_count, batch_norm_count = dpdfnet.prepare_for_export_()
            custom_stft, custom_istft = build_audio_transforms(dpdfnet)
            spectral_model = DPDFNet_CUSTOM(dpdfnet.float()).eval()
            audio_model = DPDFNet_AUDIO(
                spectral_model,
                custom_stft,
                custom_istft,
                in_sample_rate=IN_SAMPLE_RATE,
                out_sample_rate=OUT_SAMPLE_RATE,
                model_sample_rate=MODEL_SAMPLE_RATE,
                model_audio_length=MODEL_AUDIO_LENGTH,
                output_audio_length=OUTPUT_AUDIO_LENGTH,
                frame_count=STATIC_STFT_FRAMES,
                input_is_integer=IN_TORCH_DTYPE == torch.int16,
                output_dtype=OUT_TORCH_DTYPE,
            ).eval()
            noisy_audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=IN_TORCH_DTYPE)
            if EXPORT_SEQUENCE_FIRST:
                audio_model = export_static_audio_sequence_model(
                    audio_model,
                    noisy_audio,
                    raw_model_path,
                )
            else:
                export_static_audio_loop_model(audio_model, noisy_audio, raw_model_path)
            print(
                "Prepared export graph: "
                f"{grouped_linear_count} grouped Linear modules lowered; "
                f"{grouped_conv_count} grouped Conv modules packed; "
                f"{subpixel_conv_count} sub-pixel Conv branch sets fused; "
                f"{batch_norm_count} Conv/BatchNorm pairs fused."
            )

            model_metadata = build_audio_metadata()
            raw_model = onnx.load(str(raw_model_path), load_external_data=False)
            final_model = onnx.ModelProto()
            final_model.CopyFrom(raw_model)
            repaired_static_output_shape = repair_static_audio_output_shape_(final_model)
            raw_metrics = graph_metrics(raw_model)
            final_metrics = graph_metrics(final_model)
            print_graph_metrics("Raw source-optimized", raw_metrics)
            print_graph_metrics("Final static-audio", final_metrics)
            if raw_metrics["histogram"] != final_metrics["histogram"]:
                raise RuntimeError("Final static-shape repair changed the computational graph.")
            print(
                "Legacy exporter static-output metadata repair: "
                f"{'applied' if repaired_static_output_shape else 'already static'}."
            )
            validate_export_models(raw_model, final_model, audio_model, validation_audio(), model_metadata)
            _write_final_model_atomically(final_model, final_model_path)
            metadata_model_path = _write_metadata_model_atomically(model_metadata, final_model_path)
            _remove_legacy_export_artifacts(final_model_path)
            del noisy_audio
            del final_model
            del raw_model
            del audio_model
            del spectral_model
            del custom_stft
            del custom_istft
            del dpdfnet
            gc.collect()

    print(f"\nExport done: {final_model_path}")
    print(f"Runtime metadata: {metadata_model_path}")
    print("Temporary raw export will be deleted; deploy the model and metadata sidecar together.")
    inference_script = Path(__file__).resolve().with_name("Inference_DPDFNet_ONNX.py")
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


if __name__ == "__main__":
    main()
    