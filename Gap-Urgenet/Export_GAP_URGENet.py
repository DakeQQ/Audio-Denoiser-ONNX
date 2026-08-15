#!/usr/bin/env python3
"""Export the complete static F32 GAP-URGENet pipeline to ONNX.

The implementation is intentionally self-contained.  It recreates the WavLM,
Vocos, Predictor, and PostNet execution paths in this file and imports only
the local packed-real/imaginary STFT helper.  Checkpoints are never downloaded:
all five files must be supplied locally before a model is constructed.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from STFT_Process import STFT_Process


parent_path = Path(__file__).resolve().parent
for _candidate in (parent_path, *parent_path.parents):
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from audio_onnx_metadata import export_metadata_carrier, metadata_path_for_model  # noqa: E402

# User settings.
# Download the required models manually from https://huggingface.co/Xiaobin-Rong/gap-urgenet.
# Automatic checkpoint downloads are intentionally disabled.
model_path          = Path.home() / "Downloads" / "gap-urgenet"                 # Directory containing the five local checkpoint files.
onnx_model_A        = parent_path / "GAP_URGENet_ONNX" / "GAP_URGENet.onnx"     # Final ONNX model; its .data sidecar is written beside it.
INPUT_AUDIO_LENGTH  = 16000 * 2                                                 # Fixed x-second, 48 kHz input window in samples.
IN_AUDIO_DTYPE      = "F32"                                                     # Required dtype of the ONNX noisy-audio input tensor.
OUT_AUDIO_DTYPE     = "F32"                                                     # Required dtype of the ONNX enhanced-audio output tensor.

# Fixed GAP-URGENet model and ONNX export parameters.
OPSET               = 20                                                        # ONNX opset required by the static export graph.
STATIC_BATCH        = 1                                                         # Fixed batch size supported by this exported graph.
DYNAMIC_AXES        = False                                                     # The GAP-URGENet export contract has no dynamic axes.
IN_SAMPLE_RATE      = 16000                                                     # Required noisy-audio sample rate in Hz.
OUT_SAMPLE_RATE     = 16000                                                     # Enhanced-audio sample rate in Hz.
MODEL_SAMPLE_RATE   = 48000                                                     # Internal generator and PostNet processing sample rate in Hz.

_EXPORT_TEMP_PREFIX = ".gap-urgenet-export-"
_EXPECTED_STATIC_RESAMPLER_PADS = ((7, 8), (7, 8), (19, 22))

# Derived audio contract.
OUTPUT_AUDIO_LENGTH = INPUT_AUDIO_LENGTH                                        # Static output length for the 16 kHz to 16 kHz pipeline.

# Required local checkpoint layout.
CHECKPOINT_FILENAMES = {
    "predictor": "Predictor.pt",
    "dewavlm": "DeWavLM-Omni.pt",
    "adapter": "Adapter.pt",
    "vocoder": "Vocoder.pt",
    "postnet": "PostNet.pt",
}


@dataclass(frozen=True)
class StaticShapePlan:
    """All sample and frame counts fixed by the baseline export contract."""

    input_samples: int
    wavlm_input_samples: int
    wavlm_frames: int
    predictor_nfft: int
    predictor_hop: int
    predictor_frames: int
    predictor_output_samples: int
    predictor_48k_samples: int
    postnet_nfft: int
    postnet_hop: int
    postnet_frames: int
    postnet_output_samples: int
    output_samples: int
    vocoder_nfft: int
    vocoder_hop: int
    vocoder_frames: int
    vocoder_output_samples: int
    vocoder_48k_samples: int


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _resampled_length(length: int, source_rate: int, target_rate: int) -> int:
    return _ceil_div(length * target_rate, source_rate)


def _conv_output_length(length: int, kernel: int, stride: int) -> int:
    return (length - kernel) // stride + 1


def _parse_conv_feature_layers(value: Any) -> list[tuple[int, int, int]]:
    """Parse WavLM's restricted convolution specification without eval()."""
    if isinstance(value, (list, tuple)):
        return [tuple(int(number) for number in item) for item in value]

    import re

    terms = re.findall(
        r"\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]"
        r"\s*(?:\*\s*(\d+))?",
        value,
    )
    layers: list[tuple[int, int, int]] = []
    for channels, kernel, stride, repetition in terms:
        layers.extend([(int(channels), int(kernel), int(stride))] * int(repetition or 1))
    return layers


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    return {str(key): item for key, item in vars(value).items()}


def checkpoint_paths(checkpoint_dir: Path) -> dict[str, Path]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    return {name: checkpoint_dir / filename for name, filename in CHECKPOINT_FILENAMES.items()}


def load_checkpoint_bundle(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Load local checkpoint containers without constructing any model."""
    bundles: dict[str, dict[str, Any]] = {}
    for component, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint = _mapping(payload, f"{component} checkpoint")
        state_kind = "model" if component == "dewavlm" or "model" in checkpoint else "generator"
        state = dict(checkpoint[state_kind])
        bundles[component] = {
            "path": str(path),
            "cfg": _mapping(checkpoint["cfg"], f"{component}.cfg"),
            "state_kind": state_kind,
            "state": state,
        }
    return bundles


def derive_static_shape_plan(wavlm_cfg: Mapping[str, Any], vocoder_cfg: Mapping[str, Any]) -> StaticShapePlan:
    """Derive every frame/sample count used by the static graph."""
    conv_layers = _parse_conv_feature_layers(wavlm_cfg.get("conv_feature_layers"))
    wavlm_input_samples = INPUT_AUDIO_LENGTH
    remainder = wavlm_input_samples % 320
    if remainder != 80:
        wavlm_input_samples += (wavlm_input_samples // 320) * 320 + 80 - wavlm_input_samples
    wavlm_frames = wavlm_input_samples
    for _, kernel, stride in conv_layers:
        wavlm_frames = _conv_output_length(wavlm_frames, kernel, stride)

    predictor_nfft = 512
    predictor_hop = 256
    predictor_frames = INPUT_AUDIO_LENGTH // predictor_hop + 1
    predictor_output_samples = INPUT_AUDIO_LENGTH
    predictor_48k_samples = _resampled_length(predictor_output_samples, 16000, 48000)

    postnet_nfft = 1536
    postnet_hop = 768
    postnet_frames = predictor_48k_samples // postnet_hop + 1
    postnet_output_samples = predictor_48k_samples

    vocoder_nfft = int(vocoder_cfg.get("n_fft", 1280))
    vocoder_hop = int(vocoder_cfg.get("hop_length", 320))
    vocoder_frames = wavlm_frames
    vocoder_output_samples = vocoder_frames * vocoder_hop
    vocoder_48k_samples = _resampled_length(vocoder_output_samples, 16000, 48000)

    plan = StaticShapePlan(
        input_samples=INPUT_AUDIO_LENGTH,
        wavlm_input_samples=wavlm_input_samples,
        wavlm_frames=wavlm_frames,
        predictor_nfft=predictor_nfft,
        predictor_hop=predictor_hop,
        predictor_frames=predictor_frames,
        predictor_output_samples=predictor_output_samples,
        predictor_48k_samples=predictor_48k_samples,
        postnet_nfft=postnet_nfft,
        postnet_hop=postnet_hop,
        postnet_frames=postnet_frames,
        postnet_output_samples=postnet_output_samples,
        output_samples=OUTPUT_AUDIO_LENGTH,
        vocoder_nfft=vocoder_nfft,
        vocoder_hop=vocoder_hop,
        vocoder_frames=vocoder_frames,
        vocoder_output_samples=vocoder_output_samples,
        vocoder_48k_samples=vocoder_48k_samples,
    )

    return plan


def _build_sinc_kernel(
    source_rate: int,
    target_rate: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> tuple[torch.Tensor, int, int, int]:
    """Precompute torchaudio's default sinc-interpolation Conv1d kernel."""
    gcd = math.gcd(source_rate, target_rate)
    source = source_rate // gcd
    target = target_rate // gcd
    base_frequency = min(source, target) * rolloff
    width = math.ceil(lowpass_filter_width * source / base_frequency)
    index = torch.arange(-width, width + source, dtype=torch.float32)[None, None] / source
    time = torch.arange(0, -target, -1, dtype=torch.float32)[:, None, None] / target + index
    time = (time * base_frequency).clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(time * math.pi / lowpass_filter_width / 2).square()
    phase = time * math.pi
    sinc = torch.where(phase == 0, torch.ones_like(phase), torch.sin(phase) / phase)
    kernel = sinc * window * (base_frequency / source)
    return kernel, width, source, target


class StaticSincResampler(nn.Module):
    """A fixed-length, F32-only copy of torchaudio's default resample path."""

    def __init__(self, source_rate: int, target_rate: int, input_length: int, batch_size: int = STATIC_BATCH):
        super().__init__()
        self.input_length = int(input_length)
        self.output_length = _resampled_length(input_length, source_rate, target_rate)
        self.batch_size = int(batch_size)
        if source_rate == target_rate:
            self.identity = True
            self.width = 0
            self.source_stride = 1
            self.target_rate = 1
            self.conv_output_length = input_length
            self.register_buffer("kernel", torch.empty(0), persistent=False)
            return

        kernel, width, source_stride, target = _build_sinc_kernel(source_rate, target_rate)
        padded_length = input_length + width + width + source_stride
        conv_output_length = (padded_length - kernel.shape[-1]) // source_stride + 1
        self.identity = False
        self.width = width
        self.source_stride = source_stride
        self.target_rate = target
        self.conv_output_length = conv_output_length
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return waveform
        padded = F.pad(waveform.unsqueeze(1), (self.width, self.width + self.source_stride))
        convolved = F.conv1d(padded, self.kernel, stride=self.source_stride)
        flattened = convolved.transpose(1, 2).reshape(
            self.batch_size,
            self.target_rate * self.conv_output_length,
        )
        return flattened[..., : self.output_length]


class StaticLengthAlign(nn.Module):
    """Apply GAP's right-pad/right-crop branch alignment with static constants."""

    def __init__(self, source_length: int, target_length: int, batch_size: int = STATIC_BATCH):
        super().__init__()
        self.crop_length = min(source_length, target_length)
        pad_length = max(0, target_length - source_length)
        self.identity = source_length == target_length
        self.has_right_pad = pad_length > 0
        self.register_buffer("right_pad", torch.zeros(batch_size, pad_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return waveform
        if self.has_right_pad:
            return torch.cat((waveform, self.right_pad), dim=-1)
        return waveform[..., : self.crop_length]


class StaticReflectSTFT(STFT_Process):
    """Lower fixed reflect padding to one standard ONNX Pad node."""

    def _stft_B_packed_forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self._center_pad or self._pad_mode != "reflect":
            return super()._stft_B_packed_forward(waveform)
        padded = F.pad(
            waveform,
            (self.half_n_fft, self.half_n_fft),
            mode="reflect",
        )
        return F.conv1d(padded, self.stft_kernel, stride=self.hop_len)


class TransposeLast(nn.Module):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.transpose(-2, -1)


class SamePad(nn.Module):
    def __init__(self, kernel_size: int, causal: bool = False):
        super().__init__()
        self.remove = kernel_size - 1 if causal else (1 if kernel_size % 2 == 0 else 0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor[:, :, : -self.remove] if self.remove else input_tensor


class GLU_Linear(nn.Module):
    """The small WavLM GLU helper, including its original parameter names."""

    def __init__(self, input_dim: int, output_dim: int, glu_type: str = "swish"):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim * 2, True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        projected = self.linear(input_tensor)
        value, gate = projected.split(self.output_dim, dim=-1)
        return value * (gate * torch.sigmoid(gate))


class WavLMConfig:
    """Local configuration carrier matching the serialized DeWavLM cfg schema."""

    def __init__(self, cfg: Mapping[str, Any]):
        defaults = {
            "extractor_mode": "default",
            "encoder_layers": 12,
            "encoder_embed_dim": 768,
            "encoder_ffn_embed_dim": 3072,
            "encoder_attention_heads": 12,
            "activation_fn": "gelu",
            "layer_norm_first": False,
            "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            "conv_bias": False,
            "feature_grad_mult": 1.0,
            "normalize": False,
            "encoder_layerdrop": 0.0,
            "mask_length": 10,
            "mask_prob": 0.65,
            "mask_selection": "static",
            "mask_other": 0.0,
            "no_mask_overlap": False,
            "mask_min_space": 1,
            "mask_channel_length": 10,
            "mask_channel_prob": 0.0,
            "mask_channel_selection": "static",
            "mask_channel_other": 0.0,
            "no_mask_channel_overlap": False,
            "mask_channel_min_space": 1,
            "conv_pos": 128,
            "conv_pos_groups": 16,
            "relative_position_embedding": False,
            "num_buckets": 320,
            "max_distance": 1280,
            "gru_rel_pos": False,
        }
        defaults.update(_mapping(cfg, "WavLM cfg"))
        self.__dict__.update(defaults)


class ConvFeatureExtractionModel(nn.Module):
    """The serialized WavLM convolutional frontend with no external imports."""

    def __init__(self, conv_layers: list[tuple[int, int, int]], mode: str, conv_bias: bool):
        super().__init__()
        in_channels = 1
        self.conv_layers = nn.ModuleList()
        for index, (channels, kernel, stride) in enumerate(conv_layers):
            convolution = nn.Conv1d(in_channels, channels, kernel, stride=stride, bias=conv_bias)
            if mode == "layer_norm":
                block = nn.Sequential(
                    convolution,
                    nn.Dropout(p=0.0),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(channels, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif index == 0:
                block = nn.Sequential(
                    convolution,
                    nn.Dropout(p=0.0),
                    nn.GroupNorm(channels, channels, affine=True),
                    nn.GELU(),
                )
            else:
                block = nn.Sequential(convolution, nn.Dropout(p=0.0), nn.GELU())
            self.conv_layers.append(block)
            in_channels = channels

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = waveform.unsqueeze(1)
        for layer in self.conv_layers:
            features = layer(features)
        return features


def _relative_position_bucket_indices(time_steps: int, num_buckets: int, max_distance: int) -> torch.Tensor:
    """Precompute WavLM's bidirectional relative-position bucket IDs."""
    result = torch.empty((time_steps, time_steps), dtype=torch.long)
    half_buckets = num_buckets // 2
    max_exact = half_buckets // 2
    for query_index in range(time_steps):
        for key_index in range(time_steps):
            relative = key_index - query_index
            direction = half_buckets if relative > 0 else 0
            distance = abs(relative)
            if distance < max_exact:
                bucket = distance
            else:
                scaled = max_exact + int(
                    math.log(distance / max_exact) / math.log(max_distance / max_exact) * (half_buckets - max_exact)
                )
                bucket = min(scaled, half_buckets - 1)
            result[query_index, key_index] = direction + bucket
    return result


class MultiheadAttention(nn.Module):
    """Static WavLM attention preserving the source parameter layout and math."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        has_relative_attention_bias: bool,
        num_buckets: int,
        max_distance: int,
        gru_rel_pos: bool,
        time_steps: int,
        batch_size: int = STATIC_BATCH,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.has_relative_attention_bias = bool(has_relative_attention_bias)
        self.gru_rel_pos = bool(gru_rel_pos)
        self.time_steps = int(time_steps)
        self.batch_size = int(batch_size)
        self.register_buffer("static_position_bias", None, persistent=True)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
            self.register_buffer(
                "relative_position_indices",
                _relative_position_bucket_indices(time_steps, num_buckets, max_distance),
                persistent=False,
            )
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def _base_position_bias(self) -> torch.Tensor | None:
        if self.static_position_bias is not None:
            return self.static_position_bias
        if not self.has_relative_attention_bias:
            return None
        bias = self.relative_attention_bias(self.relative_position_indices)
        return bias.permute(2, 0, 1).contiguous()

    def prepare_for_export(self) -> None:
        """Replace checkpoint-compatible Q/K/V modules with one immutable projection."""
        if self.has_relative_attention_bias and self.static_position_bias is None:
            with torch.no_grad():
                self.static_position_bias = self._base_position_bias()
            del self.relative_attention_bias
            del self.relative_position_indices
        with torch.no_grad():
            self.qkv_proj.weight.copy_(
                torch.cat(
                    (self.q_proj.weight * self.scaling, self.k_proj.weight, self.v_proj.weight),
                    dim=0,
                )
            )
            self.qkv_proj.bias.copy_(
                torch.cat(
                    (self.q_proj.bias * self.scaling, self.k_proj.bias, self.v_proj.bias),
                    dim=0,
                )
            )
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def forward(self, query: torch.Tensor, position_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if position_bias is None:
            position_bias = self._base_position_bias()
        if self.batch_size == 1:
            qkv_heads = self.qkv_proj(query).reshape(
                self.time_steps,
                3,
                self.num_heads,
                self.head_dim,
            ).permute(1, 2, 0, 3)
        else:
            qkv_heads = self.qkv_proj(query).reshape(
                self.time_steps,
                self.batch_size,
                3,
                self.num_heads,
                self.head_dim,
            ).permute(2, 1, 3, 0, 4).reshape(
                3,
                self.batch_size * self.num_heads,
                self.time_steps,
                self.head_dim,
            )
        query_heads, key_heads, value_heads = qkv_heads.unbind(0)
        attention_scores = torch.bmm(query_heads, key_heads.transpose(1, 2))
        if position_bias is not None:
            attention_bias = position_bias.reshape(
                self.batch_size * self.num_heads,
                self.time_steps,
                self.time_steps,
            )
            if self.gru_rel_pos:
                query_layer = query.transpose(0, 1).reshape(
                    self.batch_size,
                    self.time_steps,
                    self.num_heads,
                    self.head_dim,
                ).permute(0, 2, 1, 3)
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer).reshape(
                        self.batch_size,
                        self.num_heads,
                        self.time_steps,
                        2,
                        4,
                    ).sum(-1)
                ).chunk(2, dim=-1)
                gate = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attention_bias = gate.reshape(
                    self.batch_size * self.num_heads,
                    self.time_steps,
                    1,
                ) * attention_bias
            attention_scores = attention_scores + attention_bias
        attention_probabilities = F.softmax(attention_scores, dim=-1)
        attended = torch.bmm(attention_probabilities, value_heads)
        attended = attended.transpose(0, 1).contiguous().reshape(
            self.time_steps,
            self.batch_size,
            self.embed_dim,
        )
        return self.out_proj(attended), position_bias


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        activation_fn: str,
        layer_norm_first: bool,
        has_relative_attention_bias: bool,
        num_buckets: int,
        max_distance: int,
        gru_rel_pos: bool,
        time_steps: int,
    ):
        super().__init__()
        self.activation_name = activation_fn
        self.self_attn = MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            has_relative_attention_bias,
            num_buckets,
            max_distance,
            gru_rel_pos,
            time_steps,
        )
        self.layer_norm_first = bool(layer_norm_first)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = GLU_Linear(embedding_dim, ffn_embedding_dim, "swish") if activation_fn == "glu" else nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def _activate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "glu":
            return self.fc1(input_tensor)
        activated = self.fc1(input_tensor)
        return F.relu(activated) if self.activation_name == "relu" else F.gelu(activated)

    def forward(self, input_tensor: torch.Tensor, position_bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = input_tensor
        if self.layer_norm_first:
            normalized = self.self_attn_layer_norm(input_tensor)
            attended, position_bias = self.self_attn(normalized, position_bias)
            input_tensor = residual + attended
            residual = input_tensor
            normalized = self.final_layer_norm(input_tensor)
            projected = self._activate(normalized)
            input_tensor = residual + self.fc2(projected)
            return input_tensor, position_bias

        attended, position_bias = self.self_attn(input_tensor, position_bias)
        input_tensor = self.self_attn_layer_norm(residual + attended)
        residual = input_tensor
        projected = self._activate(input_tensor)
        input_tensor = self.final_layer_norm(residual + self.fc2(projected))
        return input_tensor, position_bias


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: WavLMConfig, time_steps: int):
        super().__init__()
        self.embedding_dim = cfg.encoder_embed_dim
        position_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=cfg.conv_pos,
            padding=cfg.conv_pos // 2,
            groups=cfg.conv_pos_groups,
        )
        self.pos_conv = nn.Sequential(nn.utils.weight_norm(position_conv, name="weight", dim=2), SamePad(cfg.conv_pos), nn.GELU())
        self.relative_position_embedding = bool(cfg.relative_position_embedding)
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=cfg.encoder_embed_dim,
                    ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
                    num_attention_heads=cfg.encoder_attention_heads,
                    activation_fn=cfg.activation_fn,
                    layer_norm_first=cfg.layer_norm_first,
                    has_relative_attention_bias=(self.relative_position_embedding and index == 0),
                    num_buckets=cfg.num_buckets,
                    max_distance=cfg.max_distance,
                    gru_rel_pos=cfg.gru_rel_pos,
                    time_steps=time_steps,
                )
                for index in range(cfg.encoder_layers)
            ]
        )
        self.layer_norm_first = bool(cfg.layer_norm_first)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def prepare_for_export(self) -> None:
        position_conv = self.pos_conv[0]
        if hasattr(position_conv, "weight_g"):
            nn.utils.remove_weight_norm(position_conv, name="weight")

    def forward_selected(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positional = self.pos_conv(features.transpose(1, 2)).transpose(1, 2)
        features = features + positional
        if not self.layer_norm_first:
            features = self.layer_norm(features)
        hidden = features.transpose(0, 1)
        layer_one = hidden
        layer_twenty_four = layer_one
        position_bias: torch.Tensor | None = None
        for index, layer in enumerate(self.layers):
            hidden, position_bias = layer(hidden, position_bias)
            if index == 0:
                layer_one = hidden
            if index == 23:
                layer_twenty_four = hidden
        return layer_one.transpose(0, 1), layer_twenty_four.transpose(0, 1)


class WavLM(nn.Module):
    """Static inference subset of DeWavLM retaining its checkpoint topology."""

    def __init__(self, cfg_mapping: Mapping[str, Any], time_steps: int):
        super().__init__()
        self.cfg = WavLMConfig(cfg_mapping)
        feature_layers = _parse_conv_feature_layers(self.cfg.conv_feature_layers)
        self.embed = feature_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            feature_layers,
            mode=self.cfg.extractor_mode,
            conv_bias=bool(self.cfg.conv_bias),
        )
        self.post_extract_proj = nn.Linear(self.embed, self.cfg.encoder_embed_dim) if self.embed != self.cfg.encoder_embed_dim else None
        self.mask_emb = nn.Parameter(torch.empty(self.cfg.encoder_embed_dim))
        self.encoder = TransformerEncoder(self.cfg, time_steps)
        self.layer_norm = nn.LayerNorm(self.embed)
        self.time_steps = int(time_steps)
        self.embedding_dim = int(self.cfg.encoder_embed_dim)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(waveform).transpose(1, 2)
        features = self.layer_norm(features)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        layer_one, layer_twenty_four = self.encoder.forward_selected(features)
        normalized_one = F.layer_norm(layer_one, (self.time_steps, self.embedding_dim), eps=1e-6)
        normalized_twenty_four = F.layer_norm(layer_twenty_four, (self.time_steps, self.embedding_dim), eps=1e-6)
        return normalized_one, normalized_twenty_four


class WavLMFeatureExtractor(nn.Module):
    """GAP's fixed 16 kHz, PLC-disabled WavLM entry point."""

    def __init__(self, cfg: Mapping[str, Any], shape_plan: StaticShapePlan):
        super().__init__()
        self.wavlm = WavLM(cfg, shape_plan.wavlm_frames)
        pad = shape_plan.wavlm_input_samples - shape_plan.input_samples
        self.has_right_pad = pad > 0
        self.register_buffer("right_pad", torch.zeros(STATIC_BATCH, pad), persistent=False)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded = torch.cat((waveform, self.right_pad), dim=-1) if self.has_right_pad else waveform
        return self.wavlm(padded)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float | None,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6) if self.adanorm else nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value and layer_scale_init_value > 0 else None

    def forward(self, input_tensor: torch.Tensor, cond_embedding_id: torch.Tensor | None = None) -> torch.Tensor:
        residual = input_tensor
        output = self.dwconv(input_tensor).transpose(1, 2)
        if self.adanorm:
            output = self.norm(output, cond_embedding_id)
        else:
            output = self.norm(output)
        output = self.pwconv2(self.act(self.pwconv1(output)))
        if self.gamma is not None:
            output = self.gamma * output
        return residual + output.transpose(1, 2)


class AdaLayerNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, input_tensor: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input_tensor, (self.dim,), eps=self.eps) * self.scale(cond_embedding_id) + self.shift(cond_embedding_id)


def _vocos_nonlinearity(input_tensor: torch.Tensor) -> torch.Tensor:
    return input_tensor * torch.sigmoid(input_tensor)


def _vocos_normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, conv_shortcut: bool = False, temb_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = _vocos_normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = _vocos_normalize(self.out_channels)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != self.out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        output = self.conv1(_vocos_nonlinearity(self.norm1(input_tensor)))
        if temb is not None:
            output = output + self.temb_proj(_vocos_nonlinearity(temb))[:, :, None]
        output = self.conv2(_vocos_nonlinearity(self.norm2(output)))
        if self.in_channels != self.out_channels:
            input_tensor = self.conv_shortcut(input_tensor) if self.use_conv_shortcut else self.nin_shortcut(input_tensor)
        return input_tensor + output


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.attention_scale = float(in_channels ** -0.5)
        self.attention_scale_folded = False
        self.norm = _vocos_normalize(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.qkv: nn.Conv1d | None = None
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def prepare_for_export(self) -> None:
        if self.qkv is not None:
            return
        projections = (self.q, self.k, self.v)
        fused = nn.Conv1d(self.in_channels, self.in_channels * 3, kernel_size=1, bias=True)
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat((self.q.weight * self.attention_scale, self.k.weight, self.v.weight), dim=0)
            )
            fused.bias.copy_(
                torch.cat((self.q.bias * self.attention_scale, self.k.bias, self.v.bias), dim=0)
            )
        self.qkv = fused
        self.attention_scale_folded = True
        del self.q
        del self.k
        del self.v

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(input_tensor)
        if self.qkv is None:
            query = self.q(normalized)
            key = self.k(normalized)
            value = self.v(normalized)
        else:
            query, key, value = self.qkv(normalized).split(
                (self.in_channels, self.in_channels, self.in_channels),
                dim=1,
            )
        query = query.permute(0, 2, 1)
        attention_scores = torch.bmm(query, key)
        if not self.attention_scale_folded:
            attention_scores = attention_scores * self.attention_scale
        weights = F.softmax(attention_scores, dim=2)
        attended = torch.bmm(value, weights.permute(0, 2, 1))
        return input_tensor + self.proj_out(attended)


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        num_res: int = 4,
        num_attn: int = 1,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6) if self.adanorm else nn.LayerNorm(dim, eps=1e-6)
        layer_scale = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(dim, intermediate_dim, layer_scale, adanorm_num_embeddings)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        pos_modules: list[nn.Module] = [
            *[ResnetBlock(dim, dim, temb_channels=0) for _ in range(num_res // 2)],
            *[AttnBlock(dim) for _ in range(num_attn)],
            *[ResnetBlock(dim, dim, temb_channels=0) for _ in range(num_res // 2)],
            _vocos_normalize(dim),
        ]
        self.pos_net = nn.Sequential(*pos_modules)

    def forward(self, input_tensor: torch.Tensor, bandwidth_id: torch.Tensor | None = None) -> torch.Tensor:
        output = self.pos_net(self.embed(input_tensor))
        if self.adanorm:
            output = self.norm(output.transpose(1, 2), bandwidth_id)
        else:
            output = self.norm(output.transpose(1, 2))
        output = output.transpose(1, 2)
        for block in self.convnext:
            output = block(output, bandwidth_id)
        return self.final_layer_norm(output.transpose(1, 2)).transpose(1, 2)


class VocosAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int = 1024,
        dim: int = 1024,
        intermediate_dim: int = 4096,
        num_layers: int = 12,
        output_channels: int = 1024,
    ):
        super().__init__()
        self.proj = nn.Linear(input_channels, input_channels)
        self.decoder = VocosBackbone(input_channels, dim, intermediate_dim, num_layers)
        self.head = nn.Linear(dim, output_channels)
        self.register_buffer("head_bias", self.head.bias.detach()[None, :, None], persistent=False)

    def forward(self, embed_a: torch.Tensor, embed_p: torch.Tensor) -> torch.Tensor:
        output = self.proj(embed_p) + embed_a
        output = self.decoder(output.transpose(1, 2))
        return torch.matmul(self.head.weight, output) + self.head_bias


class ISTFT(nn.Module):
    """Vocos same-padding ISTFT implemented through the packed local helper."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, frames: int):
        super().__init__()
        trim = (win_length - hop_length) // 2
        self.packed_istft = STFT_Process(
            "istft_B",
            n_fft=n_fft,
            win_length=win_length,
            hop_len=hop_length,
            max_frames=frames,
            window_type="hann",
            center_pad=False,
            static_norm=True,
            istft_trim_left=trim,
            istft_trim_right=trim,
            static_batch=STATIC_BATCH,
            persistent_buffers=False,
        )

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return self.packed_istft._istft_B_packed_forward(torch.cat((real, imag), dim=1)).squeeze(1)


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, frames: int):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("out_bias", self.out.bias.detach()[None, :, None], persistent=False)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, frames=frames)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        coefficients = torch.matmul(self.out.weight, input_tensor) + self.out_bias
        magnitude_log, phase = coefficients.chunk(2, dim=1)
        magnitude = torch.exp(torch.clamp(magnitude_log, min=-20, max=5))
        return self.istft(magnitude * torch.cos(phase), magnitude * torch.sin(phase))


class VocosVocoder(nn.Module):
    def __init__(
        self,
        frames: int,
        input_channels: int = 1024,
        dim: int = 768,
        intermediate_dim: int = 2304,
        num_layers: int = 12,
        num_res: int = 4,
        num_attn: int = 1,
        n_fft: int = 1280,
        hop_length: int = 320,
    ):
        super().__init__()
        self.decoder = VocosBackbone(
            input_channels,
            dim,
            intermediate_dim,
            num_layers,
            num_res=num_res,
            num_attn=num_attn,
        )
        self.head = ISTFTHead(dim, n_fft, hop_length, frames)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(input_tensor))


class LayerNormalization(nn.Module):
    """TFGridNet's per-axis affine normalization with source state names."""

    def __init__(self, input_dim: int, dim: int = 1, total_dim: int = 4, eps: float = 1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        parameter_shape = [1 if index != self.dim else input_dim for index in range(total_dim)]
        self.gamma = nn.Parameter(torch.ones(*parameter_shape, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*parameter_shape, dtype=torch.float32))
        self.eps = eps

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        mean = input_tensor.mean(dim=self.dim, keepdim=True)
        standard_deviation = torch.sqrt(input_tensor.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        return ((input_tensor - mean) / standard_deviation) * self.gamma + self.beta


class AllHeadPReLULayerNormalization4DC(nn.Module):
    """TFGridNet's source-compatible headwise PReLU normalization."""

    def __init__(
        self,
        input_dimension: tuple[int, int],
        static_time: int,
        static_frequency: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        heads, head_features = input_dimension
        self.gamma = nn.Parameter(torch.ones(1, heads, head_features, 1, 1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, heads, head_features, 1, 1, dtype=torch.float32))
        self.act = nn.PReLU(num_parameters=heads, init=0.25)
        self.eps = eps
        self.H = heads
        self.E = head_features
        self.static_time = int(static_time)
        self.static_frequency = int(static_frequency)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = input_tensor.reshape(
            STATIC_BATCH,
            self.H,
            self.E,
            self.static_time,
            self.static_frequency,
        )
        output = self.act(output)
        mean = output.mean(dim=2, keepdim=True)
        standard_deviation = torch.sqrt(output.var(dim=2, unbiased=False, keepdim=True) + self.eps)
        return ((output - mean) / standard_deviation) * self.gamma + self.beta


class StaticGridNetV3Block(nn.Module):
    """A fixed-shape TFGridNet block with static Gather patches and LSTM states."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        hidden_channels: int,
        n_head: int,
        qk_output_channel: int,
        static_time: int,
        static_frequency: int,
        activation: str = "prelu",
        eps: float = 1e-5,
    ):
        super().__init__()
        packed_channels = emb_dim * emb_ks
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(packed_channels, hidden_channels, 1, batch_first=True, bidirectional=True)
        self.intra_linear = (
            nn.Linear(hidden_channels * 2, packed_channels)
            if emb_ks == emb_hs
            else nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)
        )
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(packed_channels, hidden_channels, 1, batch_first=True, bidirectional=True)
        self.inter_linear = (
            nn.Linear(hidden_channels * 2, packed_channels)
            if emb_ks == emb_hs
            else nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)
        )
        self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * qk_output_channel, 1)
        self.attn_norm_Q = AllHeadPReLULayerNormalization4DC(
            (n_head, qk_output_channel),
            static_time,
            static_frequency,
            eps=eps,
        )
        self.attn_conv_K = nn.Conv2d(emb_dim, n_head * qk_output_channel, 1)
        self.attn_norm_K = AllHeadPReLULayerNormalization4DC(
            (n_head, qk_output_channel),
            static_time,
            static_frequency,
            eps=eps,
        )
        self.attn_conv_V = nn.Conv2d(emb_dim, emb_dim, 1)
        self.attn_norm_V = AllHeadPReLULayerNormalization4DC(
            (n_head, emb_dim // n_head),
            static_time,
            static_frequency,
            eps=eps,
        )
        self.attn_qkv: nn.Conv2d | None = None
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.PReLU(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )
        self.emb_dim = int(emb_dim)
        self.emb_ks = int(emb_ks)
        self.emb_hs = int(emb_hs)
        self.n_head = int(n_head)
        self.static_time = int(static_time)
        self.static_frequency = int(static_frequency)
        self._prepare_static_layout(hidden_channels)

    def _prepare_static_layout(self, hidden_channels: int) -> None:
        overlap = self.emb_ks - self.emb_hs
        padded_time = _ceil_div(self.static_time + 2 * overlap - self.emb_ks, self.emb_hs) * self.emb_hs + self.emb_ks
        padded_frequency = _ceil_div(self.static_frequency + 2 * overlap - self.emb_ks, self.emb_hs) * self.emb_hs + self.emb_ks
        intra_segments = (padded_frequency - self.emb_ks) // self.emb_hs + 1
        inter_segments = (padded_time - self.emb_ks) // self.emb_hs + 1
        intra_indices = (
            torch.arange(intra_segments, dtype=torch.int64).unsqueeze(1) * self.emb_hs
            + torch.arange(self.emb_ks, dtype=torch.int64).unsqueeze(0)
        ).reshape(-1)
        inter_indices = (
            torch.arange(inter_segments, dtype=torch.int64).unsqueeze(1) * self.emb_hs
            + torch.arange(self.emb_ks, dtype=torch.int64).unsqueeze(0)
        ).reshape(-1)
        self.register_buffer("intra_indices", intra_indices, persistent=False)
        self.register_buffer("inter_indices", inter_indices, persistent=False)
        self.register_buffer(
            "intra_h0",
            torch.zeros(2, STATIC_BATCH * padded_time, hidden_channels, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "intra_c0",
            torch.zeros(2, STATIC_BATCH * padded_time, hidden_channels, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "inter_h0",
            torch.zeros(2, STATIC_BATCH * padded_frequency, hidden_channels, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "inter_c0",
            torch.zeros(2, STATIC_BATCH * padded_frequency, hidden_channels, dtype=torch.float32),
            persistent=True,
        )
        self.overlap = int(overlap)
        self.padded_time = int(padded_time)
        self.padded_frequency = int(padded_frequency)
        self.intra_segments = int(intra_segments)
        self.inter_segments = int(inter_segments)
        self.attention_scale = float(1.0 / math.sqrt(self.attn_norm_Q.E * self.static_frequency))
        self.attention_scale_folded = False

    def prepare_for_export(self) -> None:
        if self.attn_qkv is not None:
            return
        projections = (self.attn_conv_Q, self.attn_conv_K, self.attn_conv_V)
        qk_channels = self.n_head * self.attn_norm_Q.E
        fused = nn.Conv2d(self.emb_dim, qk_channels * 2 + self.emb_dim, kernel_size=1, bias=True)
        with torch.no_grad():
            fused.weight.copy_(torch.cat(tuple(projection.weight for projection in projections), dim=0))
            fused.bias.copy_(torch.cat(tuple(projection.bias for projection in projections), dim=0))
            self.attn_norm_Q.gamma.mul_(self.attention_scale)
            self.attn_norm_Q.beta.mul_(self.attention_scale)
        self.attn_qkv = fused
        self.attention_scale_folded = True
        del self.attn_conv_Q
        del self.attn_conv_K
        del self.attn_conv_V

    def _intra_path(self, padded: torch.Tensor) -> torch.Tensor:
        normalized = self.intra_norm(padded)
        patches = normalized.reshape(
            STATIC_BATCH * self.padded_time,
            self.padded_frequency,
            self.emb_dim,
        ).index_select(1, self.intra_indices).reshape(
            STATIC_BATCH * self.padded_time,
            self.intra_segments,
            self.emb_ks,
            self.emb_dim,
        ).permute(0, 1, 3, 2).reshape(
            STATIC_BATCH * self.padded_time,
            self.intra_segments,
            self.emb_ks * self.emb_dim,
        )
        recurrent, _ = self.intra_rnn(patches, (self.intra_h0, self.intra_c0))
        if self.emb_ks == self.emb_hs:
            reconstructed = self.intra_linear(recurrent).reshape(
                STATIC_BATCH,
                self.padded_time,
                self.padded_frequency,
                self.emb_dim,
            )
        else:
            reconstructed = self.intra_linear(recurrent.transpose(1, 2)).reshape(
                STATIC_BATCH,
                self.padded_time,
                self.emb_dim,
                self.padded_frequency,
            ).transpose(-2, -1)
        return padded + reconstructed

    def _inter_path(self, intra: torch.Tensor) -> torch.Tensor:
        input_tensor = intra.transpose(1, 2)
        normalized = self.inter_norm(input_tensor)
        patches = normalized.reshape(
            STATIC_BATCH * self.padded_frequency,
            self.padded_time,
            self.emb_dim,
        ).index_select(1, self.inter_indices).reshape(
            STATIC_BATCH * self.padded_frequency,
            self.inter_segments,
            self.emb_ks,
            self.emb_dim,
        ).permute(0, 1, 3, 2).reshape(
            STATIC_BATCH * self.padded_frequency,
            self.inter_segments,
            self.emb_ks * self.emb_dim,
        )
        recurrent, _ = self.inter_rnn(patches, (self.inter_h0, self.inter_c0))
        if self.emb_ks == self.emb_hs:
            reconstructed = self.inter_linear(recurrent).reshape(
                STATIC_BATCH,
                self.padded_frequency,
                self.padded_time,
                self.emb_dim,
            )
        else:
            reconstructed = self.inter_linear(recurrent.transpose(1, 2)).reshape(
                STATIC_BATCH,
                self.padded_frequency,
                self.emb_dim,
                self.padded_time,
            ).transpose(-2, -1)
        return input_tensor + reconstructed

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        padded = F.pad(
            input_tensor.permute(0, 2, 3, 1),
            (
                0,
                0,
                self.overlap,
                self.padded_frequency - self.static_frequency - self.overlap,
                self.overlap,
                self.padded_time - self.static_time - self.overlap,
            ),
        )
        intra = self._intra_path(padded)
        inter = self._inter_path(intra).permute(0, 3, 2, 1)
        cropped = inter[
            ...,
            self.overlap : self.overlap + self.static_time,
            self.overlap : self.overlap + self.static_frequency,
        ]
        if self.attn_qkv is None:
            query_input = self.attn_conv_Q(cropped)
            key_input = self.attn_conv_K(cropped)
            value_input = self.attn_conv_V(cropped)
        else:
            qk_channels = self.n_head * self.attn_norm_Q.E
            query_input, key_input, value_input = self.attn_qkv(cropped).split(
                (qk_channels, qk_channels, self.emb_dim),
                dim=1,
            )
        query = self.attn_norm_Q(query_input).reshape(
            STATIC_BATCH * self.n_head,
            self.attn_norm_Q.E,
            self.static_time,
            self.static_frequency,
        )
        key = self.attn_norm_K(key_input).reshape(
            STATIC_BATCH * self.n_head,
            self.attn_norm_K.E,
            self.static_time,
            self.static_frequency,
        )
        value = self.attn_norm_V(value_input).reshape(
            STATIC_BATCH * self.n_head,
            self.attn_norm_V.E,
            self.static_time,
            self.static_frequency,
        )
        query = query.transpose(1, 2).reshape(
            STATIC_BATCH * self.n_head,
            self.static_time,
            self.attn_norm_Q.E * self.static_frequency,
        )
        key = key.transpose(2, 3).reshape(
            STATIC_BATCH * self.n_head,
            self.attn_norm_K.E * self.static_frequency,
            self.static_time,
        )
        value_shape = (
            STATIC_BATCH * self.n_head,
            self.static_time,
            self.attn_norm_V.E,
            self.static_frequency,
        )
        value = value.transpose(1, 2).reshape(
            STATIC_BATCH * self.n_head,
            self.static_time,
            self.attn_norm_V.E * self.static_frequency,
        )
        attention_scores = torch.matmul(query, key)
        if not self.attention_scale_folded:
            attention_scores = attention_scores * self.attention_scale
        attention = F.softmax(attention_scores, dim=2)
        value = torch.matmul(attention, value).reshape(value_shape).transpose(1, 2).reshape(
            STATIC_BATCH,
            self.emb_dim,
            self.static_time,
            self.static_frequency,
        )
        return self.attn_concat_proj(value) + cropped


class StaticBandSplit(nn.Module):
    """GAP PostNet's three fixed 257-bin bands, without a Python forward loop."""

    def __init__(self, n_bands: int = 3):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                input_tensor[..., 0:257],
                input_tensor[..., 256:513],
                input_tensor[..., 512:769],
            ),
            dim=1,
        )

    def inverse(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                input_tensor[:, 0:2, :, :],
                input_tensor[:, 2:4, :, 1:],
                input_tensor[:, 4:6, :, 1:],
            ),
            dim=-1,
        )


class PredictorTFGridNet(nn.Module):
    """Static packed-spectrum version of GAP's 16 kHz one-channel Predictor."""

    def __init__(
        self,
        shape_plan: StaticShapePlan,
        fft_len: float = 0.032,
        hop_len: float = 0.016,
        win_len: float = 0.032,
        n_srcs: int = 1,
        n_imics: int = 1,
        n_layers: int = 6,
        lstm_hidden_units: int = 200,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 2,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        activation: str = "prelu",
        eps: float = 1.0e-5,
    ):
        super().__init__()
        nfft = int(round(16000 * float(fft_len)))
        hop = int(round(16000 * float(hop_len)))
        window = int(round(16000 * float(win_len)))
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList(
            [
                StaticGridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    static_time=shape_plan.predictor_frames,
                    static_frequency=shape_plan.predictor_nfft // 2 + 1,
                    activation=activation,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, (3, 3), padding=(1, 1))
        self.stft_model = StaticReflectSTFT(
            "stft_B",
            n_fft=shape_plan.predictor_nfft,
            win_length=shape_plan.predictor_nfft,
            hop_len=shape_plan.predictor_hop,
            max_frames=shape_plan.predictor_frames,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
            persistent_buffers=False,
        )
        self.istft_model = STFT_Process(
            "istft_B",
            n_fft=shape_plan.predictor_nfft,
            win_length=shape_plan.predictor_nfft,
            hop_len=shape_plan.predictor_hop,
            max_frames=shape_plan.predictor_frames,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
            static_norm=True,
            static_batch=STATIC_BATCH,
            persistent_buffers=False,
        )
        self.frequency_bins = shape_plan.predictor_nfft // 2 + 1
        self.frames = shape_plan.predictor_frames

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.std(waveform, dim=1, keepdim=True) + 1e-12
        normalized = (waveform - waveform.mean(dim=1, keepdim=True)) / standard_deviation
        packed = self.stft_model._stft_B_packed_forward(normalized.unsqueeze(1))
        features = packed.reshape(STATIC_BATCH, 2, self.frequency_bins, self.frames).permute(0, 1, 3, 2)
        features = self.conv(features)
        for block in self.blocks:
            features = block(features)
        coefficients = self.deconv(features).permute(0, 1, 3, 2).reshape(
            STATIC_BATCH,
            self.frequency_bins * 2,
            self.frames,
        )
        output = self.istft_model._istft_B_packed_forward(coefficients).squeeze(1)
        return output * standard_deviation


class PostNetTFGridNet(nn.Module):
    """Static packed-spectrum GAP 48 kHz two-input-channel fusion PostNet."""

    def __init__(
        self,
        shape_plan: StaticShapePlan,
        n_srcs: int = 1,
        n_imics: int = 2,
        n_bands: int = 3,
        n_layers: int = 5,
        lstm_hidden_units: int = 100,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 2,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        activation: str = "prelu",
        eps: float = 1.0e-5,
    ):
        super().__init__()
        self.n_imics = n_imics
        self.bs = StaticBandSplit(n_bands)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics * n_bands, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList(
            [
                StaticGridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    static_time=shape_plan.postnet_frames,
                    static_frequency=257,
                    activation=activation,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2 * n_bands, (3, 3), padding=(1, 1))
        self.stft_model = StaticReflectSTFT(
            "stft_B",
            n_fft=shape_plan.postnet_nfft,
            win_length=shape_plan.postnet_nfft,
            hop_len=shape_plan.postnet_hop,
            max_frames=shape_plan.postnet_frames,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
            persistent_buffers=False,
        )
        self.istft_model = STFT_Process(
            "istft_B",
            n_fft=shape_plan.postnet_nfft,
            win_length=shape_plan.postnet_nfft,
            hop_len=shape_plan.postnet_hop,
            max_frames=shape_plan.postnet_frames,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
            static_norm=True,
            static_batch=STATIC_BATCH,
            persistent_buffers=False,
        )
        self.frequency_bins = shape_plan.postnet_nfft // 2 + 1
        self.frames = shape_plan.postnet_frames
        self.input_samples = shape_plan.postnet_output_samples

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.std(waveform, dim=(1, 2), keepdim=True) + 1e-12
        normalized = (waveform - waveform.mean(dim=(1, 2), keepdim=True)) / standard_deviation
        packed = self.stft_model._stft_B_packed_forward(
            normalized.reshape(STATIC_BATCH * self.n_imics, 1, self.input_samples)
        )
        spectrum = packed.reshape(
            STATIC_BATCH,
            self.n_imics,
            2,
            self.frequency_bins,
            self.frames,
        ).permute(0, 1, 2, 4, 3).reshape(
            STATIC_BATCH,
            self.n_imics * 2,
            self.frames,
            self.frequency_bins,
        )
        features = self.conv(self.bs(spectrum))
        for block in self.blocks:
            features = block(features)
        coefficients = self.bs.inverse(self.deconv(features))
        packed_output = coefficients.permute(0, 1, 3, 2).reshape(
            STATIC_BATCH,
            2 * self.frequency_bins,
            self.frames,
        )
        output = self.istft_model._istft_B_packed_forward(packed_output).squeeze(1)
        return output * standard_deviation.squeeze(1)


class GAPURGENetExport(nn.Module):
    """Complete static F32 GAP-URGENet: 16 kHz noisy audio to 16 kHz output."""

    def __init__(
        self,
        shape_plan: StaticShapePlan,
        dewavlm_cfg: Mapping[str, Any],
        adapter_cfg: Mapping[str, Any],
        vocoder_cfg: Mapping[str, Any],
        predictor_cfg: Mapping[str, Any],
        postnet_cfg: Mapping[str, Any],
    ):
        super().__init__()
        self.shape_plan = shape_plan
        self.encoder = WavLMFeatureExtractor(dewavlm_cfg, shape_plan)
        self.adapter = VocosAdapter(**dict(adapter_cfg))
        self.decoder = VocosVocoder(frames=shape_plan.vocoder_frames, **dict(vocoder_cfg))
        self.predictor = PredictorTFGridNet(shape_plan, **dict(predictor_cfg))
        self.postnet = PostNetTFGridNet(shape_plan, **dict(postnet_cfg))

        self.resample_generator_16k_to_48k = StaticSincResampler(
            16000,
            48000,
            shape_plan.vocoder_output_samples,
        )
        self.resample_predictor_16k_to_48k = StaticSincResampler(
            16000,
            48000,
            shape_plan.predictor_output_samples,
        )
        self.align_generator = StaticLengthAlign(
            shape_plan.vocoder_48k_samples,
            shape_plan.predictor_48k_samples,
        )
        self.resample_output_48k_to_16k = StaticSincResampler(
            48000,
            16000,
            shape_plan.postnet_output_samples,
        )
        self.align_output = StaticLengthAlign(
            self.resample_output_48k_to_16k.output_length,
            shape_plan.output_samples,
        )

    def prepare_for_export(self) -> None:
        for module in tuple(self.modules()):
            if isinstance(module, (MultiheadAttention, TransformerEncoder, AttnBlock, StaticGridNetV3Block)):
                module.prepare_for_export()

    def _run_pipeline(self, noisy_audio: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Tensor-only static pipeline used by the legacy ONNX exporter."""
        waveform = noisy_audio.squeeze(1)
        wavlm_layer_1, wavlm_layer_24 = self.encoder(waveform)
        adapted_features = self.adapter(wavlm_layer_1, wavlm_layer_24)
        generative_16k = self.decoder(adapted_features)
        generative_48k_before_align = self.resample_generator_16k_to_48k(generative_16k)
        predictive_16k = self.predictor(waveform)
        predictive_48k = self.resample_predictor_16k_to_48k(predictive_16k)
        generative_48k = self.align_generator(generative_48k_before_align)
        fusion_input = torch.stack((predictive_48k, generative_48k), dim=1)
        fused_48k = self.postnet(fusion_input)
        final_output = self.align_output(self.resample_output_48k_to_16k(fused_48k))
        return (
            wavlm_layer_1,
            wavlm_layer_24,
            adapted_features,
            generative_16k,
            predictive_16k,
            predictive_48k,
            generative_48k_before_align,
            generative_48k,
            fused_48k,
            final_output,
        )

    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        return self._run_pipeline(noisy_audio)[-1].unsqueeze(1)


def build_full_model(
    bundles: Mapping[str, Mapping[str, Any]],
    shape_plan: StaticShapePlan,
) -> GAPURGENetExport:
    """Construct all five checkpoint-backed components."""
    model = GAPURGENetExport(
        shape_plan,
        bundles["dewavlm"]["cfg"],
        bundles["adapter"]["cfg"],
        bundles["vocoder"]["cfg"],
        bundles["predictor"]["cfg"],
        bundles["postnet"]["cfg"],
    )
    model.encoder.wavlm.load_state_dict(bundles["dewavlm"]["state"], strict=False)
    model.adapter.load_state_dict(bundles["adapter"]["state"], strict=False)
    model.decoder.load_state_dict(bundles["vocoder"]["state"], strict=False)
    model.predictor.load_state_dict(bundles["predictor"]["state"], strict=False)
    model.postnet.load_state_dict(bundles["postnet"]["state"], strict=False)
    model.prepare_for_export()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tensor_attributes(model: Any):
    import onnx

    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t


def _materialize_external_tensors(
    model: Any,
    external_data_dir: Path,
    *,
    include_initializers: bool,
) -> None:
    from onnx import TensorProto, numpy_helper

    tensors = list(_tensor_attributes(model))
    if include_initializers:
        tensors.extend(model.graph.initializer)
    for tensor in tensors:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        array = numpy_helper.to_array(tensor, base_dir=str(external_data_dir))
        materialized = numpy_helper.from_array(array, name=tensor.name)
        tensor.CopyFrom(materialized)


def _node_attribute(node: Any, name: str, default: Any = None) -> Any:
    import onnx

    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _set_node_attribute(node: Any, name: str, value: Any) -> None:
    import onnx

    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.extend((onnx.helper.make_attribute(name, value),))


def _evaluate_static_value(
    value_name: str,
    producers: Mapping[str, tuple[int, Any]],
    initializers: Mapping[str, Any],
    cache: dict[str, np.ndarray],
    visiting: set[str],
    used_nodes: set[int],
    used_initializers: set[str],
) -> np.ndarray:
    """Evaluate the static Pad controls emitted by the legacy exporter."""
    from onnx import TensorProto, numpy_helper

    if value_name in cache:
        return cache[value_name]
    if value_name in initializers:
        used_initializers.add(value_name)
        value = np.asarray(numpy_helper.to_array(initializers[value_name]))
        cache[value_name] = value
        return value

    node_index, node = producers[value_name]
    visiting.add(value_name)
    used_nodes.add(node_index)
    try:
        inputs = [
            _evaluate_static_value(
                input_name,
                producers,
                initializers,
                cache,
                visiting,
                used_nodes,
                used_initializers,
            )
            for input_name in node.input
            if input_name
        ]
        if node.op_type == "Constant":
            value = np.asarray(numpy_helper.to_array(node.attribute[0].t))
        elif node.op_type == "Cast":
            value = inputs[0].astype(np.int64, copy=False)
        elif node.op_type == "ConstantOfShape":
            dimensions = tuple(int(item) for item in inputs[0])
            fill_tensor = _node_attribute(node, "value")
            fill_value = np.asarray(numpy_helper.to_array(fill_tensor))
            value = np.full(dimensions, fill_value.reshape(()).item(), dtype=fill_value.dtype)
        elif node.op_type == "Concat":
            value = np.concatenate(inputs, axis=int(_node_attribute(node, "axis", 0)))
        elif node.op_type == "Reshape":
            shape = tuple(int(item) for item in inputs[1].reshape(-1))
            value = np.reshape(inputs[0], shape)
        elif node.op_type == "Slice":
            starts, ends = (item.reshape(-1) for item in inputs[1:3])
            axes = inputs[3].reshape(-1) if len(inputs) >= 4 else np.arange(starts.size, dtype=np.int64)
            steps = inputs[4].reshape(-1) if len(inputs) == 5 else np.ones(starts.size, dtype=np.int64)
            slices = [slice(None)] * inputs[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
                axis_index = int(axis)
                if axis_index < 0:
                    axis_index += inputs[0].ndim
                slices[axis_index] = slice(int(start), int(end), int(step))
            value = inputs[0][tuple(slices)]
        elif node.op_type == "Transpose":
            permutation = tuple(int(item) for item in _node_attribute(node, "perm", tuple(range(inputs[0].ndim - 1, -1, -1))))
            value = np.transpose(inputs[0], permutation)
        else:
            value = inputs[0] if inputs else np.empty(0, dtype=np.int64)
    finally:
        visiting.discard(value_name)
    cache[value_name] = value
    return value


def _static_tensor_array(
    value_name: str,
    producers: Mapping[str, tuple[int, Any]],
    initializers: Mapping[str, Any],
) -> np.ndarray | None:
    """Return an immutable initializer or literal Constant tensor, else refuse it."""
    from onnx import AttributeProto, numpy_helper

    if value_name in initializers:
        return np.asarray(numpy_helper.to_array(initializers[value_name]))
    producer = producers.get(value_name)
    if producer is None:
        return None
    _node_index, node = producer
    tensor_attributes = [
        attribute
        for attribute in node.attribute
        if attribute.name == "value" and attribute.type == AttributeProto.TENSOR
    ]
    if node.op_type != "Constant" or len(node.input) != 0 or len(tensor_attributes) != 1:
        return None
    return np.asarray(numpy_helper.to_array(tensor_attributes[0].t))


def _remove_dead_rewrite_controls(
    model: Any,
    node_indices: set[int],
    control_node_indices: set[int],
    control_initializers: set[str],
) -> tuple[int, int]:
    graph_output_names = {output.name for output in model.graph.output}
    nodes = list(model.graph.node)
    removed_control_nodes = 0
    for node_index in sorted(control_node_indices, reverse=True):
        if node_index in node_indices:
            continue
        node = nodes[node_index]
        remaining_consumers = {
            input_name
            for index, candidate in enumerate(nodes)
            if index not in node_indices
            for input_name in candidate.input
            if input_name
        }
        if all(output not in graph_output_names and output not in remaining_consumers for output in node.output if output):
            node_indices.add(node_index)
            removed_control_nodes += 1

    retained_nodes = [node for index, node in enumerate(nodes) if index not in node_indices]
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)

    used_values = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name
    } | graph_output_names
    retained_initializers = []
    removed_initializers = 0
    for initializer in model.graph.initializer:
        if initializer.name in control_initializers and initializer.name not in used_values:
            removed_initializers += 1
        else:
            retained_initializers.append(initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    return removed_control_nodes, removed_initializers


def _fuse_static_resampler_pads(
    model: Any,
) -> dict[str, int]:
    """Fold exactly the three zero-constant sinc Pad -> Conv pairs."""

    nodes = list(model.graph.node)
    producers = {
        output: (index, node)
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    consumers: dict[str, list[tuple[int, Any]]] = {}
    for index, node in enumerate(nodes):
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append((index, node))
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    candidates: list[tuple[int, Any, Any, int, int, set[int], set[str]]] = []
    used_control_nodes: set[int] = set()
    used_control_initializers: set[str] = set()

    for pad_index, pad in enumerate(nodes):
        if pad.op_type != "Pad" or len(pad.output) != 1 or not pad.output[0]:
            continue
        pad_mode = _node_attribute(pad, "mode", b"constant")
        if isinstance(pad_mode, bytes):
            pad_mode = pad_mode.decode("ascii")
        if pad_mode != "constant" or len(pad.input) not in (2, 3):
            continue
        direct_consumers = consumers.get(pad.output[0], [])
        if not (len(direct_consumers) == 1 and direct_consumers[0][1].op_type == "Conv"):
            continue
        _conv_index, conv = direct_consumers[0]
        if conv.domain or len(conv.input) < 2:
            continue
        conv_auto_pad = _node_attribute(conv, "auto_pad", b"NOTSET")
        if isinstance(conv_auto_pad, bytes):
            conv_auto_pad = conv_auto_pad.decode("ascii")
        conv_pads = np.asarray(_node_attribute(conv, "pads", [0, 0]), dtype=np.int64)
        if conv_auto_pad not in ("", "NOTSET") or conv_pads.shape != (2,) or np.any(conv_pads):
            continue

        local_control_nodes: set[int] = set()
        local_control_initializers: set[str] = set()
        try:
            pads = _evaluate_static_value(
                pad.input[1],
                producers,
                initializers,
                {},
                set(),
                local_control_nodes,
                local_control_initializers,
            )
            if len(pad.input) == 3 and pad.input[2]:
                pad_value = _evaluate_static_value(
                    pad.input[2],
                    producers,
                    initializers,
                    {},
                    set(),
                    local_control_nodes,
                    local_control_initializers,
                )
                if pad_value.size != 1 or float(pad_value.reshape(-1)[0]) != 0.0:
                    continue
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        pads = np.asarray(pads)
        if pads.dtype != np.dtype(np.int64) or pads.shape != (6,):
            continue
        if np.any(pads[[0, 1, 3, 4]]) or pads[2] <= 0 or pads[5] <= pads[2]:
            continue
        left, right = (int(pads[index]) for index in (2, 5))
        resampler_rates = {
            (7, 8): (16000, 48000),
            (19, 22): (48000, 16000),
        }.get((left, right))
        if resampler_rates is None:
            continue
        expected_kernel, expected_width, expected_stride, _expected_target = _build_sinc_kernel(*resampler_rates)
        kernel = _static_tensor_array(conv.input[1], producers, initializers)
        strides = tuple(int(item) for item in _node_attribute(conv, "strides", [1]))
        dilations = tuple(int(item) for item in _node_attribute(conv, "dilations", [1]))
        group = int(_node_attribute(conv, "group", 1))
        if (
            kernel is None
            or kernel.dtype != np.dtype(np.float32)
            or kernel.shape != tuple(expected_kernel.shape)
            or not np.array_equal(kernel, expected_kernel.numpy())
            or (left, right) != (expected_width, expected_width + expected_stride)
            or strides != (expected_stride,)
            or dilations != (1,)
            or group != 1
        ):
            continue
        candidates.append((pad_index, pad, conv, left, right, local_control_nodes, local_control_initializers))
        used_control_nodes.update(local_control_nodes)
        used_control_initializers.update(local_control_initializers)

    actual_pads = tuple(sorted((left, right) for _, _, _, left, right, _, _ in candidates))
    expected_pads = tuple(sorted(_EXPECTED_STATIC_RESAMPLER_PADS))
    if actual_pads != expected_pads:
        return {
            "fused_pad_conv_pairs": 0,
            "expected_pad_conv_pairs": len(expected_pads),
            "removed_pad_nodes": 0,
            "removed_control_nodes": 0,
            "removed_control_initializers": 0,
        }

    removed_node_indices: set[int] = set()
    for pad_index, pad, conv, left, right, _control_nodes, _control_initializers in candidates:
        conv.input[0] = pad.input[0]
        _set_node_attribute(conv, "pads", [left, right])
        removed_node_indices.add(pad_index)
    removed_control_nodes, removed_control_initializers = _remove_dead_rewrite_controls(
        model,
        removed_node_indices,
        used_control_nodes,
        used_control_initializers,
    )
    return {
        "fused_pad_conv_pairs": len(candidates),
        "expected_pad_conv_pairs": len(expected_pads),
        "removed_pad_nodes": len(candidates),
        "removed_control_nodes": removed_control_nodes,
        "removed_control_initializers": removed_control_initializers,
    }


def _external_data_locations(model: Any) -> set[str]:
    import onnx

    locations: set[str] = set()
    for tensor in (*model.graph.initializer, *_tensor_attributes(model)):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        details = {entry.key: entry.value for entry in tensor.external_data}
        location = details.get("location")
        if location:
            locations.add(location)
    return locations


def _external_paths(model_path: Path, locations: set[str]) -> set[Path]:
    return {model_path.parent / location for location in locations}


def _existing_external_locations(model_path: Path) -> set[str]:
    import onnx

    if not model_path.is_file():
        return set()
    return _external_data_locations(onnx.load(str(model_path), load_external_data=False))


def _runtime_metadata() -> dict[str, str]:
    """Return exactly the static constants consumed by the inference script."""
    return {
        "in_sample_rate": str(IN_SAMPLE_RATE),
        "out_sample_rate": str(OUT_SAMPLE_RATE),
    }


def _assert_static_audio_io_contract(model: Any, shape_plan: StaticShapePlan) -> None:
    import onnx

    def signature(value: Any) -> tuple[str, str, list[int | str | None]]:
        tensor_type = value.type.tensor_type
        dimensions: list[int | str | None] = []
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.dim_param:
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(None)
        return value.name, onnx.TensorProto.DataType.Name(tensor_type.elem_type), dimensions

    actual_inputs = [signature(value) for value in model.graph.input]
    actual_outputs = [signature(value) for value in model.graph.output]
    expected_inputs = [("noisy_audio", "FLOAT", [STATIC_BATCH, 1, shape_plan.input_samples])]
    expected_outputs = [("enhanced_audio", "FLOAT", [STATIC_BATCH, 1, shape_plan.output_samples])]
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        raise RuntimeError(
            "Static ONNX audio contract changed: "
            f"inputs={actual_inputs!r}, outputs={actual_outputs!r}; "
            f"expected inputs={expected_inputs!r}, outputs={expected_outputs!r}."
        )


def _save_external_model(model: Any, model_path: Path, sidecar_name: str) -> Path:
    import onnx

    sidecar_path = model_path.with_name(sidecar_name)
    model_path.unlink(missing_ok=True)
    sidecar_path.unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar_name,
        size_threshold=0,
        convert_attribute=False,
    )
    return sidecar_path


def _owned_sidecar_paths(model_path: Path) -> set[Path]:
    return set(model_path.parent.glob(f"{model_path.name}.data.*"))


def _publish_staged_artifact(
    stage_model_path: Path,
    stage_sidecar_path: Path,
    model_path: Path,
    final_sidecar_path: Path,
    prior_external_paths: set[Path],
) -> None:
    os.replace(stage_sidecar_path, final_sidecar_path)
    os.replace(stage_model_path, model_path)
    for stale_path in prior_external_paths:
        if stale_path != final_sidecar_path:
            stale_path.unlink(missing_ok=True)


def _export_model(model: GAPURGENetExport, output_dir: Path) -> Path:
    import onnx

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "GAP_URGENet.onnx"
    metadata_path = metadata_path_for_model(model_path)
    runtime_metadata = _runtime_metadata()
    prior_external_paths = _external_paths(model_path, _existing_external_locations(model_path))
    sidecar_name = f"{model_path.name}.data.{uuid4().hex}"
    final_sidecar_path = model_path.with_name(sidecar_name)
    export_audio = torch.zeros(
        (STATIC_BATCH, 1, model.shape_plan.input_samples),
        dtype=torch.float32,
    )

    with tempfile.TemporaryDirectory(prefix=_EXPORT_TEMP_PREFIX) as temporary_dir:
        temporary_dir_path = Path(temporary_dir)
        raw_model_path = temporary_dir_path / "raw.onnx"
        torch.onnx.export(
            model,
            (export_audio,),
            str(raw_model_path),
            input_names=["noisy_audio"],
            output_names=["enhanced_audio"],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
            external_data=True,
        )
        raw_export_external_paths = _external_paths(
            raw_model_path,
            _existing_external_locations(raw_model_path),
        )
        serialized_model = onnx.load(str(raw_model_path), load_external_data=False)
        _assert_static_audio_io_contract(serialized_model, model.shape_plan)
        _materialize_external_tensors(serialized_model, raw_model_path.parent, include_initializers=True)
        for raw_export_path in raw_export_external_paths:
            raw_export_path.unlink(missing_ok=True)
        _fuse_static_resampler_pads(serialized_model)
        stage_model_path = temporary_dir_path / model_path.name
        stage_sidecar_path = _save_external_model(
            serialized_model,
            stage_model_path,
            sidecar_name,
        )
        staged_model = onnx.load(str(stage_model_path), load_external_data=False)
        _assert_static_audio_io_contract(staged_model, model.shape_plan)
        del staged_model
        del serialized_model
        _publish_staged_artifact(
            stage_model_path,
            stage_sidecar_path,
            model_path,
            final_sidecar_path,
            prior_external_paths,
        )
    export_metadata_carrier(metadata_path, runtime_metadata, OPSET)
    for stale_sidecar_path in _owned_sidecar_paths(model_path):
        if stale_sidecar_path != final_sidecar_path:
            stale_sidecar_path.unlink(missing_ok=True)
    return model_path


def _run_inference_demo(export_dir: Path) -> None:
    inference_script = Path(__file__).resolve().with_name("Inference_GAP_URGENet_ONNX.py")
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


def main() -> None:
    export_dir = onnx_model_A.expanduser().resolve().parent
    paths = checkpoint_paths(model_path)
    bundles = load_checkpoint_bundle(paths)
    plan = derive_static_shape_plan(bundles["dewavlm"]["cfg"], bundles["vocoder"]["cfg"])
    model = build_full_model(bundles, plan)
    final_model = _export_model(model, export_dir)
    print(f"Final optimized model saved to: {final_model}")
    print("\nExport done!")
    _run_inference_demo(final_model.parent)


if __name__ == "__main__":
    print("Export start ...")
    main()