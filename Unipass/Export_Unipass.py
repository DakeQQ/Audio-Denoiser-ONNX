#!/usr/bin/env python3
"""Export the UniPASE ONNX pipeline without source-repository imports.

The deployable 16 kHz enhancement path is intentionally split into dynamic-time
encoder and enhancer/vocoder graphs. WavLM produces large intermediate features
and the optional PostNet has a distinct fixed 48 kHz bandwidth-extension
contract, so separate graphs make the boundary explicit and keep ordinary 16 kHz
deployment practical.

Only ``STFT_Process.py`` beside this file is imported for signal transforms.
All UniPASE model code and checkpoint adapters live in this file.
"""

from __future__ import annotations

import ast
import gc
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import export_metadata_carrier


# ---------------------------------------------------------------------------
# Export locations
# ---------------------------------------------------------------------------

SCRIPT_DIR                  = Path(__file__).resolve().parent             # Directory that contains this exporter.
OUTPUT_DIR                  = SCRIPT_DIR / "Unipass_ONNX"                 # Final ONNX model directory.


# ---------------------------------------------------------------------------
# Source checkpoint locations
# ---------------------------------------------------------------------------

# Download the published checkpoints from https://huggingface.co/Xiaobin-Rong/unipase
# and place them in this directory, or edit these paths for another local location.
SOURCE_MODEL_DIR            = Path.home() / "Downloads" / "unipase"       # Default checkpoint directory.
DEWAVLM_CHECKPOINT          = SOURCE_MODEL_DIR / "DeWavLM-Omni.pt"         # Required DeWavLM encoder checkpoint.
ADAPTER_CHECKPOINT          = SOURCE_MODEL_DIR / "Adapter.pt"              # Required Adapter checkpoint.
VOCODER_CHECKPOINT          = SOURCE_MODEL_DIR / "Vocoder_DWO-L1.pt"       # Required Vocos decoder checkpoint.
POSTNET_CHECKPOINT          = SOURCE_MODEL_DIR / "PostNet.pt"              # Required only when PostNet export is enabled.


# ---------------------------------------------------------------------------
# User export settings
# ---------------------------------------------------------------------------

DYNAMIC_AXES                   = False      # Export dynamic time axes; batch size and channel count remain fixed at 1.
IN_SAMPLE_RATE                 = 16000      # External input rate. Dynamic export currently requires 16000 Hz.
OUT_SAMPLE_RATE                = 16000      # External base-output rate. Dynamic export currently requires 16000 Hz.
INPUT_AUDIO_LENGTH             = 32000      # Representative trace length; must map to a 16 kHz multiple of 320 samples.
IN_AUDIO_DTYPE                 = "F32"      # F32 | F16 | INT16 encoder input tensor type.
OUT_AUDIO_DTYPE                = "F32"      # F32 | F16 | INT16 enhancer output tensor type.
ENABLE_PACKET_LOSS_CONCEALMENT = True       # Detect zero packets and apply the learned WavLM packet-loss embedding.
EXPORT_POSTNET_48K             = True       # Also export the optional fixed-length 16 kHz-to-48 kHz PostNet graph.
POSTNET_OUT_SAMPLE_RATE        = 48000      # Fixed PostNet output rate supported by the published source model.


# ---------------------------------------------------------------------------
# ONNX export settings
# ---------------------------------------------------------------------------

OPSET                        = 20          # Target ONNX opset; UniPASE requires opset 18 or newer.


# ---------------------------------------------------------------------------
# Fixed UniPASE model and signal parameters
# ---------------------------------------------------------------------------

# These values define the published checkpoint contract. Edit only with matching checkpoints.
MODEL_SAMPLE_RATE          = 16000                         # Internal WavLM, Adapter, and Vocos processing rate.
WAVLM_FRAME_STRIDE         = 320                           # Samples per WavLM frame at the model sample rate.
WAVLM_PAD_REMAINDER        = 80                            # Tail padding required by the WavLM convolutional frontend.
WAVLM_CONV_RECEPTIVE_FIELD = 400                           # WavLM frontend receptive field used for duration validation.
FEATURE_CHANNELS           = 1024                          # Encoder-to-enhancer feature width.
VOCODER_NFFT               = 1280                          # Vocos inverse-DFT size.
VOCODER_HOP_LENGTH         = 320                           # Vocos hop length; matches the WavLM frame stride.
VOCODER_WINDOW_TYPE        = "hann"                        # Vocos synthesis window type.
PCM_INPUT_SCALE            = float(1.0 / 32768.0)          # INT16 encoder input scale.
PCM_OUTPUT_SCALE           = 32767.0                       # Float enhancer output scale for INT16 conversion.


# ---------------------------------------------------------------------------
# Final ONNX artifact names
# ---------------------------------------------------------------------------

ENCODER_MODEL_NAME  = "UniPASE_Encoder.onnx"              # Waveform-to-L1/L24 feature graph.
ENHANCER_MODEL_NAME = "UniPASE_Enhancer.onnx"             # L1/L24 feature-to-waveform graph.
POSTNET_MODEL_NAME  = "UniPASE_PostNet_48k.onnx"          # Optional fixed-length bandwidth-extension graph.
METADATA_MODEL_NAME = "UniPASE_Metadata.onnx"             # Runtime sample-rate and dynamic-contract metadata.


@dataclass(frozen=True)
class ExportShape:
    """Representative dimensions used to trace and validate the stage wrappers."""

    input_samples: int
    model_samples: int
    padded_samples: int
    feature_frames: int
    base_output_samples: int
    output_samples: int
    postnet_output_samples: int


def _derive_export_shape() -> ExportShape:
    if MODEL_SAMPLE_RATE != 16000:
        raise ValueError("UniPASE's base WavLM/Adapter/Vocos path is fixed at 16000 Hz.")
    if IN_SAMPLE_RATE <= 0 or OUT_SAMPLE_RATE <= 0:
        raise ValueError("IN_SAMPLE_RATE and OUT_SAMPLE_RATE must be positive integers.")
    if OUT_SAMPLE_RATE > MODEL_SAMPLE_RATE:
        raise ValueError(
            "The base ONNX pipeline only emits rates up to 16 kHz. Enable the separate "
            "PostNet export for 48 kHz bandwidth extension."
        )
    if INPUT_AUDIO_LENGTH <= 0:
        raise ValueError("INPUT_AUDIO_LENGTH must be positive.")
    model_numerator = INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE
    if model_numerator % IN_SAMPLE_RATE:
        raise ValueError(
            "INPUT_AUDIO_LENGTH * MODEL_SAMPLE_RATE must be divisible by IN_SAMPLE_RATE "
            "to preserve an exact static duration through resampling."
        )
    model_samples = model_numerator // IN_SAMPLE_RATE
    if model_samples % WAVLM_FRAME_STRIDE:
        raise ValueError(
            "The resampled 16 kHz model length must be a multiple of 320 so the WavLM "
            "and Vocos frame contracts preserve duration exactly."
        )
    output_numerator = model_samples * OUT_SAMPLE_RATE
    if output_numerator % MODEL_SAMPLE_RATE:
        raise ValueError(
            "The base output duration must be integral: model samples * OUT_SAMPLE_RATE "
            "must be divisible by MODEL_SAMPLE_RATE."
        )
    output_samples = output_numerator // MODEL_SAMPLE_RATE
    if IN_AUDIO_DTYPE not in {"F32", "F16", "INT16"}:
        raise ValueError("IN_AUDIO_DTYPE must be F32, F16, or INT16.")
    if OUT_AUDIO_DTYPE not in {"F32", "F16", "INT16"}:
        raise ValueError("OUT_AUDIO_DTYPE must be F32, F16, or INT16.")
    if DYNAMIC_AXES and (IN_SAMPLE_RATE != MODEL_SAMPLE_RATE or OUT_SAMPLE_RATE != MODEL_SAMPLE_RATE):
        raise ValueError(
            "Dynamic UniPASE export currently requires 16 kHz input and output. "
            "Fixed-rate resampling retains a static output crop."
        )
    if OPSET < 18:
        raise ValueError("OPSET must be at least 18 for the target ONNX Runtime workflow.")
    if EXPORT_POSTNET_48K and (OUT_SAMPLE_RATE != MODEL_SAMPLE_RATE or POSTNET_OUT_SAMPLE_RATE != 48000):
        raise ValueError(
            "PostNet export requires a 16 kHz base output and a fixed 48 kHz PostNet output."
        )
    if EXPORT_POSTNET_48K and OUT_AUDIO_DTYPE != "F32":
        raise ValueError("PostNet export requires OUT_AUDIO_DTYPE='F32' for its waveform input boundary.")

    padded_samples = model_samples + WAVLM_PAD_REMAINDER
    feature_frames = (padded_samples - WAVLM_CONV_RECEPTIVE_FIELD) // WAVLM_FRAME_STRIDE + 1
    base_output_samples = (feature_frames - 1) * VOCODER_HOP_LENGTH + VOCODER_HOP_LENGTH
    if base_output_samples != model_samples:
        raise AssertionError(
            f"Duration contract failed: {model_samples} model samples produce "
            f"{base_output_samples} Vocos output samples."
        )
    return ExportShape(
        input_samples=INPUT_AUDIO_LENGTH,
        model_samples=model_samples,
        padded_samples=padded_samples,
        feature_frames=feature_frames,
        base_output_samples=base_output_samples,
        output_samples=output_samples,
        postnet_output_samples=base_output_samples * POSTNET_OUT_SAMPLE_RATE // MODEL_SAMPLE_RATE,
    )


EXPORT_SHAPE = _derive_export_shape()


def _torch_dtype(audio_dtype: str) -> torch.dtype:
    return {
        "F32": torch.float32,
        "F16": torch.float16,
        "INT16": torch.int16,
    }[audio_dtype]


def _dynamic_feature_layer_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Match layer_norm over the variable [frames, channels] suffix."""

    mean = tensor.mean(dim=(-2, -1), keepdim=True)
    variance = (tensor - mean).square().mean(dim=(-2, -1), keepdim=True)
    return (tensor - mean) * torch.rsqrt(variance + 1.0e-6)


def _checkpoint_path(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} checkpoint is required but was not found at {path}. "
            "Download the published checkpoint from https://huggingface.co/Xiaobin-Rong/unipase "
            "and place it in the configured SOURCE_MODEL_DIR, or edit the checkpoint paths at "
            f"the top of {Path(__file__).name}."
        )
    return path


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    checkpoint = torch.load(_checkpoint_path(path, label), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{label} checkpoint must be a dictionary, got {type(checkpoint).__name__}.")
    return checkpoint


def _checkpoint_model_state(checkpoint: dict[str, Any], label: str) -> dict[str, torch.Tensor]:
    for key in ("generator", "model"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            tensor_items = {name: value for name, value in state.items() if isinstance(value, torch.Tensor)}
            if tensor_items:
                return tensor_items
    raise KeyError(
        f"{label} checkpoint has neither a tensor state dict in 'generator' nor 'model'. "
        f"Available keys: {sorted(checkpoint)}"
    )


def _checkpoint_config(checkpoint: dict[str, Any], label: str) -> dict[str, Any]:
    config = checkpoint.get("cfg")
    if not isinstance(config, Mapping):
        raise KeyError(
            f"{label} checkpoint must contain a dictionary 'cfg'. Available keys: {sorted(checkpoint)}"
        )
    return dict(config)


def _load_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    module.load_state_dict(state, strict=False)


def _pipeline_metadata(postnet_enabled: bool) -> dict[str, int]:
    metadata = {
        "in_sample_rate": IN_SAMPLE_RATE,
        "out_sample_rate": OUT_SAMPLE_RATE,
        "dynamic_axes": int(DYNAMIC_AXES),
    }
    if DYNAMIC_AXES:
        metadata["frame_stride"] = WAVLM_FRAME_STRIDE
        metadata["feature_channels"] = FEATURE_CHANNELS
    if postnet_enabled:
        metadata["postnet_out_sample_rate"] = POSTNET_OUT_SAMPLE_RATE
    return metadata


def _export_pipeline_metadata(postnet_enabled: bool) -> Path:
    metadata_path = OUTPUT_DIR / METADATA_MODEL_NAME
    with tempfile.TemporaryDirectory(prefix="unipase_metadata_export_") as temporary_directory:
        raw_path = Path(temporary_directory) / metadata_path.name
        export_metadata_carrier(raw_path, _pipeline_metadata(postnet_enabled), OPSET)
        _atomic_publish(raw_path, metadata_path)
    return metadata_path


def _format_bytes(byte_count: int) -> str:
    return f"{byte_count / (1024 ** 2):.1f} MiB"


def _atomic_publish(raw_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = final_path.with_name(f".{final_path.name}.staging")
    staging_path.unlink(missing_ok=True)
    try:
        shutil.copyfile(raw_path, staging_path)
        with staging_path.open("rb") as staging_file:
            os.fsync(staging_file.fileno())
        os.replace(staging_path, final_path)
        Path(f"{final_path}.data").unlink(missing_ok=True)
    finally:
        staging_path.unlink(missing_ok=True)


def _remove_unrequested_final_artifacts(postnet_enabled: bool) -> None:
    if postnet_enabled:
        return
    postnet_path = OUTPUT_DIR / POSTNET_MODEL_NAME
    postnet_path.unlink(missing_ok=True)
    Path(f"{postnet_path}.data").unlink(missing_ok=True)


def _run_inference_demo(export_dir: Path) -> None:
    inference_script = Path(__file__).resolve().with_name("Inference_Unipass_ONNX.py")
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


def _export_stage(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting {output_path.name} ...")
    with tempfile.TemporaryDirectory(prefix="unipase_onnx_export_") as temporary_directory:
        raw_path = Path(temporary_directory) / output_path.name
        with torch.inference_mode():
            torch.onnx.export(
                module.eval(),
                inputs,
                str(raw_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
                external_data=False,
            )
        _atomic_publish(raw_path, output_path)
    print(f"  Final size: {_format_bytes(output_path.stat().st_size)}")


def _feature_frames_for_audio_samples(samples: int) -> int:
    if samples <= 0 or samples % WAVLM_FRAME_STRIDE:
        raise ValueError(
            f"Dynamic UniPASE input must contain a positive multiple of {WAVLM_FRAME_STRIDE} samples, "
            f"got {samples}."
        )
    return samples // WAVLM_FRAME_STRIDE


def _export_audio(samples: int = EXPORT_SHAPE.input_samples) -> torch.Tensor:
    _feature_frames_for_audio_samples(samples)
    return torch.ones((1, 1, samples), dtype=_torch_dtype(IN_AUDIO_DTYPE))


# ---------------------------------------------------------------------------
# Inlined DeWavLM / WavLM implementation
# ---------------------------------------------------------------------------


def _parse_conv_feature_layers(expression: str) -> list[tuple[int, int, int]]:
    """Safely evaluate WavLM's compact convolution-layer configuration syntax."""

    def parse_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Tuple):
            values = tuple(parse_node(element) for element in node.elts)
            if len(values) != 3 or not all(isinstance(value, int) and value > 0 for value in values):
                raise ValueError("WavLM convolution tuples must contain three positive integers.")
            return values
        if isinstance(node, ast.List):
            return [parse_node(element) for element in node.elts]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = parse_node(node.left), parse_node(node.right)
            if not isinstance(left, list) or not isinstance(right, list):
                raise ValueError("WavLM convolution configuration only supports list concatenation.")
            return left + right
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left, right = parse_node(node.left), parse_node(node.right)
            if isinstance(left, list) and isinstance(right, int) and right >= 0:
                return left * right
            if isinstance(right, list) and isinstance(left, int) and left >= 0:
                return right * left
            raise ValueError("WavLM convolution configuration only supports list * positive-int repetition.")
        raise ValueError(f"Unsupported WavLM convolution configuration node: {ast.dump(node)}")

    try:
        parsed = parse_node(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid WavLM conv_feature_layers={expression!r}: {error}") from error
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, tuple) for item in parsed):
        raise ValueError("WavLM conv_feature_layers must evaluate to a non-empty list of tuples.")
    return parsed


class WavLMConfig:
    """Checkpoint-backed WavLM configuration with upstream-compatible defaults."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.extractor_mode = "default"
        self.encoder_layers = 12
        self.encoder_embed_dim = 768
        self.encoder_ffn_embed_dim = 3072
        self.encoder_attention_heads = 12
        self.activation_fn = "gelu"
        self.layer_norm_first = False
        self.conv_feature_layers = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
        self.conv_bias = False
        self.feature_grad_mult = 1.0
        self.normalize = False
        self.encoder_layerdrop = 0.0
        self.mask_length = 10
        self.mask_prob = 0.65
        self.mask_selection = "static"
        self.mask_other = 0.0
        self.no_mask_overlap = False
        self.mask_min_space = 1
        self.mask_channel_length = 10
        self.mask_channel_prob = 0.0
        self.mask_channel_selection = "static"
        self.mask_channel_other = 0.0
        self.no_mask_channel_overlap = False
        self.mask_channel_min_space = 1
        self.conv_pos = 128
        self.conv_pos_groups = 16
        self.relative_position_embedding = False
        self.num_buckets = 320
        self.max_distance = 1280
        self.gru_rel_pos = False
        if config is not None:
            self.__dict__.update(dict(config))


class SamePad(nn.Module):
    def __init__(self, kernel_size: int, causal: bool = False):
        super().__init__()
        self.remove = kernel_size - 1 if causal else (1 if kernel_size % 2 == 0 else 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, :, : -self.remove] if self.remove else tensor


class Swish(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * torch.sigmoid(tensor)


class GLULinear(nn.Module):
    """Source-compatible GLU linear layer for uncommon WavLM configurations."""

    def __init__(self, input_dim: int, output_dim: int, glu_type: str = "swish"):
        super().__init__()
        self.output_dim = output_dim
        self.glu_type = glu_type
        self.linear = nn.Linear(input_dim, output_dim * 2, bias=True)
        self.glu_act = {
            "sigmoid": nn.Sigmoid(),
            "swish": Swish(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }.get(glu_type)
        if self.glu_act is None and glu_type != "bilinear":
            raise ValueError(f"Unsupported WavLM GLU type: {glu_type}")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.linear(tensor)
        first, second = tensor[..., : self.output_dim], tensor[..., self.output_dim :]
        return first * second if self.glu_type == "bilinear" else first * self.glu_act(second)


def _wavlm_activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name in {"gelu_fast", "gelu_accurate"}:
        coefficient = math.sqrt(2.0 / math.pi)
        return lambda tensor: 0.5 * tensor * (1.0 + torch.tanh(coefficient * (tensor + 0.044715 * tensor.pow(3))))
    if name == "tanh":
        return torch.tanh
    if name == "linear":
        return lambda tensor: tensor
    if name == "glu":
        return lambda tensor: tensor
    raise ValueError(f"Unsupported WavLM activation: {name}")


class WavLMConvFeatureExtractor(nn.Module):
    """Convolutional WavLM feature stack with source-compatible state keys."""

    def __init__(self, conv_layers: list[tuple[int, int, int]], mode: str, conv_bias: bool):
        super().__init__()
        if mode not in {"default", "layer_norm"}:
            raise ValueError(f"Unsupported WavLM extractor mode: {mode}")
        in_channels = 1
        self.conv_layers = nn.ModuleList()
        for index, (channels, kernel_size, stride) in enumerate(conv_layers):
            convolution = nn.Conv1d(in_channels, channels, kernel_size, stride=stride, bias=conv_bias)
            if mode == "layer_norm":
                block = nn.Sequential(
                    convolution,
                    nn.Dropout(p=0.0),
                    nn.Sequential(
                        _TransposeLast(),
                        nn.LayerNorm(channels, elementwise_affine=True),
                        _TransposeLast(),
                    ),
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
        tensor = waveform.unsqueeze(1)
        for block in self.conv_layers:
            tensor = block(tensor)
        return tensor


class _TransposeLast(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(-2, -1)


class WavLMMultiheadAttention(nn.Module):
    """Fixed-batch ONNX-friendly equivalent of UniPASE's WavLM attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        has_relative_attention_bias: bool,
        num_buckets: int,
        max_distance: int,
        gru_rel_pos: bool,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("WavLM embed_dim must be divisible by encoder_attention_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.qkv_proj: nn.Linear | None = None
        self.static_frames: int | None = None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gru_rel_pos = gru_rel_pos
        if gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def fuse_qkv_projection_(self) -> None:
        if self.qkv_proj is not None:
            return

        q_proj, k_proj, v_proj = self.q_proj, self.k_proj, self.v_proj
        projections = (q_proj, k_proj, v_proj)
        bias_flags = tuple(projection.bias is not None for projection in projections)
        if any(bias_flags) and not all(bias_flags):
            raise ValueError("WavLM Q/K/V projections must either all include bias or all omit it.")

        qkv_proj = nn.Linear(
            q_proj.in_features,
            q_proj.out_features + k_proj.out_features + v_proj.out_features,
            bias=all(bias_flags),
        ).to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        with torch.no_grad():
            qkv_proj.weight.copy_(torch.cat([
                q_proj.weight * self.scaling,
                k_proj.weight,
                v_proj.weight,
            ], dim=0))
            if all(bias_flags):
                assert q_proj.bias is not None
                assert k_proj.bias is not None
                assert v_proj.bias is not None
                assert qkv_proj.bias is not None
                qkv_proj.bias.copy_(torch.cat([
                    q_proj.bias * self.scaling,
                    k_proj.bias,
                    v_proj.bias,
                ], dim=0))

        self.qkv_proj = qkv_proj
        del self.q_proj, self.k_proj, self.v_proj

    def prepare_for_export_(self, frames: int | None = None) -> None:
        self.fuse_qkv_projection_()
        if frames is None:
            return
        if frames <= 0:
            raise ValueError("WavLM export frames must be positive.")
        if self.static_frames is not None and self.static_frames != frames:
            raise ValueError("A WavLM attention module cannot be prepared for two frame counts.")
        self.static_frames = frames

    def _relative_positions_bucket(self, relative_positions: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets // 2
        relative_buckets = torch.where(relative_positions > 0, num_buckets, 0)
        positions = torch.abs(relative_positions)
        max_exact = num_buckets // 2
        is_small = positions < max_exact
        large = max_exact + (
            torch.log(positions.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        large = torch.minimum(large, torch.full_like(large, num_buckets - 1))
        return relative_buckets + torch.where(is_small, positions, large)

    def compute_position_bias(self, positions: torch.Tensor) -> torch.Tensor | None:
        if not self.has_relative_attention_bias:
            return None
        buckets = self._relative_positions_bucket(positions[None, :] - positions[:, None])
        return self.relative_attention_bias(buckets).permute(2, 0, 1).contiguous()

    def compute_static_bias(self, frames: int) -> torch.Tensor | None:
        return self.compute_position_bias(torch.arange(frames, dtype=torch.long))

    def forward(self, query: torch.Tensor, base_position_bias: torch.Tensor | None) -> torch.Tensor:
        if self.static_frames is None:
            frames, batch, _ = query.shape
        else:
            frames = self.static_frames
            batch = 1
        if self.qkv_proj is None:
            raise RuntimeError("WavLMMultiheadAttention.prepare_for_export_ must run before forward.")
        qkv = self.qkv_proj(query).reshape(frames, batch, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 1, 3, 0, 4).reshape(3, batch * self.num_heads, frames, self.head_dim)
        q, k, value = qkv.unbind(dim=0)
        attention = torch.bmm(q, k.transpose(1, 2))

        if base_position_bias is not None:
            if self.static_frames is None:
                position_bias = base_position_bias.unsqueeze(0).expand(batch, -1, -1, -1)
                position_bias = position_bias.reshape(batch * self.num_heads, frames, frames)
            else:
                position_bias = base_position_bias
            if self.gru_rel_pos:
                query_heads = query.transpose(0, 1).reshape(batch, frames, self.num_heads, self.head_dim)
                query_heads = query_heads.permute(0, 2, 1, 3)
                gates = torch.sigmoid(self.grep_linear(query_heads).reshape(batch, self.num_heads, frames, 2, 4).sum(-1))
                gate_a, gate_b = gates.split(1, dim=-1)
                gate = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = position_bias * gate.reshape(batch * self.num_heads, frames, 1)
            attention = attention + position_bias

        attention = F.softmax(attention, dim=-1)
        attended = torch.bmm(attention, value)
        attended = attended.reshape(batch, self.num_heads, frames, self.head_dim)
        attended = attended.permute(2, 0, 1, 3).reshape(frames, batch, self.embed_dim)
        return self.out_proj(attended)


class WavLMTransformerLayer(nn.Module):
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
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer_norm_first = layer_norm_first
        self.activation_name = activation_fn
        self.activation_fn = _wavlm_activation(activation_fn)
        self.self_attn = WavLMMultiheadAttention(
            embedding_dim,
            num_attention_heads,
            has_relative_attention_bias,
            num_buckets,
            max_distance,
            gru_rel_pos,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = GLULinear(embedding_dim, ffn_embedding_dim, "swish") if activation_fn == "glu" else nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def _feed_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "glu":
            return self.fc2(self.fc1(tensor))
        return self.fc2(self.activation_fn(self.fc1(tensor)))

    def forward(self, tensor: torch.Tensor, position_bias: torch.Tensor | None) -> torch.Tensor:
        if self.layer_norm_first:
            residual = tensor
            tensor = self.self_attn_layer_norm(tensor)
            tensor = residual + self.self_attn(tensor, position_bias)
            residual = tensor
            tensor = self.final_layer_norm(tensor)
            return residual + self._feed_forward(tensor)

        tensor = tensor + self.self_attn(tensor, position_bias)
        tensor = self.self_attn_layer_norm(tensor)
        tensor = tensor + self._feed_forward(tensor)
        return self.final_layer_norm(tensor)


class WavLMTransformer(nn.Module):
    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.embedding_dim = int(config.encoder_embed_dim)
        position_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=int(config.conv_pos),
            padding=int(config.conv_pos) // 2,
            groups=int(config.conv_pos_groups),
        )
        self.pos_conv = nn.Sequential(
            nn.utils.weight_norm(position_conv, name="weight", dim=2),
            SamePad(int(config.conv_pos)),
            nn.GELU(),
        )
        self.relative_position_embedding = bool(config.relative_position_embedding)
        self.layers = nn.ModuleList(
            [
                WavLMTransformerLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=int(config.encoder_ffn_embed_dim),
                    num_attention_heads=int(config.encoder_attention_heads),
                    activation_fn=str(config.activation_fn),
                    layer_norm_first=bool(config.layer_norm_first),
                    has_relative_attention_bias=self.relative_position_embedding and index == 0,
                    num_buckets=int(config.num_buckets),
                    max_distance=int(config.max_distance),
                    gru_rel_pos=bool(config.gru_rel_pos),
                )
                for index in range(int(config.encoder_layers))
            ]
        )
        self.layer_norm_first = bool(config.layer_norm_first)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.register_buffer("static_position_bias", torch.empty(0), persistent=False)
        self._has_static_position_bias = False

    def prepare_for_export_(self, frames: int | None = None) -> None:
        position_conv = self.pos_conv[0]
        if hasattr(position_conv, "weight_g"):
            nn.utils.remove_weight_norm(position_conv)
        for layer in self.layers:
            layer.self_attn.prepare_for_export_(frames)
        if frames is None:
            self.static_position_bias = torch.empty(0)
            self._has_static_position_bias = False
            return
        position_bias = self.layers[0].self_attn.compute_static_bias(frames)
        if position_bias is not None:
            self.static_position_bias = position_bias.detach().contiguous()
            self._has_static_position_bias = True

    def forward_collect(self, tensor: torch.Tensor, output_layer: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if output_layer is not None and not 1 <= output_layer <= len(self.layers):
            raise ValueError(f"Requested WavLM output layer {output_layer} is unavailable.")
        tensor = tensor + self.pos_conv(tensor.transpose(1, 2)).transpose(1, 2)
        if not self.layer_norm_first:
            tensor = self.layer_norm(tensor)
        tensor = tensor.transpose(0, 1)
        position_bias = self.static_position_bias if self._has_static_position_bias else None
        if position_bias is None and self.relative_position_embedding:
            positions = torch.arange(tensor.shape[0], dtype=torch.long, device=tensor.device)
            position_bias = self.layers[0].self_attn.compute_position_bias(positions)
        layer_one = None
        for index, layer in enumerate(self.layers):
            tensor = layer(tensor, position_bias)
            if index == 0:
                layer_one = tensor.transpose(0, 1)
            if output_layer is not None and index + 1 == output_layer:
                break
        if layer_one is None:
            raise RuntimeError("DeWavLM must contain at least one transformer layer.")
        return layer_one, tensor.transpose(0, 1)


class WavLMModel(nn.Module):
    """WavLM inference subset used by UniPASE's DeWavLM feature extractor."""

    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        conv_layers = _parse_conv_feature_layers(str(config.conv_feature_layers))
        self.embed = conv_layers[-1][0]
        self.feature_extractor = WavLMConvFeatureExtractor(
            conv_layers,
            mode=str(config.extractor_mode),
            conv_bias=bool(config.conv_bias),
        )
        self.post_extract_proj = nn.Linear(self.embed, int(config.encoder_embed_dim)) if self.embed != int(config.encoder_embed_dim) else None
        self.mask_emb = nn.Parameter(torch.empty(int(config.encoder_embed_dim)).uniform_())
        self.encoder = WavLMTransformer(config)
        self.layer_norm = nn.LayerNorm(self.embed)

    def expected_frames(self, samples: int) -> int:
        length = samples
        for _channels, kernel_size, stride in _parse_conv_feature_layers(str(self.config.conv_feature_layers)):
            length = (length - kernel_size) // stride + 1
        return length

    def prepare_for_export_(self, frames: int | None = None) -> None:
        self.encoder.prepare_for_export_(frames)

    def forward(
        self,
        waveform: torch.Tensor,
        packet_mask: torch.Tensor | None,
        output_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(waveform).transpose(1, 2)
        features = self.layer_norm(features)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        if packet_mask is not None:
            features = torch.where(packet_mask.unsqueeze(-1), self.mask_emb.reshape(1, 1, -1), features)
        return self.encoder.forward_collect(features, output_layer=output_layer)


class UniPASEEncoderExport(nn.Module):
    """Waveform-to-DeWavLM-L1/L24 stage with optional PLC detection."""

    def __init__(self, checkpoint_config: Mapping[str, Any], checkpoint_state: dict[str, torch.Tensor]):
        super().__init__()
        self.config = WavLMConfig(checkpoint_config)
        if int(self.config.encoder_layers) < 24:
            raise ValueError("UniPASE requires a DeWavLM checkpoint with at least 24 transformer layers.")
        if int(self.config.encoder_embed_dim) != FEATURE_CHANNELS:
            raise ValueError(
                f"UniPASE Adapter expects {FEATURE_CHANNELS} channels, but DeWavLM cfg specifies "
                f"encoder_embed_dim={self.config.encoder_embed_dim}."
            )
        self.wavlm = WavLMModel(self.config)
        _load_state(self.wavlm, checkpoint_state)
        actual_frames = self.wavlm.expected_frames(EXPORT_SHAPE.padded_samples)
        if actual_frames != EXPORT_SHAPE.feature_frames:
            raise ValueError(
                f"DeWavLM checkpoint produces {actual_frames} feature frames for "
                f"{EXPORT_SHAPE.padded_samples} padded samples, but the representative export requires "
                f"{EXPORT_SHAPE.feature_frames}."
            )
        self.wavlm.prepare_for_export_(None if DYNAMIC_AXES else actual_frames)
        self.enable_plc = ENABLE_PACKET_LOSS_CONCEALMENT
        self.input_resample = IN_SAMPLE_RATE != MODEL_SAMPLE_RATE
        self.resample_to_model = (
            FixedSincResample(IN_SAMPLE_RATE, MODEL_SAMPLE_RATE, EXPORT_SHAPE.input_samples)
            if self.input_resample
            else None
        )
        self.register_buffer(
            "wavlm_tail_padding",
            torch.zeros((1, 1, WAVLM_PAD_REMAINDER), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, noisy_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio = noisy_audio if IN_AUDIO_DTYPE == "F32" else noisy_audio.float()
        if IN_AUDIO_DTYPE == "INT16":
            audio = audio * PCM_INPUT_SCALE
        if self.input_resample:
            audio = self.resample_to_model(audio)
        packet_mask = None
        if self.enable_plc:
            if DYNAMIC_AXES:
                packets = audio.reshape(1, -1, WAVLM_FRAME_STRIDE)
            else:
                packets = audio.reshape(1, EXPORT_SHAPE.feature_frames, WAVLM_FRAME_STRIDE)
            packet_mask = (packets.abs() < 1e-7).float().mean(dim=-1) >= 0.99
        audio = torch.cat((audio, self.wavlm_tail_padding), dim=-1)
        layer_one, layer_last = self.wavlm(audio.squeeze(1), packet_mask, output_layer=24)
        if DYNAMIC_AXES:
            return (
                _dynamic_feature_layer_norm(layer_one),
                _dynamic_feature_layer_norm(layer_last),
            )
        normalized_shape = (EXPORT_SHAPE.feature_frames, FEATURE_CHANNELS)
        return (
            F.layer_norm(layer_one, normalized_shape, eps=1e-6),
            F.layer_norm(layer_last, normalized_shape, eps=1e-6),
        )


# The inlined Adapter, Vocos, and optional PostNet module definitions follow
# below. They are added before export execution so this script remains portable
# when copied with its checkpoints and STFT_Process.py.


# ---------------------------------------------------------------------------
# Inlined Adapter / Vocos implementation
# ---------------------------------------------------------------------------


def _swish(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * torch.sigmoid(tensor)


def _vocos_group_norm(channels: int) -> nn.GroupNorm:
    if channels % 32:
        raise ValueError(f"Vocos GroupNorm requires channels divisible by 32, got {channels}.")
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class VocosConvNeXtBlock(nn.Module):
    """State-key-compatible ConvNeXt block shared by Adapter and Vocoder."""

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = tensor
        tensor = self.dwconv(tensor).transpose(1, 2)
        tensor = self.pwconv2(self.act(self.pwconv1(self.norm(tensor))))
        tensor = (self.gamma * tensor).transpose(1, 2)
        return residual + tensor


class VocosResnetBlock(nn.Module):
    """The pre-ConvNeXt temporal context block used in both UniPASE stages."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.norm1 = _vocos_group_norm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _vocos_group_norm(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(_swish(self.norm1(tensor)))
        hidden = self.conv2(_swish(self.norm2(hidden)))
        return tensor + hidden


class VocosAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_channels = channels
        self.norm = _vocos_group_norm(channels)
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.k = nn.Conv1d(channels, channels, kernel_size=1)
        self.v = nn.Conv1d(channels, channels, kernel_size=1)
        self.qkv_proj: nn.Conv1d | None = None
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = float(channels ** -0.5)

    def fuse_qkv_for_export_(self) -> None:
        """Combine immutable Q/K/V projections and absorb the score scale into Q."""

        if self.qkv_proj is not None:
            return
        q_proj, k_proj, v_proj = self.q, self.k, self.v
        projections = (q_proj, k_proj, v_proj)
        if (
            q_proj.groups != 1
            or any(
                projection.in_channels != q_proj.in_channels
                or projection.out_channels != q_proj.out_channels
                or projection.kernel_size != q_proj.kernel_size
                or projection.stride != q_proj.stride
                or projection.padding != q_proj.padding
                or projection.dilation != q_proj.dilation
                or projection.groups != q_proj.groups
                or projection.padding_mode != q_proj.padding_mode
                for projection in projections[1:]
            )
        ):
            raise ValueError("Vocos attention Q/K/V fusion requires matching ungrouped Conv1d projections.")
        bias_flags = tuple(projection.bias is not None for projection in projections)
        if any(bias_flags) and not all(bias_flags):
            raise ValueError("Vocos attention Q/K/V projections must either all include bias or all omit it.")

        qkv_proj = nn.Conv1d(
            q_proj.in_channels,
            q_proj.out_channels * 3,
            q_proj.kernel_size,
            stride=q_proj.stride,
            padding=q_proj.padding,
            dilation=q_proj.dilation,
            groups=q_proj.groups,
            bias=all(bias_flags),
            padding_mode=q_proj.padding_mode,
        ).to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        with torch.no_grad():
            qkv_proj.weight.copy_(torch.cat([
                q_proj.weight * self.scale,
                k_proj.weight,
                v_proj.weight,
            ], dim=0))
            if all(bias_flags):
                assert q_proj.bias is not None
                assert k_proj.bias is not None
                assert v_proj.bias is not None
                assert qkv_proj.bias is not None
                qkv_proj.bias.copy_(torch.cat([
                    q_proj.bias * self.scale,
                    k_proj.bias,
                    v_proj.bias,
                ], dim=0))
        self.qkv_proj = qkv_proj
        del self.q, self.k, self.v

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(tensor)
        if self.qkv_proj is None:
            query, key, value = self.q(hidden), self.k(hidden), self.v(hidden)
        else:
            query, key, value = self.qkv_proj(hidden).split(self.in_channels, dim=1)
        query = query.transpose(1, 2)
        attention_scores = torch.bmm(query, key)
        if self.qkv_proj is None:
            attention_scores = attention_scores * self.scale
        attention = F.softmax(attention_scores, dim=2)
        hidden = self.proj_out(torch.bmm(value, attention.transpose(1, 2)))
        return tensor + hidden


class VocosBackbone(nn.Module):
    """Source-compatible Vocos backbone used by Adapter and Vocoder checkpoints."""

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        num_res: int = 4,
        num_attn: int = 1,
    ):
        super().__init__()
        if num_res % 2:
            raise ValueError("Vocos num_res must be even.")
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale = 1.0 / num_layers
        self.convnext = nn.ModuleList(
            [VocosConvNeXtBlock(dim, intermediate_dim, layer_scale) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        context_layers: list[nn.Module] = []
        context_layers.extend(VocosResnetBlock(dim) for _ in range(num_res // 2))
        context_layers.extend(VocosAttentionBlock(dim) for _ in range(num_attn))
        context_layers.extend(VocosResnetBlock(dim) for _ in range(num_res // 2))
        context_layers.append(_vocos_group_norm(dim))
        self.pos_net = nn.Sequential(*context_layers)

    def prepare_for_export_(self) -> None:
        for block in self.pos_net:
            if isinstance(block, VocosAttentionBlock):
                block.fuse_qkv_for_export_()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.embed(tensor)
        tensor = self.pos_net(tensor)
        tensor = self.norm(tensor.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            tensor = block(tensor)
        return self.final_layer_norm(tensor.transpose(1, 2)).transpose(1, 2)


class VocosAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int = FEATURE_CHANNELS,
        dim: int = FEATURE_CHANNELS,
        intermediate_dim: int = 4096,
        num_layers: int = 12,
        output_channels: int = FEATURE_CHANNELS,
    ):
        super().__init__()
        self.proj = nn.Linear(input_channels, input_channels)
        self.decoder = VocosBackbone(input_channels, dim, intermediate_dim, num_layers, num_res=4, num_attn=1)
        self.head = nn.Linear(dim, output_channels)

    def forward(self, noisy_features: torch.Tensor, reference_features: torch.Tensor) -> torch.Tensor:
        tensor = self.proj(reference_features) + noisy_features
        tensor = self.decoder(tensor.transpose(1, 2)).transpose(1, 2)
        return self.head(tensor)


class LegacyVocosISTFT(nn.Module):
    """Retains the checkpoint's legacy ``head.istft.window`` state key only."""

    def __init__(self, n_fft: int):
        super().__init__()
        self.register_buffer("window", torch.hann_window(n_fft))


class VocosISTFTHead(nn.Module):
    """Checkpoint-compatible Vocos output head; export wrapper owns reconstruction."""

    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = LegacyVocosISTFT(n_fft)
        self.n_fft = n_fft
        self.hop_length = hop_length


class VocosVocoder(nn.Module):
    def __init__(
        self,
        input_channels: int = FEATURE_CHANNELS,
        dim: int = 768,
        intermediate_dim: int = 2304,
        num_layers: int = 12,
        num_res: int = 4,
        num_attn: int = 1,
        n_fft: int = VOCODER_NFFT,
        hop_length: int = VOCODER_HOP_LENGTH,
    ):
        super().__init__()
        self.decoder = VocosBackbone(input_channels, dim, intermediate_dim, num_layers, num_res, num_attn)
        self.head = VocosISTFTHead(dim, n_fft, hop_length)


class VocosSamePaddingISTFT(nn.Module):
    """Use the approved STFT_Process inverse DFT then reproduce Vocos's crop."""

    def __init__(self, n_fft: int, hop_length: int, frames: int, dynamic: bool = False):
        super().__init__()
        self.same_pad = (n_fft - hop_length) // 2
        self.istft_model = STFT_Process(
            model_type="istft_B",
            n_fft=n_fft,
            hop_len=hop_length,
            win_length=n_fft,
            max_frames=frames,
            window_type=VOCODER_WINDOW_TYPE,
            center_pad=False,
            pad_mode="constant",
            static_norm=not dynamic,
        ).eval()
        raw_output_samples = n_fft + hop_length * (frames - 1)
        expected = raw_output_samples - 2 * self.same_pad
        if not dynamic and expected != EXPORT_SHAPE.base_output_samples:
            raise ValueError(
                f"Vocos ISTFT contract produced {expected} samples; expected "
                f"{EXPORT_SHAPE.base_output_samples}."
            )

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        packed = torch.cat((real, imag), dim=1)
        waveform = self.istft_model._istft_B_packed_forward(packed)
        return waveform[..., self.same_pad : -self.same_pad]


class UniPASEEnhancerExport(nn.Module):
    """Adapter + Vocos stage: [B,T,1024] x2 -> enhanced waveform."""

    def __init__(
        self,
        adapter_config: Mapping[str, Any],
        adapter_state: dict[str, torch.Tensor],
        vocoder_config: Mapping[str, Any],
        vocoder_state: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.adapter = VocosAdapter(**dict(adapter_config))
        self.vocoder = VocosVocoder(**dict(vocoder_config))
        _load_state(self.adapter, adapter_state)
        _load_state(self.vocoder, vocoder_state)
        self.adapter.decoder.prepare_for_export_()
        self.vocoder.decoder.prepare_for_export_()
        if self.adapter.proj.in_features != FEATURE_CHANNELS or self.adapter.head.out_features != FEATURE_CHANNELS:
            raise ValueError("Adapter checkpoint does not expose the required 1024-channel UniPASE interface.")
        if self.vocoder.head.n_fft != VOCODER_NFFT or self.vocoder.head.hop_length != VOCODER_HOP_LENGTH:
            raise ValueError(
                "Vocoder checkpoint must use UniPASE's n_fft=1280 and hop_length=320 "
                "for the base waveform contract."
            )
        self.istft = VocosSamePaddingISTFT(
            self.vocoder.head.n_fft,
            self.vocoder.head.hop_length,
            EXPORT_SHAPE.feature_frames,
            dynamic=DYNAMIC_AXES,
        )
        self.output_resample = OUT_SAMPLE_RATE != MODEL_SAMPLE_RATE
        self.resample_to_output = (
            FixedSincResample(MODEL_SAMPLE_RATE, OUT_SAMPLE_RATE, EXPORT_SHAPE.base_output_samples)
            if self.output_resample
            else None
        )

    def forward(self, noisy_features: torch.Tensor, reference_features: torch.Tensor) -> torch.Tensor:
        features = self.adapter(noisy_features, reference_features)
        hidden = self.vocoder.decoder(features.transpose(1, 2))
        spectrum = self.vocoder.head.out(hidden.transpose(1, 2)).transpose(1, 2)
        log_magnitude, phase = spectrum.split(VOCODER_NFFT // 2 + 1, dim=1)
        magnitude = torch.exp(torch.clamp(log_magnitude, min=-20.0, max=5.0))
        waveform = self.istft(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        if self.output_resample:
            waveform = self.resample_to_output(waveform)
        if OUT_AUDIO_DTYPE == "INT16":
            return torch.clamp(waveform * PCM_OUTPUT_SCALE, min=-32768.0, max=32767.0).to(torch.int16)
        if OUT_AUDIO_DTYPE == "F16":
            return waveform.to(torch.float16)
        return waveform


# ---------------------------------------------------------------------------
# Inlined optional PostNet bandwidth-extension implementation
# ---------------------------------------------------------------------------


class FixedSincResample(nn.Module):
    """Static ONNX-friendly Hann-windowed sinc resampler for a fixed rate pair.

    This mirrors torchaudio's default sinc-interpolation construction for the
    fixed 16 kHz -> 48 kHz PostNet boundary while leaving only Conv and Slice in
    the exported graph. The kernel is immutable after construction.
    """

    def __init__(self, orig_rate: int, new_rate: int, input_samples: int):
        super().__init__()
        if orig_rate <= 0 or new_rate <= 0:
            raise ValueError("Resampler rates must be positive.")
        divisor = math.gcd(orig_rate, new_rate)
        reduced_orig = orig_rate // divisor
        reduced_new = new_rate // divisor
        lowpass_filter_width = 6
        rolloff = 0.99
        base_frequency = min(reduced_orig, reduced_new) * rolloff
        width = math.ceil(lowpass_filter_width * reduced_orig / base_frequency)
        indices = torch.arange(-width, width + reduced_orig, dtype=torch.float32)[None, None]
        phases = torch.arange(0, -reduced_new, -1, dtype=torch.float32)[:, None, None]
        time = phases / reduced_new + indices / reduced_orig
        time = torch.clamp(time * base_frequency, -lowpass_filter_width, lowpass_filter_width)
        radians = time * math.pi
        window = torch.cos(radians / lowpass_filter_width / 2.0).pow(2)
        sinc = torch.where(radians == 0, torch.ones_like(radians), torch.sin(radians) / radians)
        kernel = sinc * window * (base_frequency / reduced_orig)
        self.register_buffer("kernel", kernel.contiguous())
        self.width = width
        self.reduced_orig = reduced_orig
        self.output_samples = math.ceil(input_samples * new_rate / orig_rate)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        padded = F.pad(audio, (self.width, self.width + self.reduced_orig))
        resampled = F.conv1d(padded, self.kernel, stride=self.reduced_orig)
        resampled = resampled.transpose(1, 2).reshape(1, 1, -1)
        return resampled[..., : self.output_samples]


class PostNetBandSplit(nn.Module):
    def __init__(self, n_bands: int = 3):
        super().__init__()
        self.n_bands = n_bands
        self.frequency_bins = 769
        self.sub_bins = self.frequency_bins // n_bands + 1

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                tensor[..., index * (self.sub_bins - 1) : index * (self.sub_bins - 1) + self.sub_bins]
                for index in range(self.n_bands)
            ],
            dim=1,
        ).contiguous()

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        pieces = [tensor[:, :2]]
        pieces.extend(tensor[:, 2 * index : 2 * (index + 1), :, 1:] for index in range(1, self.n_bands))
        return torch.cat(pieces, dim=-1).contiguous()


class PostNetLayerNormalization(nn.Module):
    """Source-compatible per-axis affine normalization for GridNet attention."""

    def __init__(self, input_dim: int, dim: int = 1, total_dim: int = 4, eps: float = 1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        parameter_shape = [1 if index != self.dim else input_dim for index in range(total_dim)]
        self.gamma = nn.Parameter(torch.ones(*parameter_shape, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*parameter_shape, dtype=torch.float32))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=self.dim, keepdim=True)
        std = torch.sqrt(tensor.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        return ((tensor - mean) / std) * self.gamma + self.beta


class PostNetAllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension: tuple[int, int], eps: float = 1e-5):
        super().__init__()
        heads, head_features = input_dimension
        self.gamma = nn.Parameter(torch.ones(1, heads, head_features, 1, 1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, heads, head_features, 1, 1, dtype=torch.float32))
        self.act = nn.PReLU(num_parameters=heads, init=0.25)
        self.eps = eps
        self.heads = heads
        self.head_features = head_features
        self.register_buffer("last_feature_gamma", torch.empty(0), persistent=False)
        self.register_buffer("last_feature_beta", torch.empty(0), persistent=False)
        self._has_precomputed_last_feature_affine = False

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _channels, time_frames, frequency_bins = tensor.shape
        tensor = tensor.reshape(batch, self.heads, self.head_features, time_frames, frequency_bins)
        return self.forward_heads(tensor)

    def forward_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize an already head-shaped [B, H, D, T, F] tensor."""

        tensor = self.act(tensor)
        mean = tensor.mean(dim=2, keepdim=True)
        std = torch.sqrt(tensor.var(dim=2, unbiased=False, keepdim=True) + self.eps)
        return ((tensor - mean) / std) * self.gamma + self.beta

    def prepare_last_feature_for_export_(self) -> None:
        """Cache fixed affine tensors in the [B, H, T, F, D] export layout."""

        with torch.no_grad():
            self.last_feature_gamma = self.gamma.detach().permute(0, 1, 3, 4, 2).contiguous()
            self.last_feature_beta = self.beta.detach().permute(0, 1, 3, 4, 2).contiguous()
        self._has_precomputed_last_feature_affine = True

    def forward_last_feature(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize an already head-shaped [B, H, T, F, D] tensor."""

        tensor = self.act(tensor)
        if self._has_precomputed_last_feature_affine:
            tensor = F.layer_norm(tensor, (self.head_features,), eps=self.eps)
            return tensor * self.last_feature_gamma + self.last_feature_beta
        mean = tensor.mean(dim=-1, keepdim=True)
        std = torch.sqrt(tensor.var(dim=-1, unbiased=False, keepdim=True) + self.eps)
        gamma = self.gamma.permute(0, 1, 3, 4, 2)
        beta = self.beta.permute(0, 1, 3, 4, 2)
        return ((tensor - mean) / std) * gamma + beta


class PostNetGridBlock(nn.Module):
    """Fixed-shape GridNet block with static Gather patch extraction.

    The source uses F.unfold for the intra/inter LSTM patch views. At a static
    export shape those patches are fixed, so static Gather indices are exact and
    avoid a dynamic im2col graph.
    """

    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        hidden_channels: int,
        n_head: int,
        qk_output_channel: int,
        activation: str = "prelu",
        eps: float = 1e-5,
    ):
        super().__init__()
        if activation != "prelu":
            raise ValueError("UniPASE PostNet only supports PReLU GridNet activation.")
        if emb_dim % n_head:
            raise ValueError("PostNet emb_dim must be divisible by attn_n_head.")
        packed_channels = emb_dim * emb_ks
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(packed_channels, hidden_channels, 1, batch_first=True, bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(packed_channels, hidden_channels, 1, batch_first=True, bidirectional=True)
        self.inter_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)
        self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * qk_output_channel, 1)
        self.attn_norm_Q = PostNetAllHeadPReLULayerNormalization4DC((n_head, qk_output_channel), eps=eps)
        self.attn_conv_K = nn.Conv2d(emb_dim, n_head * qk_output_channel, 1)
        self.attn_norm_K = PostNetAllHeadPReLULayerNormalization4DC((n_head, qk_output_channel), eps=eps)
        self.attn_norm_QK: PostNetAllHeadPReLULayerNormalization4DC | None = None
        self.attn_conv_V = nn.Conv2d(emb_dim, emb_dim, 1)
        self.attn_qkv_proj: nn.Conv2d | None = None
        self.attn_qkv_split_sizes = (n_head * qk_output_channel, n_head * qk_output_channel, emb_dim)
        self.attn_qkv_head_split_sizes = (qk_output_channel, qk_output_channel, emb_dim // n_head)
        self.attn_norm_V = PostNetAllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps)
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.PReLU(),
            PostNetLayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        self.register_buffer("intra_indices", torch.empty(0, dtype=torch.int64), persistent=False)
        self.register_buffer("inter_indices", torch.empty(0, dtype=torch.int64), persistent=False)
        self.static_time = 0
        self.static_frequency = 0
        self.padded_time = 0
        self.padded_frequency = 0
        self.intra_segments = 0
        self.inter_segments = 0
        self.attention_scale = 0.0

    def fuse_attention_qk_normalization_for_export_(self) -> None:
        """Pack independent Q/K head normalizers into one equivalent module."""

        if self.attn_norm_QK is not None:
            return
        q_norm, k_norm = self.attn_norm_Q, self.attn_norm_K
        if (
            q_norm.heads != self.n_head
            or k_norm.heads != self.n_head
            or q_norm.head_features != k_norm.head_features
            or q_norm.eps != k_norm.eps
            or q_norm.gamma.device != k_norm.gamma.device
            or q_norm.gamma.dtype != k_norm.gamma.dtype
        ):
            raise ValueError("PostNet Q/K normalization fusion requires matching head layouts and parameter types.")
        qk_norm = PostNetAllHeadPReLULayerNormalization4DC(
            (q_norm.heads + k_norm.heads, q_norm.head_features),
            eps=q_norm.eps,
        ).to(device=q_norm.gamma.device, dtype=q_norm.gamma.dtype)
        if self.attention_scale <= 0.0:
            raise RuntimeError("PostNet attention scale must be prepared before Q/K normalization fusion.")
        with torch.no_grad():
            qk_norm.gamma.copy_(torch.cat((q_norm.gamma, k_norm.gamma), dim=1))
            qk_norm.beta.copy_(torch.cat((q_norm.beta, k_norm.beta), dim=1))
            qk_norm.act.weight.copy_(torch.cat((q_norm.act.weight, k_norm.act.weight)))
            qk_norm.gamma[:, :self.n_head].mul_(self.attention_scale)
            qk_norm.beta[:, :self.n_head].mul_(self.attention_scale)
        qk_norm.prepare_last_feature_for_export_()
        self.attn_norm_QK = qk_norm

    def fuse_attention_qkv_for_export_(self) -> None:
        """Combine immutable same-input PostNet attention projections into one Conv2d."""

        if self.attn_qkv_proj is not None:
            self.fuse_attention_qk_normalization_for_export_()
            self.attn_norm_V.prepare_last_feature_for_export_()
            return
        q_proj, k_proj, v_proj = self.attn_conv_Q, self.attn_conv_K, self.attn_conv_V
        projections = (q_proj, k_proj, v_proj)
        if (
            q_proj.groups != 1
            or any(
                projection.in_channels != q_proj.in_channels
                or projection.kernel_size != q_proj.kernel_size
                or projection.stride != q_proj.stride
                or projection.padding != q_proj.padding
                or projection.dilation != q_proj.dilation
                or projection.groups != q_proj.groups
                or projection.padding_mode != q_proj.padding_mode
                for projection in projections[1:]
            )
        ):
            raise ValueError("PostNet attention Q/K/V fusion requires matching ungrouped Conv2d projections.")
        bias_flags = tuple(projection.bias is not None for projection in projections)
        if any(bias_flags) and not all(bias_flags):
            raise ValueError("PostNet attention Q/K/V projections must either all include bias or all omit it.")

        qkv_proj = nn.Conv2d(
            q_proj.in_channels,
            sum(self.attn_qkv_split_sizes),
            q_proj.kernel_size,
            stride=q_proj.stride,
            padding=q_proj.padding,
            dilation=q_proj.dilation,
            groups=q_proj.groups,
            bias=all(bias_flags),
            padding_mode=q_proj.padding_mode,
        ).to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        with torch.no_grad():
            q_head_features, k_head_features, v_head_features = self.attn_qkv_head_split_sizes
            expected_channels = (
                self.n_head * q_head_features,
                self.n_head * k_head_features,
                self.n_head * v_head_features,
            )
            if tuple(projection.out_channels for projection in projections) != expected_channels:
                raise ValueError("PostNet attention projection widths do not match the configured head layout.")
            qkv_proj.weight.copy_(torch.cat([
                q_proj.weight.reshape(self.n_head, q_head_features, *q_proj.weight.shape[1:]),
                k_proj.weight.reshape(self.n_head, k_head_features, *k_proj.weight.shape[1:]),
                v_proj.weight.reshape(self.n_head, v_head_features, *v_proj.weight.shape[1:]),
            ], dim=1).reshape_as(qkv_proj.weight))
            if all(bias_flags):
                assert q_proj.bias is not None
                assert k_proj.bias is not None
                assert v_proj.bias is not None
                assert qkv_proj.bias is not None
                qkv_proj.bias.copy_(torch.cat([
                    q_proj.bias.reshape(self.n_head, q_head_features),
                    k_proj.bias.reshape(self.n_head, k_head_features),
                    v_proj.bias.reshape(self.n_head, v_head_features),
                ], dim=1).reshape_as(qkv_proj.bias))
        self.attn_qkv_proj = qkv_proj
        del self.attn_conv_Q, self.attn_conv_K, self.attn_conv_V
        self.fuse_attention_qk_normalization_for_export_()
        self.attn_norm_V.prepare_last_feature_for_export_()

    def prepare_for_export_(self, time_frames: int, frequency_bins: int) -> None:
        overlap = self.emb_ks - self.emb_hs
        padded_time = math.ceil((time_frames + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        padded_frequency = math.ceil((frequency_bins + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        intra_segments = (padded_frequency - self.emb_ks) // self.emb_hs + 1
        inter_segments = (padded_time - self.emb_ks) // self.emb_hs + 1
        self.intra_indices = (
            torch.arange(intra_segments, dtype=torch.int64).unsqueeze(1) * self.emb_hs
            + torch.arange(self.emb_ks, dtype=torch.int64).unsqueeze(0)
        ).reshape(-1)
        self.inter_indices = (
            torch.arange(inter_segments, dtype=torch.int64).unsqueeze(1) * self.emb_hs
            + torch.arange(self.emb_ks, dtype=torch.int64).unsqueeze(0)
        ).reshape(-1)
        self.static_time = time_frames
        self.static_frequency = frequency_bins
        self.padded_time = padded_time
        self.padded_frequency = padded_frequency
        self.intra_segments = intra_segments
        self.inter_segments = inter_segments
        self.attention_scale = float(1.0 / math.sqrt(self.attn_norm_Q.head_features * frequency_bins))
        self.fuse_attention_qkv_for_export_()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.static_time <= 0:
            raise RuntimeError("PostNetGridBlock.prepare_for_export_ must run before forward.")
        overlap = self.emb_ks - self.emb_hs
        batch = 1
        padded = tensor.permute(0, 2, 3, 1)
        padded = F.pad(
            padded,
            (0, 0, overlap, self.padded_frequency - self.static_frequency - overlap, overlap, self.padded_time - self.static_time - overlap),
        )

        intra_input = padded
        intra = self.intra_norm(intra_input).reshape(batch * self.padded_time, self.padded_frequency, self.emb_dim)
        intra = intra.index_select(1, self.intra_indices).reshape(
            batch * self.padded_time, self.intra_segments, self.emb_ks, self.emb_dim
        ).permute(0, 1, 3, 2).reshape(batch * self.padded_time, self.intra_segments, self.emb_ks * self.emb_dim)
        intra, _ = self.intra_rnn(intra)
        intra = self.intra_linear(intra.transpose(1, 2))
        intra = intra.reshape(batch, self.padded_time, self.emb_dim, self.padded_frequency).transpose(-2, -1)
        intra = intra + intra_input

        inter_input = intra.transpose(1, 2)
        inter = self.inter_norm(inter_input).reshape(batch * self.padded_frequency, self.padded_time, self.emb_dim)
        inter = inter.index_select(1, self.inter_indices).reshape(
            batch * self.padded_frequency, self.inter_segments, self.emb_ks, self.emb_dim
        ).permute(0, 1, 3, 2).reshape(batch * self.padded_frequency, self.inter_segments, self.emb_ks * self.emb_dim)
        inter, _ = self.inter_rnn(inter)
        inter = self.inter_linear(inter.transpose(1, 2))
        inter = inter.reshape(batch, self.padded_frequency, self.emb_dim, self.padded_time).transpose(-2, -1)
        inter = inter + inter_input
        inter = inter.permute(0, 3, 2, 1)
        batch_features = inter[..., overlap : overlap + self.static_time, overlap : overlap + self.static_frequency]

        if self.attn_qkv_proj is None:
            query = self.attn_conv_Q(batch_features)
            key = self.attn_conv_K(batch_features)
            value = self.attn_conv_V(batch_features)
            query = self.attn_norm_Q(query).reshape(-1, self.attn_norm_Q.head_features, self.static_time, self.static_frequency)
            key = self.attn_norm_K(key).reshape(-1, self.attn_norm_K.head_features, self.static_time, self.static_frequency)
            value = self.attn_norm_V(value).reshape(-1, self.attn_norm_V.head_features, self.static_time, self.static_frequency)
            query = query.transpose(1, 2).flatten(start_dim=2)
            key = key.transpose(2, 3).reshape(batch * self.n_head, -1, self.static_time)
            value_shape = value.transpose(1, 2).shape
            value = value.transpose(1, 2).flatten(start_dim=2)
        else:
            qkv = self.attn_qkv_proj(batch_features).reshape(
                batch,
                self.n_head,
                sum(self.attn_qkv_head_split_sizes),
                self.static_time,
                self.static_frequency,
            ).permute(0, 1, 3, 4, 2)
            query, key, value = qkv.split(self.attn_qkv_head_split_sizes, dim=4)
            if self.attn_norm_QK is None:
                raise RuntimeError("PostNetGridBlock Q/K normalization must be prepared before export.")
            query, key = self.attn_norm_QK.forward_last_feature(torch.cat((query, key), dim=1)).split(
                self.n_head,
                dim=1,
            )
            query = query.reshape(
                batch * self.n_head, self.static_time, -1
            )
            key = key.permute(0, 1, 3, 4, 2).reshape(
                batch * self.n_head, -1, self.static_time
            )
            value = self.attn_norm_V.forward_last_feature(value).reshape(
                batch * self.n_head, self.static_time, -1
            )
        attention_scores = torch.matmul(query, key)
        if self.attn_qkv_proj is None:
            attention_scores = attention_scores * self.attention_scale
        attention = F.softmax(attention_scores, dim=2)
        value = torch.matmul(attention, value)
        if self.attn_qkv_proj is None:
            value = value.reshape(value_shape).transpose(1, 2)
            value = value.reshape(batch, self.emb_dim, self.static_time, self.static_frequency)
        else:
            value = value.reshape(
                batch,
                self.n_head,
                self.static_time,
                self.static_frequency,
                self.attn_norm_V.head_features,
            ).permute(0, 1, 4, 2, 3).reshape(batch, self.emb_dim, self.static_time, self.static_frequency)
        return self.attn_concat_proj(value) + inter[..., overlap : overlap + self.static_time, overlap : overlap + self.static_frequency]


class UniPASEPostNetCore(nn.Module):
    """State-key-compatible TFGridNet body used by UniPASE PostNet."""

    def __init__(
        self,
        n_srcs: int = 1,
        n_imics: int = 1,
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
        if n_srcs != 1 or n_imics != 1 or n_bands != 3:
            raise ValueError("UniPASE PostNet export supports its published mono, one-source, three-band configuration only.")
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        self.bs = PostNetBandSplit(n_bands=n_bands)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics * n_bands, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList(
            [
                PostNetGridBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    activation=activation,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2 * n_bands, (3, 3), padding=(1, 1))

    def prepare_for_export_(self, time_frames: int) -> None:
        for block in self.blocks:
            block.prepare_for_export_(time_frames, 257)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        tensor = self.bs(spectrum)
        tensor = self.conv(tensor)
        for block in self.blocks:
            tensor = block(tensor)
        return self.bs.inverse(self.deconv(tensor))


class UniPASEPostNetExport(nn.Module):
    """Optional static full-band stage: enhanced 16 kHz waveform -> 48 kHz waveform."""

    def __init__(self, config: Mapping[str, Any], state: dict[str, torch.Tensor]):
        super().__init__()
        if POSTNET_OUT_SAMPLE_RATE != 48000:
            raise ValueError("The first PostNet export supports the source model's fixed 48 kHz output only.")
        self.postnet = UniPASEPostNetCore(**dict(config))
        _load_state(self.postnet, state)
        self.resample_16k_to_48k = FixedSincResample(
            MODEL_SAMPLE_RATE,
            POSTNET_OUT_SAMPLE_RATE,
            EXPORT_SHAPE.base_output_samples,
        )
        self.n_fft = 1536
        self.hop_length = 768
        self.frames = EXPORT_SHAPE.postnet_output_samples // self.hop_length + 1
        self.stft_model = STFT_Process(
            model_type="stft_B",
            n_fft=self.n_fft,
            hop_len=self.hop_length,
            win_length=self.n_fft,
            max_frames=0,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
        ).eval()
        self.istft_model = STFT_Process(
            model_type="istft_B",
            n_fft=self.n_fft,
            hop_len=self.hop_length,
            win_length=self.n_fft,
            max_frames=self.frames,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
            static_norm=True,
        ).eval()
        self.postnet.prepare_for_export_(self.frames)
        frequencies = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32) * (POSTNET_OUT_SAMPLE_RATE / self.n_fft)
        alpha = torch.zeros_like(frequencies)
        alpha[frequencies <= 7200.0] = 1.0
        transition = (frequencies > 7200.0) & (frequencies <= 8000.0)
        alpha[transition] = (8000.0 - frequencies[transition]) / 800.0
        self.register_buffer("residual_gain", (1.0 - alpha).reshape(1, 1, 1, -1), persistent=False)

    def forward(self, enhanced_16k_audio: torch.Tensor) -> torch.Tensor:
        audio_48k = self.resample_16k_to_48k(enhanced_16k_audio)
        waveform = audio_48k.squeeze(1)
        std = torch.std(waveform, dim=1, keepdim=True) + 1e-12
        normalized = (waveform - waveform.mean(dim=1, keepdim=True)) / std
        packed = self.stft_model._stft_B_packed_forward(normalized.unsqueeze(1))
        spectrum = packed.reshape(1, 2, self.n_fft // 2 + 1, self.frames).permute(0, 1, 3, 2)
        residual = self.postnet(spectrum)
        output_spectrum = spectrum + self.residual_gain * residual
        output_packed = output_spectrum.permute(0, 1, 3, 2).reshape(1, self.n_fft + 2, self.frames)
        output = self.istft_model._istft_B_packed_forward(output_packed)
        return output * std.unsqueeze(1)


def _load_export_modules() -> tuple[UniPASEEncoderExport, UniPASEEnhancerExport, UniPASEPostNetExport | None]:
    dewavlm_checkpoint = _load_checkpoint(DEWAVLM_CHECKPOINT, "DeWavLM-Omni.pt")
    adapter_checkpoint = _load_checkpoint(ADAPTER_CHECKPOINT, "Adapter.pt")
    vocoder_checkpoint = _load_checkpoint(VOCODER_CHECKPOINT, "Vocoder_DWO-L1.pt")
    encoder = UniPASEEncoderExport(
        _checkpoint_config(dewavlm_checkpoint, "DeWavLM-Omni.pt"),
        _checkpoint_model_state(dewavlm_checkpoint, "DeWavLM-Omni.pt"),
    ).eval()
    enhancer = UniPASEEnhancerExport(
        _checkpoint_config(adapter_checkpoint, "Adapter.pt"),
        _checkpoint_model_state(adapter_checkpoint, "Adapter.pt"),
        _checkpoint_config(vocoder_checkpoint, "Vocoder_DWO-L1.pt"),
        _checkpoint_model_state(vocoder_checkpoint, "Vocoder_DWO-L1.pt"),
    ).eval()
    postnet = None
    if EXPORT_POSTNET_48K:
        postnet_checkpoint = _load_checkpoint(POSTNET_CHECKPOINT, "PostNet.pt")
        postnet = UniPASEPostNetExport(
            _checkpoint_config(postnet_checkpoint, "PostNet.pt"),
            _checkpoint_model_state(postnet_checkpoint, "PostNet.pt"),
        ).eval()
    return encoder, enhancer, postnet


def main() -> None:
    print("Export start ...")
    encoder, enhancer, postnet = _load_export_modules()
    encoder_path = OUTPUT_DIR / ENCODER_MODEL_NAME
    enhancer_path = OUTPUT_DIR / ENHANCER_MODEL_NAME
    postnet_path = OUTPUT_DIR / POSTNET_MODEL_NAME if postnet is not None else None
    export_audio = _export_audio()
    export_features = torch.ones(
        (1, EXPORT_SHAPE.feature_frames, FEATURE_CHANNELS),
        dtype=torch.float32,
    )
    encoder_dynamic_axes = None
    enhancer_dynamic_axes = None
    if DYNAMIC_AXES:
        encoder_dynamic_axes = {
            "noisy_audio": {2: "audio_samples"},
            "noisy_features_l1": {1: "feature_frames"},
            "reference_features_l24": {1: "feature_frames"},
        }
        enhancer_dynamic_axes = {
            "noisy_features_l1": {1: "feature_frames"},
            "reference_features_l24": {1: "feature_frames"},
            "denoised_audio": {2: "audio_samples"},
        }
    _export_stage(
        encoder,
        (export_audio,),
        encoder_path,
        ["noisy_audio"],
        ["noisy_features_l1", "reference_features_l24"],
        dynamic_axes=encoder_dynamic_axes,
    )
    _export_stage(
        enhancer,
        (export_features, export_features),
        enhancer_path,
        ["noisy_features_l1", "reference_features_l24"],
        ["denoised_audio"],
        dynamic_axes=enhancer_dynamic_axes,
    )

    if postnet is not None and postnet_path is not None:
        _export_stage(
            postnet,
            (torch.ones((1, 1, EXPORT_SHAPE.base_output_samples), dtype=torch.float32),),
            postnet_path,
            ["enhanced_16k_audio"],
            ["denoised_audio"],
        )

    metadata_path = _export_pipeline_metadata(postnet is not None)
    _remove_unrequested_final_artifacts(postnet is not None)
    print(f"Metadata saved to: {metadata_path}")
    print("\nExport done!")
    del encoder, enhancer, postnet, export_audio, export_features
    gc.collect()
    _run_inference_demo(OUTPUT_DIR.expanduser().resolve())


if __name__ == "__main__":
    main()