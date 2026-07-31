"""Weight-only MatMul quantization helpers shared by ONNX optimization scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass
class RefineStats:
    blocks: int = 0
    improved_blocks: int = 0
    seed_error: float = 0.0
    refined_error: float = 0.0

    def add(self, other: "RefineStats") -> None:
        self.blocks += other.blocks
        self.improved_blocks += other.improved_blocks
        self.seed_error += other.seed_error
        self.refined_error += other.refined_error


def _iter_blocks(values: np.ndarray, block_size: int, max_blocks: int):
    rows, columns = values.shape
    blocks_per_row = (columns + block_size - 1) // block_size
    padded_columns = blocks_per_row * block_size
    rows_per_chunk = max(1, max_blocks // blocks_per_row)
    for start in range(0, rows, rows_per_chunk):
        chunk = np.asarray(values[start:start + rows_per_chunk], dtype=np.float32)
        if padded_columns != columns:
            chunk = np.pad(chunk, ((0, 0), (0, padded_columns - columns)))
        yield start, start + chunk.shape[0], chunk.reshape(-1, block_size)


def _weighted_error(weight: np.ndarray, dequantized: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(weight), axis=1, dtype=np.float32))
    importance = np.abs(weight) + rms[:, None]
    return np.sum(np.square(dequantized - weight) * importance, axis=1, dtype=np.float32)


def _quantize_with_params(
    weight: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray,
    maxq: int,
) -> tuple[np.ndarray, np.ndarray]:
    safe_scale = np.maximum(scale, np.finfo(np.float32).tiny)
    codes = np.rint(weight / safe_scale[:, None] + zero_point[:, None])
    codes = np.clip(codes, 0, maxq).astype(np.uint8)
    dequantized = (codes.astype(np.float32) - zero_point[:, None]) * safe_scale[:, None]
    return codes, dequantized


def quant_tensor_k_quant_cpu(
    weight: np.ndarray,
    bits: int = 4,
    block_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic magnitude-weighted affine seed for row blocks."""
    if bits not in (4, 8):
        raise ValueError(f"k-quant helper supports 4 or 8 bits, got {bits}.")
    values = np.asarray(weight, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != block_size:
        raise ValueError(f"expected [blocks, {block_size}] weights, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("quantization refuses weights containing NaN or Inf.")

    maxq = (1 << bits) - 1
    minimum = np.min(values, axis=1)
    maximum = np.max(values, axis=1)
    scale = np.maximum((maximum - minimum) / maxq, np.finfo(np.float32).tiny)
    zero_point = np.clip(np.rint(-minimum / scale), 0, maxq).astype(np.uint8)
    best_codes, best_dequantized = _quantize_with_params(values, scale, zero_point, maxq)
    best_error = _weighted_error(values, best_dequantized)

    # Match Qwen-v3's k-quant seed shape: search clipped affine ranges, then
    # retain the candidate minimizing magnitude-weighted reconstruction error.
    for ratio in np.linspace(1.0, 0.55, 10, dtype=np.float32):
        center = (minimum + maximum) * np.float32(0.5)
        half_span = (maximum - minimum) * np.float32(0.5) * ratio
        candidate_min = center - half_span
        candidate_max = center + half_span
        candidate_scale = np.maximum(
            (candidate_max - candidate_min) / maxq, np.finfo(np.float32).tiny
        )
        candidate_zp = np.clip(
            np.rint(-candidate_min / candidate_scale), 0, maxq
        ).astype(np.uint8)
        candidate_codes, candidate_dequantized = _quantize_with_params(
            values, candidate_scale, candidate_zp, maxq
        )
        candidate_error = _weighted_error(values, candidate_dequantized)
        improved = candidate_error < best_error
        best_codes[improved] = candidate_codes[improved]
        scale[improved] = candidate_scale[improved]
        zero_point[improved] = candidate_zp[improved]
        best_error[improved] = candidate_error[improved]
    return best_codes, scale.astype(np.float32), zero_point


def affine_refine_v2_rows(
    values: np.ndarray,
    block_size: int,
    bits: int,
    symmetric: bool = False,
    iterations: int = 6,
    weighted_tolerance: float = 0.15,
    max_blocks_per_chunk: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RefineStats]:
    """Refine a weighted seed for minimum plain block MSE."""
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype.kind != "f":
        raise ValueError(f"AFFINE_REFINE_V2 expects a 2-D float matrix, got {values.shape}.")
    if block_size < 16 or block_size > 256 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two in [16, 256].")
    if bits not in (4, 8):
        raise ValueError("AFFINE_REFINE_V2 supports Q4 and Q8.")
    if iterations < 1 or weighted_tolerance < 0 or max_blocks_per_chunk < 1:
        raise ValueError("invalid AFFINE_REFINE_V2 refinement settings.")

    rows, columns = values.shape
    blocks_per_row = (columns + block_size - 1) // block_size
    output_codes = np.empty((rows * blocks_per_row, block_size), dtype=np.uint8)
    output_scales = np.empty(rows * blocks_per_row, dtype=np.float32)
    output_zero_points = np.empty(rows * blocks_per_row, dtype=np.uint8)
    stats = RefineStats(blocks=rows * blocks_per_row)
    maxq = (1 << bits) - 1
    midpoint = 1 << (bits - 1)

    output_offset = 0
    for _, _, weight in _iter_blocks(values, block_size, max_blocks_per_chunk):
        seed_codes, seed_scale, seed_zp = quant_tensor_k_quant_cpu(weight, bits, block_size)
        if symmetric:
            seed_zp.fill(midpoint)
            max_abs = np.max(np.abs(weight), axis=1)
            seed_scale = np.maximum(max_abs / (midpoint - 1), np.finfo(np.float32).tiny)
            seed_codes, seed_dequantized = _quantize_with_params(weight, seed_scale, seed_zp, maxq)
        else:
            _, seed_dequantized = _quantize_with_params(weight, seed_scale, seed_zp, maxq)

        seed_weighted = _weighted_error(weight, seed_dequantized)
        best_codes = seed_codes.copy()
        best_scale = seed_scale.copy()
        best_zp = seed_zp.copy()
        best_mse = np.sum(np.square(seed_dequantized - weight), axis=1, dtype=np.float32)

        zero_point_candidates = (midpoint,) if symmetric else range(maxq + 1)
        for zero_point in zero_point_candidates:
            zp = np.full(weight.shape[0], zero_point, dtype=np.uint8)
            for ratio in (1.0, 0.94, 0.82, 0.70, 0.55):
                scale = np.maximum(
                    np.max(np.abs(weight), axis=1) * np.float32(ratio) / max(1, midpoint - 1),
                    np.finfo(np.float32).tiny,
                )
                codes = None
                dequantized = None
                for _ in range(iterations):
                    codes, dequantized = _quantize_with_params(weight, scale, zp, maxq)
                    centered = codes.astype(np.float32) - zp[:, None]
                    denominator = np.sum(np.square(centered), axis=1, dtype=np.float32)
                    numerator = np.sum(weight * centered, axis=1, dtype=np.float32)
                    scale = np.maximum(
                        np.divide(numerator, denominator, out=scale.copy(), where=denominator > 0),
                        np.finfo(np.float32).tiny,
                    )
                codes, dequantized = _quantize_with_params(weight, scale, zp, maxq)
                mse = np.sum(np.square(dequantized - weight), axis=1, dtype=np.float32)
                weighted = _weighted_error(weight, dequantized)
                improved = (mse < best_mse) & (
                    weighted <= seed_weighted * np.float32(1.0 + weighted_tolerance)
                )
                best_codes[improved] = codes[improved]
                best_scale[improved] = scale[improved]
                best_zp[improved] = zp[improved]
                best_mse[improved] = mse[improved]

        count = weight.shape[0]
        output_codes[output_offset:output_offset + count] = best_codes
        output_scales[output_offset:output_offset + count] = best_scale
        output_zero_points[output_offset:output_offset + count] = best_zp
        stats.improved_blocks += int(np.count_nonzero(np.any(best_codes != seed_codes, axis=1)))
        stats.seed_error += float(np.sum(np.square(seed_dequantized - weight), dtype=np.float64))
        final = (best_codes.astype(np.float32) - best_zp[:, None]) * best_scale[:, None]
        stats.refined_error += float(np.sum(np.square(final - weight), dtype=np.float64))
        output_offset += count

    return (
        output_codes.reshape(rows, blocks_per_row, block_size),
        output_scales.reshape(rows, blocks_per_row),
        output_zero_points.reshape(rows, blocks_per_row),
        stats,
    )


def _pack_codes(values: np.ndarray, bits: int, pad_value: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if bits == 8:
        return values
    if values.shape[-1] & 1:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)], constant_values=pad_value)
    return (values[..., 0::2] | (values[..., 1::2] << 4)).astype(np.uint8)


def _make_name_factory(graph, prefix: str) -> Callable[[str], str]:
    used = {value.name for value in graph.initializer}
    used.update(node.name for node in graph.node if node.name)
    counter = 0

    def make(suffix: str) -> str:
        nonlocal counter
        while True:
            name = f"{prefix}{counter}_{suffix}"
            counter += 1
            if name not in used:
                used.add(name)
                return name

    return make


def quantize_matmul_model(
    model: onnx.ModelProto,
    *,
    bits: int,
    block_size: int,
    algorithm: str,
    symmetric: bool,
    accuracy_level: int,
    nodes_to_include: set[str] | None = None,
    nodes_to_exclude: set[str] | None = None,
) -> RefineStats:
    """Rewrite selected constant 2-D MatMuls as MatMulNBits nodes."""
    total = RefineStats()
    rewritten = 0

    def rewrite_graph(graph) -> None:
        nonlocal rewritten
        initializers = {tensor.name: tensor for tensor in graph.initializer}
        make_name = _make_name_factory(graph, f"{algorithm.lower()}_q{bits}_")
        replacements = []
        removed_initializers = set()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)
            selected = (
                node.op_type == "MatMul"
                and len(node.input) >= 2
                and node.input[1] in initializers
                and (not nodes_to_include or node.name in nodes_to_include)
                and (not nodes_to_exclude or node.name not in nodes_to_exclude)
            )
            if not selected:
                replacements.append(node)
                continue
            weight_tensor = initializers[node.input[1]]
            weight = numpy_helper.to_array(weight_tensor)
            if weight.ndim != 2 or weight.dtype.kind != "f":
                replacements.append(node)
                continue
            input_features, output_features = weight.shape
            if algorithm == "AFFINE_REFINE_V2":
                codes, scales, zero_points, stats = affine_refine_v2_rows(
                    weight.T, block_size, bits, symmetric
                )
                total.add(stats)
            elif algorithm in ("k_quant", "RTN"):
                block_count = (input_features + block_size - 1) // block_size
                padded = np.pad(weight.T, ((0, 0), (0, block_count * block_size - input_features)))
                blocks = padded.reshape(-1, block_size)
                if algorithm == "k_quant":
                    flat_codes, flat_scales, flat_zp = quant_tensor_k_quant_cpu(
                        blocks, bits, block_size
                    )
                else:
                    maxq = (1 << bits) - 1
                    if symmetric:
                        midpoint = 1 << (bits - 1)
                        flat_zp = np.full(blocks.shape[0], midpoint, dtype=np.uint8)
                        flat_scales = np.maximum(
                            np.max(np.abs(blocks), axis=1) / max(1, midpoint - 1),
                            np.finfo(np.float32).tiny,
                        )
                    else:
                        minimum = np.min(blocks, axis=1)
                        maximum = np.max(blocks, axis=1)
                        flat_scales = np.maximum(
                            (maximum - minimum) / maxq, np.finfo(np.float32).tiny
                        )
                        flat_zp = np.clip(
                            np.rint(-minimum / flat_scales), 0, maxq
                        ).astype(np.uint8)
                    flat_codes, _ = _quantize_with_params(
                        blocks, flat_scales, flat_zp, maxq
                    )
                codes = flat_codes.reshape(output_features, block_count, block_size)
                scales = flat_scales.reshape(output_features, block_count)
                zero_points = flat_zp.reshape(output_features, block_count)
            else:
                raise ValueError(f"unsupported helper algorithm {algorithm!r}.")

            weight_name = make_name("weight")
            scale_name = make_name("scales")
            zero_point_name = make_name("zero_points")
            graph.initializer.extend([
                numpy_helper.from_array(_pack_codes(codes, bits), name=weight_name),
                numpy_helper.from_array(scales.astype(weight.dtype, copy=False), name=scale_name),
                numpy_helper.from_array(
                    _pack_codes(zero_points, bits, pad_value=1 << (bits - 1)),
                    name=zero_point_name,
                ),
            ])
            attributes = {
                "K": input_features,
                "N": output_features,
                "bits": bits,
                "block_size": block_size,
            }
            if accuracy_level:
                attributes["accuracy_level"] = accuracy_level
            replacements.append(helper.make_node(
                "MatMulNBits",
                [node.input[0], weight_name, scale_name, zero_point_name],
                list(node.output),
                name=f"{node.name}_{algorithm}_Q{bits}" if node.name else make_name("matmul"),
                domain="com.microsoft",
                **attributes,
            ))
            removed_initializers.add(weight_tensor.name)
            rewritten += 1

        if removed_initializers:
            used = {name for node in replacements for name in node.input}
            kept = [
                tensor for tensor in graph.initializer
                if tensor.name not in removed_initializers or tensor.name in used
            ]
            graph.ClearField("initializer")
            graph.initializer.extend(kept)
            graph.ClearField("node")
            graph.node.extend(replacements)

    rewrite_graph(model.graph)
    if rewritten and not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  {algorithm}: {rewritten} MatMul -> MatMulNBits; "
        f"refined {total.improved_blocks}/{total.blocks} blocks, MSE ratio={ratio:.6f}."
    )
    return total