"""Weight-only MatMul quantization helpers shared by ONNX optimization scripts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _numba_refine_kernel():
    """Build the optional parallel refinement kernel lazily."""
    try:
        from numba import njit, prange
    except ImportError:
        return None

    @njit(parallel=True, nogil=True, cache=True)
    def refine(
        weight,
        quantized,
        scales,
        zero_points,
        clip_ratios,
        iterations,
        tolerance,
        tiny,
        maxq,
        midpoint,
        sweep_limit,
        symmetric,
    ):
        block_count, width = weight.shape
        baseline_errors = np.empty(block_count, dtype=np.float32)
        refined_errors = np.empty(block_count, dtype=np.float32)
        improved = np.zeros(block_count, dtype=np.bool_)

        for block_index in prange(block_count):
            rms_sum = np.float32(0.0)
            positive_max = np.float32(0.0)
            negative_max = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                rms_sum += value * value
                positive_max = max(positive_max, value)
                negative_max = max(negative_max, -value)
            rms = np.float32(np.sqrt(rms_sum / np.float32(width)))
            seed_scale = np.float32(scales[block_index])
            seed_zp_int = int(zero_points[block_index])
            seed_zp = np.float32(seed_zp_int)
            baseline_plain = np.float32(0.0)
            baseline_weighted = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                centered = np.float32(quantized[block_index, column]) - seed_zp
                residual = value - seed_scale * centered
                squared = residual * residual
                baseline_plain += squared
                baseline_weighted += (rms + np.abs(value)) * squared

            local_plain = baseline_plain
            weighted_bound = tolerance * baseline_weighted
            if symmetric:
                zp_lo = midpoint
                zp_hi = midpoint
            elif maxq + 1 <= sweep_limit:
                zp_lo = 0
                zp_hi = maxq
            else:
                zp_lo = seed_zp_int - sweep_limit // 2
                zp_lo = max(0, min(zp_lo, maxq - sweep_limit + 1))
                zp_hi = zp_lo + sweep_limit - 1

            candidate_codes = np.empty(width, dtype=np.uint8)
            for zero_point_int in range(zp_lo, zp_hi + 1):
                zero_point = np.float32(zero_point_int)
                positive_scale = (
                    positive_max / np.float32(maxq - zero_point_int)
                    if zero_point_int < maxq else np.float32(0.0)
                )
                negative_scale = (
                    negative_max / np.float32(zero_point_int)
                    if zero_point_int > 0 else np.float32(0.0)
                )
                coverage_scale = max(positive_scale, negative_scale)
                if coverage_scale <= tiny:
                    coverage_scale = np.float32(1.0)

                for start_index in range(clip_ratios.size + 1):
                    candidate_scale = (
                        seed_scale if start_index == 0
                        else coverage_scale * clip_ratios[start_index - 1]
                    )
                    for _ in range(iterations):
                        denominator = np.float32(0.0)
                        numerator = np.float32(0.0)
                        for column in range(width):
                            value = np.float32(weight[block_index, column])
                            code = np.rint(value / candidate_scale + zero_point)
                            code = min(np.float32(maxq), max(np.float32(0.0), code))
                            centered = code - zero_point
                            denominator += centered * centered
                            numerator += centered * value
                        if denominator <= tiny:
                            break
                        fitted_scale = numerator / denominator
                        if not np.isfinite(fitted_scale) or fitted_scale <= tiny:
                            break
                        if fitted_scale == candidate_scale:
                            break
                        candidate_scale = fitted_scale

                    candidate_plain = np.float32(0.0)
                    candidate_weighted = np.float32(0.0)
                    for column in range(width):
                        value = np.float32(weight[block_index, column])
                        code = np.rint(value / candidate_scale + zero_point)
                        code = min(np.float32(maxq), max(np.float32(0.0), code))
                        candidate_codes[column] = np.uint8(code)
                        residual = value - candidate_scale * (code - zero_point)
                        squared = residual * residual
                        candidate_plain += squared
                        candidate_weighted += (rms + np.abs(value)) * squared
                    if candidate_plain < local_plain and candidate_weighted <= weighted_bound:
                        local_plain = candidate_plain
                        scales[block_index] = candidate_scale
                        zero_points[block_index] = np.uint8(zero_point_int)
                        for column in range(width):
                            quantized[block_index, column] = candidate_codes[column]

            baseline_errors[block_index] = baseline_plain
            refined_errors[block_index] = local_plain
            improved[block_index] = local_plain < baseline_plain
        return baseline_errors, refined_errors, improved

    return refine


def affine_refine_v2_rows(
    values: np.ndarray,
    block_size: int,
    bits: int,
    symmetric: bool = False,
    iterations: int = 6,
    weighted_tolerance: float = 0.15,
    max_blocks_per_chunk: int = 32768,
    asymmetric_zero_point_sweep_limit: int = 32,
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
    if asymmetric_zero_point_sweep_limit < 16:
        raise ValueError("asymmetric_zero_point_sweep_limit must be at least 16.")

    rows, columns = values.shape
    blocks_per_row = (columns + block_size - 1) // block_size
    output_codes = np.empty((rows * blocks_per_row, block_size), dtype=np.uint8)
    output_scales = np.empty(rows * blocks_per_row, dtype=np.float32)
    output_zero_points = np.empty(rows * blocks_per_row, dtype=np.uint8)
    stats = RefineStats(blocks=rows * blocks_per_row)
    maxq = (1 << bits) - 1
    midpoint = 1 << (bits - 1)

    output_offset = 0
    numba_kernel = _numba_refine_kernel()
    clip_ratios = np.asarray((1.0, 0.94, 0.82, 0.70, 0.55), dtype=np.float32)
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
        max_abs = np.max(np.abs(weight), axis=1)

        if numba_kernel is not None:
            baseline, refined, improved = numba_kernel(
                weight,
                best_codes,
                best_scale,
                best_zp,
                clip_ratios,
                iterations,
                np.float32(1.0 + weighted_tolerance),
                np.float32(np.finfo(np.float32).tiny),
                maxq,
                midpoint,
                asymmetric_zero_point_sweep_limit,
                symmetric,
            )
            count = weight.shape[0]
            output_codes[output_offset:output_offset + count] = best_codes
            output_scales[output_offset:output_offset + count] = best_scale
            output_zero_points[output_offset:output_offset + count] = best_zp
            stats.improved_blocks += int(np.count_nonzero(improved))
            stats.seed_error += float(baseline.sum(dtype=np.float64))
            stats.refined_error += float(refined.sum(dtype=np.float64))
            output_offset += count
            continue

        if symmetric:
            zero_point_candidates = [np.full(weight.shape[0], midpoint, dtype=np.uint8)]
        elif maxq + 1 <= asymmetric_zero_point_sweep_limit:
            zero_point_candidates = (
                np.full(weight.shape[0], zero_point, dtype=np.uint8)
                for zero_point in range(maxq + 1)
            )
        else:
            # Q8 has 256 possible zero points. Match Qwen-v3 by searching a
            # bounded window centered independently on each block's k-quant seed.
            half_window = asymmetric_zero_point_sweep_limit // 2
            window_start = np.clip(
                seed_zp.astype(np.int16) - half_window,
                0,
                maxq - asymmetric_zero_point_sweep_limit + 1,
            )
            zero_point_candidates = (
                (window_start + offset).astype(np.uint8)
                for offset in range(asymmetric_zero_point_sweep_limit)
            )
        for zp in zero_point_candidates:
            for ratio in (1.0, 0.94, 0.82, 0.70, 0.55):
                scale = np.maximum(
                    max_abs * np.float32(ratio) / max(1, midpoint - 1),
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
    op_types: tuple[str, ...] = ("MatMul",),
    nodes_to_include: set[str] | None = None,
    nodes_to_exclude: set[str] | None = None,
) -> RefineStats:
    """Rewrite selected constant MatMul/Gemm weights as MatMulNBits nodes."""
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
                node.op_type in op_types
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
            if weight.ndim not in (2, 3) or weight.dtype.kind != "f":
                replacements.append(node)
                continue
            is_gemm = node.op_type == "Gemm"
            attributes_by_name = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            if is_gemm:
                trans_a = int(attributes_by_name.get("transA", 0))
                trans_b = int(attributes_by_name.get("transB", 0))
                alpha = float(attributes_by_name.get("alpha", 1.0))
                beta = float(attributes_by_name.get("beta", 1.0))
                if weight.ndim != 2 or trans_a or alpha != 1.0 or beta != 1.0:
                    replacements.append(node)
                    continue
                logical_weight = weight.T if trans_b else weight
            else:
                logical_weight = weight
            input_features, output_features = logical_weight.shape[-2:]
            if algorithm == "AFFINE_REFINE_V2":
                codes, scales, zero_points, stats = affine_refine_v2_rows(
                    logical_weight.swapaxes(-1, -2).reshape(-1, input_features),
                    block_size,
                    bits,
                    symmetric,
                )
                if logical_weight.ndim == 3:
                    batch = logical_weight.shape[0]
                    blocks = codes.shape[-2]
                    codes = codes.reshape(batch, output_features, blocks, block_size)
                    scales = scales.reshape(batch, output_features, blocks)
                    zero_points = zero_points.reshape(batch, output_features, blocks)
                total.add(stats)
            elif algorithm in ("k_quant", "RTN"):
                block_count = (input_features + block_size - 1) // block_size
                rows = logical_weight.swapaxes(-1, -2).reshape(-1, input_features)
                padded = np.pad(rows, ((0, 0), (0, block_count * block_size - input_features)))
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
                leading_shape = (*logical_weight.shape[:-2], output_features)
                codes = flat_codes.reshape(*leading_shape, block_count, block_size)
                scales = flat_scales.reshape(*leading_shape, block_count)
                zero_points = flat_zp.reshape(*leading_shape, block_count)
            else:
                raise ValueError(f"unsupported helper algorithm {algorithm!r}.")

            attributes = {
                "K": input_features,
                "N": output_features,
                "bits": bits,
                "block_size": block_size,
            }
            if accuracy_level:
                attributes["accuracy_level"] = accuracy_level

            def append_nbits(weight_codes, weight_scales, weight_zero_points, data_name, output_name, suffix):
                weight_name = make_name(f"weight_{suffix}")
                scale_name = make_name(f"scales_{suffix}")
                zero_point_name = make_name(f"zero_points_{suffix}")
                graph.initializer.extend([
                    numpy_helper.from_array(_pack_codes(weight_codes, bits), name=weight_name),
                    numpy_helper.from_array(
                        weight_scales.astype(weight.dtype, copy=False), name=scale_name
                    ),
                    numpy_helper.from_array(
                        _pack_codes(weight_zero_points, bits, pad_value=1 << (bits - 1)),
                        name=zero_point_name,
                    ),
                ])
                replacements.append(helper.make_node(
                    "MatMulNBits",
                    [data_name, weight_name, scale_name, zero_point_name],
                    [output_name],
                    name=(f"{node.name}_{algorithm}_Q{bits}_{suffix}" if node.name else make_name("matmul")),
                    domain="com.microsoft",
                    **attributes,
                ))

            if logical_weight.ndim == 3:
                batch = logical_weight.shape[0]
                split_outputs = [make_name(f"batch_input_{index}") for index in range(batch)]
                replacements.append(helper.make_node(
                    "Split",
                    [node.input[0]],
                    split_outputs,
                    name=make_name("batch_split"),
                    axis=0,
                    num_outputs=batch,
                ))
                batch_outputs = []
                for index, split_output in enumerate(split_outputs):
                    batch_output = make_name(f"batch_output_{index}")
                    append_nbits(
                        codes[index], scales[index], zero_points[index],
                        split_output, batch_output, f"batch_{index}",
                    )
                    batch_outputs.append(batch_output)
                replacements.append(helper.make_node(
                    "Concat",
                    batch_outputs,
                    list(node.output),
                    name=make_name("batch_concat"),
                    axis=0,
                ))
            elif is_gemm:
                matmul_output = make_name("gemm_matmul_output")
                append_nbits(codes, scales, zero_points, node.input[0], matmul_output, "gemm")
                if len(node.input) >= 3 and node.input[2]:
                    replacements.append(helper.make_node(
                        "Add",
                        [matmul_output, node.input[2]],
                        list(node.output),
                        name=make_name("gemm_bias_add"),
                    ))
                else:
                    replacements[-1].output[0] = node.output[0]
            else:
                append_nbits(codes, scales, zero_points, node.input[0], node.output[0], "matmul")
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
        f"  {algorithm}: {rewritten} MatMul/Gemm weights -> MatMulNBits; "
        f"refined {total.improved_blocks}/{total.blocks} blocks, MSE ratio={ratio:.6f}."
    )
    return total