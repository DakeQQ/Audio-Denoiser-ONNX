#!/usr/bin/env python
"""Rewrite two validated ZipEnhancer padding exporter limitations.

PyTorch Conv2d cannot represent the DenseBlockV2 padding contract
(top=dilation, bottom=0, left=right=1). The export source therefore emits an
exact symmetric Conv followed by a tail Slice. This narrow rewrite changes only
those eight validated regions. The legacy exporter also expands the fixed
ReflectionPad1d pads vector into a 15-node constant-construction chain; this
rewrite replaces that private chain with one int64 initializer. A separate
deployment model is written and the raw export is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_STANDARD_DOMAINS = ("", "ai.onnx")
_EXPECTED_PATTERNS = Counter(
    {
        (1, 1): 1,
        (1, 2): 1,
        (1, 4): 1,
        (1, 8): 1,
        (2, 1): 1,
        (2, 2): 1,
        (2, 4): 1,
        (2, 8): 1,
    }
)
_EXPECTED_REFLECT_PADS = [0, 0, 200, 0, 0, 200]


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _replace_ints_attribute(
    node: onnx.NodeProto, name: str, values: list[int]
) -> None:
    matches = [attribute for attribute in node.attribute if attribute.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {name!r} attribute on {node.name!r}, got {len(matches)}."
        )
    del matches[0].ints[:]
    matches[0].ints.extend(values)


def _constant_tensor(
    name: str,
    initializers: dict[str, onnx.TensorProto],
    producers: dict[str, onnx.NodeProto],
) -> tuple[onnx.TensorProto, onnx.NodeProto | None]:
    initializer = initializers.get(name)
    if initializer is not None:
        return initializer, None

    producer = producers.get(name)
    if (
        producer is None
        or producer.domain not in _STANDARD_DOMAINS
        or producer.op_type != "Constant"
    ):
        raise ValueError(f"Tensor {name!r} is not a Constant or initializer.")
    attrs = _attributes(producer)
    value = attrs.get("value")
    if set(attrs) != {"value"} or not isinstance(value, onnx.TensorProto):
        raise ValueError(f"Constant producer for {name!r} is not a tensor Constant.")
    return value, producer


def _constant_array(
    name: str,
    initializers: dict[str, onnx.TensorProto],
    producers: dict[str, onnx.NodeProto],
) -> tuple[np.ndarray, onnx.NodeProto | None]:
    tensor, producer = _constant_tensor(name, initializers, producers)
    return numpy_helper.to_array(tensor), producer


def _interface_signature(model: onnx.ModelProto) -> tuple[bytes, bytes]:
    inputs = onnx.GraphProto()
    outputs = onnx.GraphProto()
    inputs.input.extend(model.graph.input)
    outputs.output.extend(model.graph.output)
    return inputs.SerializeToString(), outputs.SerializeToString()


def _rank_type_shape(
    model: onnx.ModelProto,
) -> dict[str, tuple[int | None, int | None, tuple[int | None, ...] | None]]:
    result: dict[
        str, tuple[int | None, int | None, tuple[int | None, ...] | None]
    ] = {}
    for value in (
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            result[value.name] = (None, tensor_type.elem_type or None, None)
            continue
        shape = tuple(
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in tensor_type.shape.dim
        )
        result[value.name] = (
            len(shape),
            tensor_type.elem_type or None,
            shape,
        )
    for initializer in model.graph.initializer:
        shape = tuple(int(dimension) for dimension in initializer.dims)
        result[initializer.name] = (
            len(shape), initializer.data_type, shape
        )
    return result


def _already_rewritten_patterns(model: onnx.ModelProto) -> Counter[tuple[int, int]]:
    patterns: Counter[tuple[int, int]] = Counter()
    for node in model.graph.node:
        if node.domain not in _STANDARD_DOMAINS or node.op_type != "Conv":
            continue
        attrs = _attributes(node)
        dilations = list(attrs.get("dilations", []))
        pads = list(attrs.get("pads", []))
        group = int(attrs.get("group", 1))
        if (
            len(dilations) == 2
            and dilations[1] == 1
            and pads == [dilations[0], 1, 0, 1]
            and list(attrs.get("kernel_shape", [])) == [2, 3]
            and list(attrs.get("strides", [])) == [1, 1]
        ):
            patterns[(group, int(dilations[0]))] += 1
    return patterns


def rewrite_asymmetric_causal_convs(
    source_model_path: str | os.PathLike[str],
    final_model_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Rewrite exactly eight causal Conv+Slice pairs and one reflect-pad chain."""

    source_path = Path(source_model_path).expanduser().resolve()
    final_path = Path(final_model_path).expanduser().resolve()
    if source_path == final_path:
        raise ValueError("Source and final ONNX paths must be different.")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    unloaded_model = onnx.load(str(source_path), load_external_data=False)
    external_initializers = [
        initializer.name
        for initializer in unloaded_model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL
        or len(initializer.external_data) > 0
    ]
    if external_initializers:
        raise ValueError(
            "External-data ONNX models are not supported by this rewrite; refusing "
            f"to change serialization for {external_initializers[:4]}."
        )

    model = onnx.load(str(source_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    interface_before = _interface_signature(model)
    opsets_before = [(entry.domain, entry.version) for entry in model.opset_import]
    metadata_before = [(item.key, item.value) for item in model.metadata_props]
    initializer_names_before = [item.name for item in model.graph.initializer]
    nodes_before = len(model.graph.node)

    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=False
    )
    rank_type_shape = _rank_type_shape(inferred)
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    graph_outputs = {output.name for output in model.graph.output}

    metadata = {item.key: item.value for item in model.metadata_props}
    expected_metadata = {
        "center_pad": "1",
        "dynamic_axes": "0",
        "nfft": "400",
        "pad_mode": "reflect",
        "use_batch_fold": "1",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Required metadata {key!r}={expected!r}, got {metadata.get(key)!r}."
            )
    export_audio_length = int(metadata.get("export_audio_length", "0"))
    fold_window_length = int(metadata.get("fold_window_length", "0"))
    if (
        export_audio_length <= 0
        or fold_window_length <= 0
        or export_audio_length % fold_window_length != 0
    ):
        raise ValueError("Export metadata does not describe whole static fold windows.")
    expected_pad_input_shape = (
        export_audio_length // fold_window_length,
        1,
        fold_window_length,
    )

    def require_private_producer(
        tensor_name: str,
        op_type: str,
        consumer: onnx.NodeProto,
    ) -> onnx.NodeProto:
        producer = producers.get(tensor_name)
        if (
            producer is None
            or producer.domain not in _STANDARD_DOMAINS
            or producer.op_type != op_type
            or len(producer.output) != 1
        ):
            raise ValueError(
                f"Tensor {tensor_name!r} must have one private {op_type} producer."
            )
        tensor_consumers = consumers[tensor_name]
        if (
            len(tensor_consumers) != 1
            or tensor_consumers[0] is not consumer
            or tensor_name in graph_outputs
        ):
            raise ValueError(f"Tensor {tensor_name!r} is not private to {consumer.name!r}.")
        return producer

    def require_private_constant(
        tensor_name: str,
        expected: list[int],
        consumer: onnx.NodeProto,
    ) -> onnx.NodeProto:
        array, producer = _constant_array(tensor_name, initializers, producers)
        if (
            producer is None
            or array.dtype != np.int64
            or array.shape != (len(expected),)
            or array.tolist() != expected
        ):
            raise ValueError(
                f"Tensor {tensor_name!r} must be a private int64 Constant "
                f"with value {expected}."
            )
        return require_private_producer(tensor_name, "Constant", consumer)

    reflect_matches: list[tuple[onnx.NodeProto, list[onnx.NodeProto]]] = []
    already_rewritten_reflect = 0
    for pad_node in model.graph.node:
        if pad_node.domain not in _STANDARD_DOMAINS or pad_node.op_type != "Pad":
            continue
        pad_attrs = _attributes(pad_node)
        if pad_attrs.get("mode", b"constant") != b"reflect":
            continue
        if set(pad_attrs) != {"mode"} or len(pad_node.input) != 2 or len(pad_node.output) != 1:
            raise ValueError(f"Reflect Pad {pad_node.name!r} has an unexpected schema.")
        input_rank, input_type, input_shape = rank_type_shape.get(
            pad_node.input[0], (None, None, None)
        )
        output_rank, output_type, _ = rank_type_shape.get(
            pad_node.output[0], (None, None, None)
        )
        if (
            input_rank != 3
            or output_rank != 3
            or input_type != TensorProto.FLOAT
            or output_type != TensorProto.FLOAT
            or input_shape != expected_pad_input_shape
        ):
            raise ValueError(
                f"Reflect Pad {pad_node.name!r} must map float32 "
                f"{expected_pad_input_shape} rank-3 data."
            )

        existing_pads = initializers.get(pad_node.input[1])
        if existing_pads is not None:
            array = numpy_helper.to_array(existing_pads)
            if (
                array.dtype != np.int64
                or array.shape != (6,)
                or array.tolist() != _EXPECTED_REFLECT_PADS
            ):
                raise ValueError(f"Reflect Pad {pad_node.name!r} has invalid initializer pads.")
            already_rewritten_reflect += 1
            continue

        chain: list[onnx.NodeProto] = []
        cast = require_private_producer(pad_node.input[1], "Cast", pad_node)
        chain.append(cast)
        if _attributes(cast) != {"to": TensorProto.INT64} or len(cast.input) != 1:
            raise ValueError(f"Pad-control Cast {cast.name!r} is not an int64 cast.")
        control_rank, control_type, control_shape = rank_type_shape.get(
            cast.output[0], (None, None, None)
        )
        if (control_rank, control_type, control_shape) != (1, TensorProto.INT64, (6,)):
            raise ValueError("Reflect Pad control must infer as int64[6].")

        final_reshape = require_private_producer(cast.input[0], "Reshape", cast)
        chain.append(final_reshape)
        if _attributes(final_reshape) != {"allowzero": 0} or len(final_reshape.input) != 2:
            raise ValueError("Final pad-control Reshape has unexpected attributes.")
        chain.append(require_private_constant(final_reshape.input[1], [-1], final_reshape))

        transpose = require_private_producer(final_reshape.input[0], "Transpose", final_reshape)
        chain.append(transpose)
        if _attributes(transpose) != {"perm": [1, 0]} or len(transpose.input) != 1:
            raise ValueError("Pad-control Transpose must use perm=[1,0].")

        reverse = require_private_producer(transpose.input[0], "Slice", transpose)
        chain.append(reverse)
        if _attributes(reverse) or len(reverse.input) != 5:
            raise ValueError("Pad-control reverse Slice has unexpected attributes.")
        reverse_controls = ([-1], [-9223372036854775807], [0], [-1])
        for tensor_name, expected in zip(reverse.input[1:], reverse_controls):
            chain.append(require_private_constant(tensor_name, expected, reverse))

        first_reshape = require_private_producer(reverse.input[0], "Reshape", reverse)
        chain.append(first_reshape)
        if _attributes(first_reshape) != {"allowzero": 0} or len(first_reshape.input) != 2:
            raise ValueError("Initial pad-control Reshape has unexpected attributes.")
        chain.append(require_private_constant(first_reshape.input[1], [-1, 2], first_reshape))

        concat = require_private_producer(first_reshape.input[0], "Concat", first_reshape)
        chain.append(concat)
        if _attributes(concat) != {"axis": 0} or len(concat.input) != 2:
            raise ValueError("Pad-control Concat must concatenate two tensors on axis zero.")
        chain.append(require_private_constant(concat.input[0], [200, 200], concat))

        zero_fill = require_private_producer(concat.input[1], "ConstantOfShape", concat)
        chain.append(zero_fill)
        zero_attrs = _attributes(zero_fill)
        zero_value = zero_attrs.get("value")
        if (
            set(zero_attrs) != {"value"}
            or not isinstance(zero_value, onnx.TensorProto)
            or numpy_helper.to_array(zero_value).dtype != np.int64
            or numpy_helper.to_array(zero_value).shape != (1,)
            or numpy_helper.to_array(zero_value).tolist() != [0]
            or len(zero_fill.input) != 1
        ):
            raise ValueError("Pad-control ConstantOfShape must create int64 zeros.")
        chain.append(require_private_constant(zero_fill.input[0], [4], zero_fill))

        if len({id(node) for node in chain}) != 15:
            raise ValueError("Reflect-pad control chain must contain 15 distinct private nodes.")
        reflect_matches.append((pad_node, chain))

    matches: list[
        tuple[onnx.NodeProto, onnx.NodeProto, int, int, list[onnx.NodeProto]]
    ] = []
    pattern_counts: Counter[tuple[int, int]] = Counter()

    for slice_node in model.graph.node:
        if (
            slice_node.domain not in _STANDARD_DOMAINS
            or slice_node.op_type != "Slice"
            or len(slice_node.input) != 5
            or len(slice_node.output) != 1
        ):
            continue

        conv_node = producers.get(slice_node.input[0])
        if (
            conv_node is None
            or conv_node.domain not in _STANDARD_DOMAINS
            or conv_node.op_type != "Conv"
        ):
            continue

        attrs = _attributes(conv_node)
        dilations = list(attrs.get("dilations", []))
        group = int(attrs.get("group", 1))
        if (
            len(dilations) != 2
            or dilations[1] != 1
            or (group, int(dilations[0])) not in _EXPECTED_PATTERNS
        ):
            continue
        dilation = int(dilations[0])

        expected_attrs = {
            "dilations": [dilation, 1],
            "group": group,
            "kernel_shape": [2, 3],
            "pads": [dilation, 1, dilation, 1],
            "strides": [1, 1],
        }
        for name, expected in expected_attrs.items():
            actual = attrs.get(name)
            if isinstance(expected, list):
                actual = list(actual) if actual is not None else None
            if actual != expected:
                raise ValueError(
                    f"Candidate Conv {conv_node.name!r} has {name}={actual!r}; "
                    f"expected {expected!r}."
                )
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise ValueError(f"Candidate Conv {conv_node.name!r} uses auto_pad.")
        if len(conv_node.input) != 3 or len(conv_node.output) != 1:
            raise ValueError(
                f"Candidate Conv {conv_node.name!r} must have data/weight/bias inputs."
            )

        conv_consumers = consumers[conv_node.output[0]]
        if len(conv_consumers) != 1 or conv_consumers[0] is not slice_node:
            raise ValueError(
                f"Candidate Conv output {conv_node.output[0]!r} is shared."
            )
        if conv_node.output[0] in graph_outputs:
            raise ValueError("Intermediate symmetric Conv is a graph output.")

        private_constants: list[onnx.NodeProto] = []
        expected_controls = ([0], [-dilation], [2], [1])
        for input_name, expected in zip(slice_node.input[1:], expected_controls):
            array, constant_node = _constant_array(
                input_name, initializers, producers
            )
            if (
                array.dtype != np.int64
                or array.shape != (1,)
                or array.tolist() != expected
            ):
                raise ValueError(
                    f"Slice {slice_node.name!r} control {input_name!r} is "
                    f"{array.dtype} {array.shape} {array.tolist()}, expected "
                    f"int64 (1,) {expected}."
                )
            if constant_node is not None:
                control_consumers = consumers[input_name]
                if (
                    len(control_consumers) != 1
                    or control_consumers[0] is not slice_node
                    or input_name in graph_outputs
                ):
                    raise ValueError(
                        f"Slice control Constant {constant_node.name!r} is shared."
                    )
                private_constants.append(constant_node)

        input_rank, input_type, input_shape = rank_type_shape.get(
            conv_node.input[0], (None, None, None)
        )
        output_rank, output_type, _ = rank_type_shape.get(
            slice_node.output[0], (None, None, None)
        )
        weight, _ = _constant_tensor(
            conv_node.input[1], initializers, producers
        )
        bias, _ = _constant_tensor(
            conv_node.input[2], initializers, producers
        )
        if input_rank != 4 or output_rank != 4:
            raise ValueError(
                f"Candidate ranks must be 4, got {input_rank} and {output_rank}."
            )
        if input_type != TensorProto.FLOAT or output_type != TensorProto.FLOAT:
            raise ValueError("Candidate data tensors must be float32.")
        if (
            weight.data_type != TensorProto.FLOAT
            or len(weight.dims) != 4
            or list(weight.dims[-2:]) != [2, 3]
        ):
            raise ValueError(
                f"Candidate Conv {conv_node.name!r} has invalid float32 weights."
            )
        if (
            bias.data_type != TensorProto.FLOAT
            or len(bias.dims) != 1
            or bias.dims[0] != weight.dims[0]
        ):
            raise ValueError(
                f"Candidate Conv {conv_node.name!r} has invalid float32 bias."
            )
        if (
            input_shape is not None
            and input_shape[1] is not None
            and int(weight.dims[1]) * group != input_shape[1]
        ):
            raise ValueError(
                f"Candidate Conv {conv_node.name!r} input/group channels disagree."
            )
        if int(weight.dims[0]) % group != 0:
            raise ValueError(
                f"Candidate Conv {conv_node.name!r} output channels are not divisible by group."
            )

        matches.append(
            (conv_node, slice_node, group, dilation, private_constants)
        )
        pattern_counts[(group, dilation)] += 1

    already_rewritten = _already_rewritten_patterns(model)
    if (
        pattern_counts != _EXPECTED_PATTERNS
        or len(matches) != 8
        or len(reflect_matches) != 1
        or already_rewritten_reflect != 0
    ):
        if (
            already_rewritten == _EXPECTED_PATTERNS
            and already_rewritten_reflect == 1
            and not matches
            and not reflect_matches
        ):
            raise ValueError(
                "Model already contains all eight asymmetric causal Convs and "
                "the static reflect-pad initializer."
            )
        raise ValueError(
            "Expected one causal Conv+Slice pair for each group=1/2 and "
            "dilation=1/2/4/8 plus one raw reflect-pad chain; found convs="
            f"{dict(sorted(pattern_counts.items()))}, raw_reflect="
            f"{len(reflect_matches)}, rewritten_reflect={already_rewritten_reflect}, "
            f"rewritten_convs={dict(sorted(already_rewritten.items()))}."
        )

    match_report = [
        {
            "conv_node": conv_node.name,
            "slice_node": slice_node.name,
            "input_tensor": conv_node.input[0],
            "intermediate_tensor": conv_node.output[0],
            "output_tensor": slice_node.output[0],
            "group": group,
            "dilation": dilation,
            "new_pads": [dilation, 1, 0, 1],
            "deleted_private_constants": [
                constant_node.name for constant_node in private_constants
            ],
        }
        for conv_node, slice_node, group, dilation, private_constants in matches
    ]
    pad_node, reflect_chain = reflect_matches[0]
    reflect_report = {
        "pad_node": pad_node.name,
        "data_tensor": pad_node.input[0],
        "old_control_tensor": pad_node.input[1],
        "output_tensor": pad_node.output[0],
        "mode": "reflect",
        "pads": _EXPECTED_REFLECT_PADS,
        "deleted_control_nodes": [node.name for node in reflect_chain],
    }

    tensor_names = (
        set(initializers)
        | {value.name for value in model.graph.input}
        | {value.name for value in model.graph.output}
        | {
            tensor_name
            for node in model.graph.node
            for tensor_name in (*node.input, *node.output)
            if tensor_name
        }
    )
    pads_initializer_name = "zipenhancer.reflect_pad_pads"
    suffix = 1
    while pads_initializer_name in tensor_names:
        pads_initializer_name = f"zipenhancer.reflect_pad_pads_{suffix}"
        suffix += 1
    pads_initializer = numpy_helper.from_array(
        np.asarray(_EXPECTED_REFLECT_PADS, dtype=np.int64),
        name=pads_initializer_name,
    )
    reflect_report["new_control_initializer"] = pads_initializer_name

    remove_keys: set[tuple[str, tuple[str, ...]]] = set()
    for conv_node, slice_node, _, dilation, private_constants in matches:
        remove_keys.add((slice_node.op_type, tuple(slice_node.output)))
        for constant_node in private_constants:
            remove_keys.add((constant_node.op_type, tuple(constant_node.output)))
        _replace_ints_attribute(conv_node, "pads", [dilation, 1, 0, 1])
        conv_node.output[0] = slice_node.output[0]

    for node in reflect_chain:
        remove_keys.add((node.op_type, tuple(node.output)))
    pad_node.input[1] = pads_initializer_name
    model.graph.initializer.append(pads_initializer)
    initializers[pads_initializer_name] = pads_initializer

    retained = [
        node
        for node in model.graph.node
        if (node.op_type, tuple(node.output)) not in remove_keys
    ]
    del model.graph.node[:]
    model.graph.node.extend(retained)

    produced = {
        output
        for node in model.graph.node
        for output in node.output
        if output
    }
    available = produced | {value.name for value in model.graph.input} | set(initializers)
    dangling = sorted(
        {
            input_name
            for node in model.graph.node
            for input_name in node.input
            if input_name and input_name not in available
        }
    )
    if dangling:
        raise ValueError(f"Rewrite produced dangling inputs: {dangling[:8]}")
    if _interface_signature(model) != interface_before:
        raise ValueError("Graph input/output contract changed.")
    if [(entry.domain, entry.version) for entry in model.opset_import] != opsets_before:
        raise ValueError("Opset imports changed.")
    if [(item.key, item.value) for item in model.metadata_props] != metadata_before:
        raise ValueError("Model metadata changed.")
    expected_initializer_names = initializer_names_before + [pads_initializer_name]
    if [item.name for item in model.graph.initializer] != expected_initializer_names:
        raise ValueError("Unexpected initializer change during padding rewrite.")

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.", suffix=".onnx", dir=final_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        onnx.save(model, str(temporary_path))
        saved = onnx.load(str(temporary_path), load_external_data=True)
        onnx.checker.check_model(saved, full_check=True)
        if _interface_signature(saved) != interface_before:
            raise ValueError("Saved final model changed the graph interface.")
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256:
        final_path.unlink(missing_ok=True)
        raise ValueError("Immutable source model changed during the rewrite.")
    final_sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()

    report = {
        "source_model": str(source_path),
        "final_model": str(final_path),
        "source_sha256": source_sha256,
        "final_sha256": final_sha256,
        "source_nodes": nodes_before,
        "final_nodes": len(model.graph.node),
        "rewritten_pairs": len(matches),
        "matched_regions": match_report,
        "rewritten_reflect_pads": len(reflect_matches),
        "reflect_pad_region": reflect_report,
        "inserted_nodes": 0,
        "added_initializers": [pads_initializer_name],
        "source_initializers": len(initializer_names_before),
        "final_initializers": len(model.graph.initializer),
        "rewired_conv_outputs": len(matches),
        "updated_conv_pads": len(matches),
        "deleted_slices": len(matches),
        "deleted_private_constants": sum(len(item[4]) for item in matches),
        "deleted_slice_nodes": [item[1].name for item in matches],
        "deleted_constant_nodes": [
            node.name for item in matches for node in item[4]
        ],
        "deleted_reflect_control_nodes": [node.name for node in reflect_chain],
        "deleted_reflect_control_node_count": len(reflect_chain),
        "transformed_initializers": 0,
        "patterns": {
            f"group_{group}_dilation_{dilation}": count
            for (group, dilation), count in sorted(pattern_counts.items())
        },
        "opset_imports_changed": False,
        "custom_domains_added": [],
        "graph_interface_changed": False,
        "external_data": False,
        "source_model_unchanged": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite validated ZipEnhancer exporter padding limitations."
    )
    parser.add_argument("source_model", help="Immutable raw PyTorch ONNX export")
    parser.add_argument("final_model", help="Separate rewritten deployment model")
    args = parser.parse_args()
    rewrite_asymmetric_causal_convs(args.source_model, args.final_model)


if __name__ == "__main__":
    main()
