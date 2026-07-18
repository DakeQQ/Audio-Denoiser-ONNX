#!/usr/bin/env python
"""Fold the nine DFSMN zero-prefix Concat nodes into asymmetric ONNX Conv pads.

PyTorch has no source-level Conv1d form for left-only causal padding. The raw
export therefore concatenates a known all-zero prefix before every depthwise
FSMN convolution. Standard ONNX Conv supports asymmetric ``pads=[19, 0]``.
This fail-closed rewrite validates the complete intended topology, preserves the
raw model, changes only those nine Conv inputs/attributes, and writes a distinct
final model atomically.
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


EXPECTED_MATCHES = 9
EXPECTED_CHANNELS = 256
EXPECTED_FRAMES = 99
EXPECTED_KERNEL = 20
EXPECTED_LEFT_PAD = EXPECTED_KERNEL - 1
EXPECTED_OPSET = 20
STANDARD_DOMAINS = ("", "ai.onnx")


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _interface_signature(model: onnx.ModelProto) -> tuple[bytes, bytes]:
    inputs = onnx.GraphProto()
    outputs = onnx.GraphProto()
    inputs.input.extend(model.graph.input)
    outputs.output.extend(model.graph.output)
    return inputs.SerializeToString(), outputs.SerializeToString()


def _value_specs(
    model: onnx.ModelProto,
) -> dict[str, tuple[int | None, tuple[int | None, ...]]]:
    specs: dict[str, tuple[int | None, tuple[int | None, ...]]] = {}
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        tensor_type = value.type.tensor_type
        shape = tuple(
            int(dimension.dim_value) if dimension.HasField("dim_value") else None
            for dimension in tensor_type.shape.dim
        )
        specs[value.name] = (tensor_type.elem_type or None, shape)
    for initializer in model.graph.initializer:
        specs[initializer.name] = (
            initializer.data_type,
            tuple(int(dimension) for dimension in initializer.dims),
        )
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_pads(node: onnx.NodeProto, values: list[int]) -> None:
    matches = [attribute for attribute in node.attribute if attribute.name == "pads"]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one pads attribute on {node.name!r}, found {len(matches)}.")
    del matches[0].ints[:]
    matches[0].ints.extend(values)


def rewrite_causal_fsmn_padding(
    raw_model_path: str | os.PathLike[str],
    final_model_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Rewrite exactly nine validated zero-prefix Concat + depthwise Conv pairs."""

    raw_path = Path(raw_model_path).expanduser().resolve()
    final_path = Path(final_model_path).expanduser().resolve()
    if raw_path == final_path:
        raise ValueError("Raw and final ONNX paths must be different.")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    model = onnx.load(str(raw_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    interface_before = _interface_signature(model)
    opsets_before = [(entry.domain, entry.version) for entry in model.opset_import]
    standard_opsets = [
        version for domain, version in opsets_before if domain in STANDARD_DOMAINS
    ]
    if standard_opsets != [EXPECTED_OPSET]:
        raise ValueError(
            f"Expected standard ONNX opset {EXPECTED_OPSET}, got {standard_opsets}.")

    metadata = {item.key: item.value for item in model.metadata_props}
    expected_metadata = {
        "model_name": "DFSMN",
        "dynamic_axes": "0",
        "opset": str(EXPECTED_OPSET),
        "input_audio_dtype": "INT16",
        "output_audio_dtype": "INT16",
        "export_audio_length": "96000",
    }
    mismatched_metadata = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatched_metadata:
        raise ValueError(f"Raw-model metadata mismatch: {mismatched_metadata}.")

    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=False)
    specs = _value_specs(inferred)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
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
    graph_outputs = {value.name for value in model.graph.output}

    already_rewritten = 0
    matches: list[tuple[onnx.NodeProto, onnx.NodeProto]] = []
    for conv in model.graph.node:
        if conv.domain not in STANDARD_DOMAINS or conv.op_type != "Conv":
            continue
        attrs = _attributes(conv)
        weight = initializers.get(conv.input[1]) if len(conv.input) >= 2 else None
        if (
            weight is None
            or weight.data_type != TensorProto.FLOAT
            or tuple(weight.dims) != (EXPECTED_CHANNELS, 1, EXPECTED_KERNEL)
            or attrs.get("group") != EXPECTED_CHANNELS
            or list(attrs.get("kernel_shape", [])) != [EXPECTED_KERNEL]
            or list(attrs.get("strides", [])) != [1]
            or list(attrs.get("dilations", [])) != [1]
        ):
            continue
        pads = list(attrs.get("pads", []))
        if pads == [EXPECTED_LEFT_PAD, 0]:
            already_rewritten += 1
            continue
        if pads != [0, 0] or len(conv.input) != 2 or len(conv.output) != 1:
            raise ValueError(f"Candidate FSMN Conv {conv.name!r} has an unexpected signature.")

        concat = producers.get(conv.input[0])
        if (
            concat is None
            or concat.domain not in STANDARD_DOMAINS
            or concat.op_type != "Concat"
            or len(concat.input) != 2
            or len(concat.output) != 1
            or _attributes(concat) != {"axis": 2}
        ):
            raise ValueError(
                f"Candidate FSMN Conv {conv.name!r} is not fed by the expected Concat.")
        pad = initializers.get(concat.input[0])
        if (
            pad is None
            or pad.data_type != TensorProto.FLOAT
            or tuple(pad.dims) != (1, EXPECTED_CHANNELS, EXPECTED_LEFT_PAD)
            or not np.all(numpy_helper.to_array(pad) == 0.0)
        ):
            raise ValueError(f"Concat {concat.name!r} does not use the exact zero pad.")
        if concat.output[0] in graph_outputs or consumers[concat.output[0]] != [conv]:
            raise ValueError(f"Concat output {concat.output[0]!r} is shared or public.")

        project = producers.get(concat.input[1])
        project_weight = (
            initializers.get(project.input[1])
            if project is not None and len(project.input) >= 2 else None
        )
        if (
            project is None
            or project.domain not in STANDARD_DOMAINS
            or project.op_type != "Conv"
            or len(project.input) != 2
            or project_weight is None
            or project_weight.data_type != TensorProto.FLOAT
            or tuple(project_weight.dims) != (EXPECTED_CHANNELS, EXPECTED_CHANNELS, 1)
        ):
            raise ValueError(f"Concat {concat.name!r} is not fed by a 256x256 projection.")

        conv_consumers = consumers[conv.output[0]]
        if (
            conv.output[0] in graph_outputs
            or len(conv_consumers) != 1
            or conv_consumers[0].domain not in STANDARD_DOMAINS
            or conv_consumers[0].op_type != "Add"
        ):
            raise ValueError(f"FSMN Conv output {conv.output[0]!r} lacks its private residual Add.")
        expected_spec = (TensorProto.FLOAT, (1, EXPECTED_CHANNELS, EXPECTED_FRAMES))
        if specs.get(concat.input[1]) != expected_spec or specs.get(conv.output[0]) != expected_spec:
            raise ValueError(
                f"Unexpected FSMN tensor geometry: {specs.get(concat.input[1])} -> "
                f"{specs.get(conv.output[0])}.")
        matches.append((concat, conv))

    if already_rewritten:
        if already_rewritten == EXPECTED_MATCHES and not matches:
            raise ValueError("Model already contains the asymmetric causal-Conv rewrite.")
        raise ValueError(
            f"Found a partially rewritten graph: {already_rewritten} final and "
            f"{len(matches)} raw FSMN convolutions.")
    if len(matches) != EXPECTED_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_MATCHES} causal FSMN patterns, found {len(matches)}.")

    pad_names = {concat.input[0] for concat, _ in matches}
    if len(pad_names) != 1:
        raise ValueError(f"Expected one shared zero-pad initializer, found {sorted(pad_names)}.")
    pad_name = next(iter(pad_names))
    if {id(node) for node in consumers[pad_name]} != {
        id(concat) for concat, _ in matches
    }:
        raise ValueError("The shared FSMN zero-pad initializer has unrelated consumers.")

    remove_ids = {id(concat) for concat, _ in matches}
    for concat, conv in matches:
        conv.input[0] = concat.input[1]
        _set_pads(conv, [EXPECTED_LEFT_PAD, 0])
    retained_nodes = [node for node in model.graph.node if id(node) not in remove_ids]
    retained_initializers = [
        initializer for initializer in model.graph.initializer
        if initializer.name != pad_name
    ]
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)

    if _interface_signature(model) != interface_before:
        raise ValueError("Graph input/output contract changed.")
    if [(entry.domain, entry.version) for entry in model.opset_import] != opsets_before:
        raise ValueError("Opset imports changed.")
    if {item.key: item.value for item in model.metadata_props} != metadata:
        raise ValueError("Model metadata changed.")
    if any(node.domain not in STANDARD_DOMAINS for node in model.graph.node):
        raise ValueError("The rewritten graph contains a non-standard operator domain.")

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.", suffix=".onnx", dir=final_path.parent)
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

    raw_histogram = Counter(node.op_type for node in inferred.graph.node)
    final_histogram = Counter(node.op_type for node in model.graph.node)
    report = {
        "raw_model": str(raw_path),
        "final_model": str(final_path),
        "raw_sha256": _sha256(raw_path),
        "final_sha256": _sha256(final_path),
        "matched_patterns": len(matches),
        "rewired_conv_inputs": len(matches),
        "updated_conv_pads": {"from": [0, 0], "to": [EXPECTED_LEFT_PAD, 0]},
        "inserted_nodes": 0,
        "deleted_nodes": len(matches),
        "deleted_initializers": [pad_name],
        "raw_node_count": len(inferred.graph.node),
        "final_node_count": len(model.graph.node),
        "raw_initializer_count": len(inferred.graph.initializer),
        "final_initializer_count": len(model.graph.initializer),
        "raw_operator_histogram": dict(sorted(raw_histogram.items())),
        "final_operator_histogram": dict(sorted(final_histogram.items())),
        "interface_changed": False,
        "opset_changed": False,
        "metadata_changed": False,
        "custom_domains_added": [],
        "standard_operator": "ai.onnx::Conv",
        "provider_requirements": "Standard ONNX Conv-20 with asymmetric 1-D pads",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_model", type=Path)
    parser.add_argument("final_model", type=Path)
    arguments = parser.parse_args()
    rewrite_causal_fsmn_padding(arguments.raw_model, arguments.final_model)


if __name__ == "__main__":
    main()