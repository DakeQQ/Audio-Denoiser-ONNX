#!/usr/bin/env python
"""Rewrite the two static PyTorch reflect-Pad decompositions to ONNX Pad.

The legacy exporter expands ``torch.nn.functional.pad(..., mode='reflect')`` into
shape/control plumbing around a standard Pad node. For this static export the pads
are immutable. This fail-closed rewrite replaces exactly the mel and crossover
regions with direct standard-domain Pad-11 nodes while preserving the raw export.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_STANDARD_DOMAINS = ("", "ai.onnx")
_EXPECTED = {
    384: (96000, "/reflect_pad_mel"),
    255: (96000, "/reflect_pad_crossover"),
}


def _attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def _interface(model: onnx.ModelProto) -> tuple[bytes, bytes]:
    ins = onnx.GraphProto()
    outs = onnx.GraphProto()
    ins.input.extend(model.graph.input)
    outs.output.extend(model.graph.output)
    return ins.SerializeToString(), outs.SerializeToString()


def _rank_type(model: onnx.ModelProto) -> dict[str, tuple[int | None, int | None, tuple[int | None, ...]]]:
    result: dict[str, tuple[int | None, int | None, tuple[int | None, ...]]] = {}
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        tensor = value.type.tensor_type
        dims = tuple(int(dim.dim_value) if dim.dim_value else None for dim in tensor.shape.dim)
        result[value.name] = (len(dims), tensor.elem_type or None, dims)
    for initializer in model.graph.initializer:
        result[initializer.name] = (len(initializer.dims), initializer.data_type, tuple(initializer.dims))
    return result


def rewrite_reflect_padding(source_model_path: str | os.PathLike[str], final_model_path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(source_model_path).expanduser().resolve()
    final = Path(final_model_path).expanduser().resolve()
    if source == final:
        raise ValueError("Raw and final model paths must differ.")
    if not source.is_file():
        raise FileNotFoundError(source)

    model = onnx.load(str(source), load_external_data=True)
    interface_before = _interface(model)
    opsets_before = [(entry.domain, entry.version) for entry in model.opset_import]
    initializers_before = [item.name for item in model.graph.initializer]
    node_count_before = len(model.graph.node)

    if any(node.name in {item[1] for item in _EXPECTED.values()} for node in model.graph.node):
        raise ValueError("Model already contains the direct reflect-Pad rewrite.")
    tensor_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    tensor_names.update(item.name for item in model.graph.input)
    tensor_names.update(item.name for item in model.graph.output)
    tensor_names.update(item.name for item in model.graph.value_info)
    tensor_names.update(item.name for item in model.graph.initializer)
    reserved_pad_names = {f"reflect_pads_{half}" for half in _EXPECTED}
    collisions = sorted(tensor_names & reserved_pad_names)
    if collisions:
        raise ValueError(f"Reserved rewrite tensor names already exist: {collisions}.")

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    rank_type = _rank_type(inferred)
    producers = {output: node for node in model.graph.node for output in node.output if output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    graph_outputs = {item.name for item in model.graph.output}

    matches: dict[int, tuple[onnx.NodeProto, onnx.NodeProto, list[onnx.NodeProto]]] = {}
    for pad in model.graph.node:
        attrs = _attrs(pad)
        if (
            pad.domain not in _STANDARD_DOMAINS
            or pad.op_type != "Pad"
            or attrs != {"mode": b"reflect"}
            or len(pad.input) not in (2, 3)
            or len(pad.output) != 1
        ):
            continue
        pads_cast = producers.get(pad.input[1])
        reshape = producers.get(pads_cast.input[0]) if pads_cast is not None and pads_cast.op_type == "Cast" else None
        transpose = producers.get(reshape.input[0]) if reshape is not None and reshape.op_type == "Reshape" else None
        slice_node = producers.get(transpose.input[0]) if transpose is not None and transpose.op_type == "Transpose" else None
        if slice_node is None or slice_node.op_type != "Slice":
            continue

        # The exporter starts from a constant [left, right] tensor, then builds rank-aware pads.
        private_nodes: list[onnx.NodeProto] = []
        stack = [pad.input[1]]
        seen_tensors: set[str] = set()
        source_half = None
        while stack:
            tensor_name = stack.pop()
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)
            producer = producers.get(tensor_name)
            if producer is None:
                continue
            if producer is pad:
                continue
            private_nodes.append(producer)
            if producer.op_type == "Constant":
                value = _attrs(producer).get("value")
                if isinstance(value, onnx.TensorProto):
                    array = numpy_helper.to_array(value)
                    if array.dtype == np.int64 and array.shape == (2,) and int(array[0]) == int(array[1]) and int(array[0]) in _EXPECTED:
                        source_half = int(array[0])
            stack.extend(name for name in producer.input if name)

        if source_half is None:
            continue
        if source_half in matches:
            raise ValueError(f"Ambiguous reflect Pad with half-width {source_half}.")
        for node in private_nodes:
            for output in node.output:
                if output in graph_outputs:
                    raise ValueError("Reflect-Pad control plumbing reaches a graph output.")
                node_consumers = consumers.get(output, [])
                allowed_ids = {id(item) for item in private_nodes} | {id(pad)}
                if any(id(item) not in allowed_ids for item in node_consumers):
                    raise ValueError(f"Reflect-Pad control tensor {output!r} is shared.")

        data_rank, data_type, data_shape = rank_type.get(pad.input[0], (None, None, ()))
        output_rank, output_type, output_shape = rank_type.get(pad.output[0], (None, None, ()))
        expected_input_len, _ = _EXPECTED[source_half]
        if data_rank != 3 or output_rank != 3 or data_type != TensorProto.FLOAT or output_type != TensorProto.FLOAT:
            raise ValueError("Reflect Pad candidates must be rank-3 float32 tensors.")
        expected_output_shape = (1, 1, expected_input_len + 2 * source_half)
        output_shape_matches = len(output_shape) == 3 and all(
            actual is None or actual == expected
            for actual, expected in zip(output_shape, expected_output_shape)
        )
        if data_shape != (1, 1, expected_input_len) or not output_shape_matches:
            raise ValueError(
                f"Unexpected reflect Pad geometry for {source_half}: {data_shape} -> {output_shape}."
            )
        matches[source_half] = (pad, slice_node, private_nodes)

    if set(matches) != set(_EXPECTED):
        raise ValueError(f"Expected reflect Pad widths {sorted(_EXPECTED)}, found {sorted(matches)}.")

    remove_ids: set[int] = set()
    inserted: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    insert_before: dict[int, onnx.NodeProto] = {}
    for half, (old_pad, _, private_nodes) in matches.items():
        for node in private_nodes:
            remove_ids.add(id(node))
        remove_ids.add(id(old_pad))
        pads_name = f"reflect_pads_{half}"
        new_initializers.append(numpy_helper.from_array(np.array([0, 0, half, 0, 0, half], dtype=np.int64), pads_name))
        new_pad = helper.make_node(
            "Pad",
            [old_pad.input[0], pads_name],
            [old_pad.output[0]],
            name=_EXPECTED[half][1],
            mode="reflect",
        )
        insert_before[id(old_pad)] = new_pad
        inserted.append(new_pad)

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        replacement = insert_before.get(id(node))
        if replacement is not None:
            rewritten_nodes.append(replacement)
        if id(node) not in remove_ids:
            rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)

    if _interface(model) != interface_before:
        raise ValueError("Graph interface changed.")
    if [(entry.domain, entry.version) for entry in model.opset_import] != opsets_before:
        raise ValueError("Opset imports changed.")
    if [item.name for item in model.graph.initializer[: len(initializers_before)]] != initializers_before:
        raise ValueError("Existing initializer ordering changed.")
    if any(node.domain not in _STANDARD_DOMAINS for node in inserted):
        raise ValueError("Rewrite inserted a non-standard-domain node.")

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)

    final.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{final.stem}.", suffix=".onnx", dir=final.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        onnx.save(model, str(temporary))
        saved = onnx.load(str(temporary), load_external_data=True)
        onnx.checker.check_model(saved, full_check=True)
        if _interface(saved) != interface_before:
            raise ValueError("Saved graph interface changed.")
        os.replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)

    report = {
        "source_model": str(source),
        "final_model": str(final),
        "source_nodes": node_count_before,
        "final_nodes": len(model.graph.node),
        "matched_reflect_pads": sorted(matches),
        "inserted_standard_pad_nodes": len(inserted),
        "deleted_private_nodes": len(remove_ids) - len(matches),
        "added_int64_pad_initializers": len(new_initializers),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite static reflect-Pad decompositions.")
    parser.add_argument("source_model")
    parser.add_argument("final_model")
    args = parser.parse_args()
    rewrite_reflect_padding(args.source_model, args.final_model)


if __name__ == "__main__":
    main()
