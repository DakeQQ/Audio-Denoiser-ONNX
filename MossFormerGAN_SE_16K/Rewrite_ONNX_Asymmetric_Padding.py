#!/usr/bin/env python
"""Replace the exporter's causal Conv+Slice pattern with asymmetric ONNX Conv.

The PyTorch exporter writes a transient source model containing symmetric
padding followed by a tail Slice. ONNX Conv supports the required asymmetric
padding directly. This checked rewrite produces the deployment model; callers
may delete the transient source model immediately afterward.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_EXPECTED_DILATIONS = Counter({1: 3, 2: 3, 4: 3, 8: 3})
_STANDARD_DOMAINS = ("", "ai.onnx")


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


def _constant_array(
    name: str,
    initializers: dict[str, onnx.TensorProto],
    producers: dict[str, onnx.NodeProto],
) -> tuple[np.ndarray, onnx.NodeProto | None]:
    initializer = initializers.get(name)
    if initializer is not None:
        return numpy_helper.to_array(initializer), None

    producer = producers.get(name)
    if (
        producer is None
        or producer.domain not in _STANDARD_DOMAINS
        or producer.op_type != "Constant"
    ):
        raise ValueError(f"Slice control {name!r} is not a Constant or initializer.")
    attrs = _attributes(producer)
    value = attrs.get("value")
    if set(attrs) != {"value"} or not isinstance(value, onnx.TensorProto):
        raise ValueError(f"Constant producer for {name!r} is not a tensor Constant.")
    return numpy_helper.to_array(value), producer


def _interface_signature(model: onnx.ModelProto) -> tuple[bytes, bytes]:
    inputs = onnx.GraphProto()
    outputs = onnx.GraphProto()
    inputs.input.extend(model.graph.input)
    outputs.output.extend(model.graph.output)
    return inputs.SerializeToString(), outputs.SerializeToString()


def _rank_and_type(
    model: onnx.ModelProto,
) -> dict[str, tuple[int | None, int | None]]:
    result: dict[str, tuple[int | None, int | None]] = {}
    values = [
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ]
    for value in values:
        tensor_type = value.type.tensor_type
        rank = len(tensor_type.shape.dim) if tensor_type.HasField("shape") else None
        result[value.name] = (rank, tensor_type.elem_type or None)
    for initializer in model.graph.initializer:
        result[initializer.name] = (
            len(initializer.dims), initializer.data_type
        )
    return result


def rewrite_asymmetric_causal_convs(
    source_model_path: str | os.PathLike[str],
    final_model_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write the final model after replacing exactly 12 validated patterns."""

    source_path = Path(source_model_path).expanduser().resolve()
    final_path = Path(final_model_path).expanduser().resolve()
    if source_path == final_path:
        raise ValueError("Source and final ONNX paths must be different.")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    model = onnx.load(str(source_path), load_external_data=True)
    interface_before = _interface_signature(model)
    opsets_before = [(entry.domain, entry.version) for entry in model.opset_import]
    initializer_names_before = [item.name for item in model.graph.initializer]
    nodes_before = len(model.graph.node)

    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=False
    )
    rank_type = _rank_and_type(inferred)
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

    matches: list[
        tuple[onnx.NodeProto, onnx.NodeProto, int, list[onnx.NodeProto]]
    ] = []
    dilation_counts: Counter[int] = Counter()

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
        if len(dilations) != 2 or dilations[1] != 1:
            continue
        dilation = int(dilations[0])
        if dilation not in _EXPECTED_DILATIONS:
            continue

        expected_attrs = {
            "dilations": [dilation, 1],
            "group": 1,
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
        for input_name, expected in zip(
            slice_node.input[1:], expected_controls
        ):
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

        input_rank, input_type = rank_type.get(
            conv_node.input[0], (None, None)
        )
        output_rank, output_type = rank_type.get(
            slice_node.output[0], (None, None)
        )
        weight = initializers.get(conv_node.input[1])
        bias = initializers.get(conv_node.input[2])
        if input_rank != 4 or output_rank != 4:
            raise ValueError(
                f"Candidate ranks must be 4, got {input_rank} and {output_rank}."
            )
        if input_type != TensorProto.FLOAT or output_type != TensorProto.FLOAT:
            raise ValueError("Candidate tensors must be float32.")
        if (
            weight is None
            or weight.data_type != TensorProto.FLOAT
            or list(weight.dims[-2:]) != [2, 3]
        ):
            raise ValueError(
                f"Candidate Conv {conv_node.name!r} has invalid weights."
            )
        if (
            bias is None
            or bias.data_type != TensorProto.FLOAT
            or len(bias.dims) != 1
        ):
            raise ValueError(
                f"Candidate Conv {conv_node.name!r} has invalid bias."
            )

        matches.append(
            (conv_node, slice_node, dilation, private_constants)
        )
        dilation_counts[dilation] += 1

    if dilation_counts != _EXPECTED_DILATIONS or len(matches) != 12:
        raise ValueError(
            "Expected three causal Conv+Slice pairs for each dilation 1/2/4/8; "
            f"found {dict(sorted(dilation_counts.items()))}."
        )

    remove_keys: set[tuple[str, tuple[str, ...]]] = set()
    for conv_node, slice_node, dilation, private_constants in matches:
        remove_keys.add((slice_node.op_type, tuple(slice_node.output)))
        for constant_node in private_constants:
            remove_keys.add(
                (constant_node.op_type, tuple(constant_node.output))
            )
        _replace_ints_attribute(
            conv_node, "pads", [dilation, 1, 0, 1]
        )
        conv_node.output[0] = slice_node.output[0]

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
    available = (
        produced
        | {value.name for value in model.graph.input}
        | set(initializers)
    )
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
    if [item.name for item in model.graph.initializer] != initializer_names_before:
        raise ValueError("Initializers changed during padding rewrite.")

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=False
    )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.",
        suffix=".onnx",
        dir=final_path.parent,
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

    report = {
        "source_nodes": nodes_before,
        "final_nodes": len(model.graph.node),
        "rewritten_pairs": len(matches),
        "deleted_slices": len(matches),
        "deleted_private_constants": sum(
            len(item[3]) for item in matches
        ),
        "dilations": dict(sorted(dilation_counts.items())),
        "final_model": str(final_path),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite validated causal Conv+Slice pairs."
    )
    parser.add_argument("source_model", help="Transient PyTorch ONNX export")
    parser.add_argument("final_model", help="Final rewritten ONNX model")
    args = parser.parse_args()
    rewrite_asymmetric_causal_convs(
        args.source_model, args.final_model
    )


if __name__ == "__main__":
    main()
