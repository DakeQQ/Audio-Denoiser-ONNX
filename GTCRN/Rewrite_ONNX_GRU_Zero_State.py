#!/usr/bin/env python3
"""Remove legacy-exporter zero-state scaffolding from raw GTCRN ONNX."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


EXPECTED_GRU_COUNT = 14
EXPECTED_DEAD_NODES = Counter({
    "Constant": 64,
    "Shape": 14,
    "Gather": 14,
    "Unsqueeze": 14,
    "Concat": 14,
    "Expand": 14,
})


def _attribute(node, name, default=None):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _constant_array(name, producers, initializers, allowed_nodes, allowed_initializers):
    if name in initializers:
        allowed_initializers.add(name)
        return numpy_helper.to_array(initializers[name])
    node = producers.get(name)
    if node is None or node.domain not in ("", "ai.onnx") or node.op_type != "Constant":
        raise ValueError(f"{name!r} is not a standard Constant or initializer")
    values = [attribute.t for attribute in node.attribute if attribute.name == "value" and attribute.HasField("t")]
    if len(values) != 1:
        raise ValueError(f"Constant {node.name!r} must contain exactly one tensor value")
    allowed_nodes.add(id(node))
    return numpy_helper.to_array(values[0])


def _producer(name, producers, op_type, allowed_nodes):
    node = producers.get(name)
    if node is None or node.domain not in ("", "ai.onnx") or node.op_type != op_type:
        actual = None if node is None else (node.domain, node.op_type, node.name)
        raise ValueError(f"Expected {op_type} producer for {name!r}, got {actual}")
    allowed_nodes.add(id(node))
    return node


def _single_consumer(name, consumers, expected):
    users = consumers.get(name, [])
    if users != [expected]:
        raise ValueError(
            f"{name!r} has consumers {[(node.op_type, node.name) for node in users]}, "
            f"expected only {expected.name!r}"
        )


def _require_int64(name, expected, producers, initializers, allowed_nodes, allowed_initializers):
    value = _constant_array(name, producers, initializers, allowed_nodes, allowed_initializers)
    if value.dtype != np.int64 or value.reshape(-1).tolist() != list(expected):
        raise ValueError(f"{name!r} must be int64 {list(expected)}, got {value.dtype} {value}")


def _validate_gru_weights(gru, initializers, num_directions, hidden_size, input_size):
    expected = (
        (gru.input[1], (num_directions, 3 * hidden_size, input_size)),
        (gru.input[2], (num_directions, 3 * hidden_size, hidden_size)),
        (gru.input[3], (num_directions, 6 * hidden_size)),
    )
    for name, shape in expected:
        tensor = initializers.get(name)
        if tensor is None:
            raise ValueError(f"GRU {gru.name!r} weight {name!r} is not an initializer")
        value = numpy_helper.to_array(tensor)
        if value.dtype != np.float32 or value.shape != shape:
            raise ValueError(
                f"GRU {gru.name!r} weight {name!r} must be float32 {shape}, "
                f"got {value.dtype} {value.shape}"
            )


def _validate_zero_state(gru, producers, consumers, initializers, value_info):
    allowed_nodes = set()
    allowed_initializers = set()
    if gru.domain not in ("", "ai.onnx") or len(gru.input) != 6 or not gru.input[5]:
        raise ValueError(f"GRU {gru.name!r} does not have the expected materialized initial_h")
    if gru.input[4] or not (1 <= len(gru.output) <= 2):
        raise ValueError(f"GRU {gru.name!r} does not match the legacy exporter signature")
    if int(_attribute(gru, "layout", 0)) != 0 or int(_attribute(gru, "linear_before_reset", 0)) != 1:
        raise ValueError(f"GRU {gru.name!r} must use standard sequence-first PyTorch GRU semantics")
    if _attribute(gru, "clip", None) is not None or _attribute(gru, "activations", None) is not None:
        raise ValueError(f"GRU {gru.name!r} contains unsupported activation or clip attributes")

    hidden_size = int(_attribute(gru, "hidden_size", 0))
    direction = _attribute(gru, "direction", b"forward")
    if isinstance(direction, str):
        direction = direction.encode()
    num_directions = 2 if direction == b"bidirectional" else 1 if direction in (b"forward", b"reverse") else 0
    if hidden_size <= 0 or not num_directions:
        raise ValueError(f"GRU {gru.name!r} has unsupported hidden_size/direction")

    x_info = value_info.get(gru.input[0])
    if x_info is None or not x_info.type.HasField("tensor_type"):
        raise ValueError(f"Missing inferred tensor type for {gru.input[0]!r}")
    x_type = x_info.type.tensor_type
    if x_type.elem_type != TensorProto.FLOAT or len(x_type.shape.dim) != 3:
        raise ValueError(f"GRU {gru.name!r} X must be rank-3 float32")
    input_dim = x_type.shape.dim[2]
    if not input_dim.HasField("dim_value") or input_dim.dim_value <= 0:
        raise ValueError(f"GRU {gru.name!r} input width must be statically known")
    batch_dim = x_type.shape.dim[1]
    static_batch = batch_dim.dim_value if batch_dim.HasField("dim_value") else None
    _validate_gru_weights(gru, initializers, num_directions, hidden_size, input_dim.dim_value)

    state = producers.get(gru.input[5])
    if state is None or state.domain not in ("", "ai.onnx") or state.op_type not in ("Expand", "ConstantOfShape"):
        actual = None if state is None else (state.domain, state.op_type, state.name)
        raise ValueError(f"GRU {gru.name!r} has unsupported initial_h producer {actual}")
    allowed_nodes.add(id(state))
    _single_consumer(state.output[0], consumers, gru)

    if state.op_type == "Expand":
        if len(state.input) != 2:
            raise ValueError(f"Expand {state.name!r} must have two inputs")
        seed = _constant_array(
            state.input[0], producers, initializers, allowed_nodes, allowed_initializers
        )
        if (
            seed.dtype != np.float32
            or seed.ndim != 3
            or seed.shape[0] != num_directions
            or seed.shape[1] not in ({1, static_batch} if static_batch is not None else {1})
            or seed.shape[2] != hidden_size
            or not np.all(seed == 0.0)
        ):
            raise ValueError(
                f"GRU {gru.name!r} initial-state seed must be all-zero float32 with "
                f"directions={num_directions}, batch=1 or {static_batch}, hidden={hidden_size}; "
                f"got {seed.dtype} {seed.shape}"
            )
        shape_name = state.input[1]
    else:
        if len(state.input) != 1:
            raise ValueError(f"ConstantOfShape {state.name!r} must have one shape input")
        fill = _attribute(state, "value", None)
        if fill is not None:
            fill = numpy_helper.to_array(fill)
            if fill.dtype != np.float32 or fill.size != 1 or float(fill.reshape(-1)[0]) != 0.0:
                raise ValueError(f"ConstantOfShape {state.name!r} must use float32 zero fill")
        shape_name = state.input[0]

    concat = _producer(shape_name, producers, "Concat", allowed_nodes)
    _single_consumer(concat.output[0], consumers, state)
    if int(_attribute(concat, "axis", -1)) != 0 or len(concat.input) != 3:
        raise ValueError(f"Concat {concat.name!r} does not construct a three-element shape")
    _require_int64(
        concat.input[0], [num_directions], producers, initializers, allowed_nodes, allowed_initializers
    )
    _require_int64(
        concat.input[2], [hidden_size], producers, initializers, allowed_nodes, allowed_initializers
    )

    unsqueeze = _producer(concat.input[1], producers, "Unsqueeze", allowed_nodes)
    _single_consumer(unsqueeze.output[0], consumers, concat)
    if len(unsqueeze.input) != 2:
        raise ValueError(f"Unsqueeze {unsqueeze.name!r} must have data and axes inputs")
    _require_int64(
        unsqueeze.input[1], [0], producers, initializers, allowed_nodes, allowed_initializers
    )

    gather = _producer(unsqueeze.input[0], producers, "Gather", allowed_nodes)
    _single_consumer(gather.output[0], consumers, unsqueeze)
    if int(_attribute(gather, "axis", 0)) != 0 or len(gather.input) != 2:
        raise ValueError(f"Gather {gather.name!r} must read one axis-0 Shape element")
    _require_int64(
        gather.input[1], [1], producers, initializers, allowed_nodes, allowed_initializers
    )

    shape = _producer(gather.input[0], producers, "Shape", allowed_nodes)
    _single_consumer(shape.output[0], consumers, gather)
    if len(shape.input) != 1 or shape.input[0] != gru.input[0]:
        raise ValueError(f"Shape {shape.name!r} must read the corresponding GRU X input")

    return allowed_nodes, allowed_initializers


def _live_region(model):
    producers = {output: node for node in model.graph.node for output in node.output if output}
    live_values = {output.name for output in model.graph.output}
    queue = deque(live_values)
    live_node_ids = set()
    while queue:
        node = producers.get(queue.popleft())
        if node is None or id(node) in live_node_ids:
            continue
        live_node_ids.add(id(node))
        for name in node.input:
            if name and name not in live_values:
                live_values.add(name)
                queue.append(name)
    return live_values, live_node_ids


def rewrite(raw_path, final_path, report_path=None):
    raw_path = Path(raw_path).resolve()
    final_path = Path(final_path).resolve()
    if raw_path == final_path:
        raise ValueError("Raw and final paths must differ; the raw export is immutable")
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {final_path}")

    model = onnx.load(str(raw_path), load_external_data=True)
    if any(tensor.external_data for tensor in model.graph.initializer):
        raise ValueError("External-data models are not expected for GTCRN and are not rewritten")
    onnx.checker.check_model(model, full_check=True)

    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    value_info = {
        value.name: value
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }
    producers = {output: node for node in model.graph.node for output in node.output if output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    graph_outputs = {value.name for value in model.graph.output}

    grus = [node for node in model.graph.node if node.domain in ("", "ai.onnx") and node.op_type == "GRU"]
    if len(grus) != EXPECTED_GRU_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_GRU_COUNT} standard GRUs, found {len(grus)}")
    if any(len(node.input) < 6 or not node.input[5] for node in grus):
        raise ValueError("Model is already rewritten or contains a GRU outside the expected pattern")

    allowed_dead_nodes = set()
    allowed_dead_initializers = set()
    rewired = []
    trimmed_outputs = []
    for gru in grus:
        region_nodes, region_initializers = _validate_zero_state(
            gru, producers, consumers, initializers, value_info
        )
        allowed_dead_nodes.update(region_nodes)
        allowed_dead_initializers.update(region_initializers)
        rewired.append({"gru": gru.name, "removed_initial_h": gru.input[5]})
        if len(gru.output) == 2:
            hidden_output = gru.output[1]
            if hidden_output in graph_outputs or consumers.get(hidden_output):
                raise ValueError(f"GRU {gru.name!r} hidden output is unexpectedly live")
            trimmed_outputs.append({"gru": gru.name, "removed_hidden_output": hidden_output})

    for gru in grus:
        gru.input[5] = ""
        if len(gru.output) == 2:
            del gru.output[1:]

    old_nodes = list(model.graph.node)
    live_values, live_node_ids = _live_region(model)
    dead_nodes = [node for node in old_nodes if id(node) not in live_node_ids]
    dead_counts = Counter(node.op_type for node in dead_nodes)
    if dead_counts != EXPECTED_DEAD_NODES:
        raise ValueError(
            f"Dead region mismatch: got {dict(dead_counts)}, "
            f"expected {dict(EXPECTED_DEAD_NODES)}"
        )
    unexpected_dead = [node for node in dead_nodes if id(node) not in allowed_dead_nodes]
    if unexpected_dead:
        raise ValueError(
            "Rewrite would delete nodes outside the matched zero-state regions: "
            f"{[(node.op_type, node.name) for node in unexpected_dead]}"
        )
    dead_initializers = [
        tensor.name for tensor in model.graph.initializer if tensor.name not in live_values
    ]
    unexpected_initializers = sorted(set(dead_initializers) - allowed_dead_initializers)
    if unexpected_initializers:
        raise ValueError(
            f"Rewrite would delete initializers outside the matched regions: {unexpected_initializers}"
        )

    del model.graph.node[:]
    model.graph.node.extend(node for node in old_nodes if id(node) in live_node_ids)
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        tensor for tensor in initializers.values() if tensor.name not in dead_initializers
    )
    del model.graph.value_info[:]
    model.graph.value_info.extend(
        value for value in inferred.graph.value_info if value.name in live_values
    )

    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(final_path))

    report = {
        "raw_model": str(raw_path),
        "final_model": str(final_path),
        "opset": 18,
        "matched_grus": len(grus),
        "rewired_inputs": rewired,
        "trimmed_optional_outputs": trimmed_outputs,
        "inserted_nodes": 0,
        "transformed_initializers": 0,
        "deleted_nodes": len(dead_nodes),
        "deleted_node_types": dict(sorted(dead_counts.items())),
        "deleted_initializers": dead_initializers,
        "custom_domains_added": [],
        "input_names": [value.name for value in model.graph.input],
        "output_names": [value.name for value in model.graph.output],
    }
    report_path = final_path.with_suffix(final_path.suffix + ".rewrite.json") if report_path is None else Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_model", type=Path)
    parser.add_argument("final_model", type=Path)
    args = parser.parse_args()
    print(json.dumps(rewrite(args.raw_model, args.final_model), indent=2))


if __name__ == "__main__":
    main()
