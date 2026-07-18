"""Remove legacy-exporter Identity aliases of immutable NKF initializers.

The TorchScript ONNX exporter emits one Identity per reuse of shared GRU
weights, biases, and initial zero states after the fixed Kalman loop is
unrolled. ONNX initializers may legally fan out directly, so these aliases are
pure exporter artifacts. This narrow rewrite validates the exact topology and
writes a separate final model; it never modifies the raw export.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx
from onnx import TensorProto


EXPECTED_GRU_USES_PER_INITIALIZER = 251
EXPECTED_ALIAS_COUNT = 1510
EXPECTED_CONSUMER_EDGE_COUNTS = {
    # Two GRUs, each with W, R, and B. Their weights are each reused by the
    # real and imaginary branches across 126 unrolled Kalman frames.
    ("GRU", 1, (1, 54, 18)): 2 * EXPECTED_GRU_USES_PER_INITIALIZER,
    ("GRU", 2, (1, 54, 18)): 2 * EXPECTED_GRU_USES_PER_INITIALIZER,
    ("GRU", 3, (1, 108)): 2 * EXPECTED_GRU_USES_PER_INITIALIZER,
    # The four initial hidden states share one immutable zero initializer.
    ("GRU", 5, (1, 513, 18)): 3,
    # The imaginary zero tap/state aliases the real zero tensor at t=0.
    ("Concat", 5, (1, 513, 4)): 1,
    ("Sub", 1, (1, 513, 4)): 1,
}


def _tensor_shape(tensor: onnx.TensorProto) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.dims)


def rewrite_initializer_identities(raw_path: Path, final_path: Path) -> dict:
    raw_path = raw_path.expanduser().resolve()
    final_path = final_path.expanduser().resolve()
    if raw_path == final_path:
        raise ValueError("Raw and final ONNX paths must be different.")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing final model: {final_path}")

    model = onnx.load(str(raw_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    graph_outputs = {value.name for value in graph.output}
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name:
                consumers[input_name].append((node, input_index))

    identities: list[onnx.NodeProto] = []
    rewrites: dict[str, str] = {}
    observed = Counter()
    for node in graph.node:
        if node.op_type != "Identity":
            continue
        if node.domain or len(node.input) != 1 or len(node.output) != 1 or node.attribute:
            raise RuntimeError(f"Unexpected Identity signature: {node.name or node.output[0]}")
        source_name, output_name = node.input[0], node.output[0]
        if source_name not in initializers:
            raise RuntimeError(f"Identity source is not an initializer: {node.name or output_name}")
        if output_name in graph_outputs:
            raise RuntimeError(f"Refusing to remove graph-output Identity: {node.name or output_name}")
        output_consumers = consumers.get(output_name, [])
        if not output_consumers:
            raise RuntimeError(
                f"Expected at least one consumer for {node.name or output_name}"
            )
        tensor = initializers[source_name]
        if tensor.data_type != TensorProto.FLOAT:
            raise RuntimeError(f"Expected FLOAT initializer for {node.name or output_name}")
        for consumer, input_index in output_consumers:
            key = (consumer.op_type, input_index, _tensor_shape(tensor))
            observed[key] += 1
        identities.append(node)
        rewrites[output_name] = source_name

    if len(identities) != EXPECTED_ALIAS_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ALIAS_COUNT} initializer Identity aliases, got {len(identities)}"
        )
    if observed != Counter(EXPECTED_CONSUMER_EDGE_COUNTS):
        raise RuntimeError(
            "Initializer Identity topology mismatch.\n"
            f"Expected: {dict(EXPECTED_CONSUMER_EDGE_COUNTS)}\nObserved: {dict(observed)}"
        )

    identity_ids = {id(node) for node in identities}
    rewired_inputs = 0
    retained_nodes = []
    for node in graph.node:
        if id(node) in identity_ids:
            continue
        for input_index, input_name in enumerate(node.input):
            replacement = rewrites.get(input_name)
            if replacement is not None:
                node.input[input_index] = replacement
                rewired_inputs += 1
        retained_nodes.append(node)
    expected_rewired_inputs = sum(EXPECTED_CONSUMER_EDGE_COUNTS.values())
    if rewired_inputs != expected_rewired_inputs:
        raise RuntimeError(f"Expected {expected_rewired_inputs} rewired inputs, got {rewired_inputs}")

    del graph.node[:]
    graph.node.extend(retained_nodes)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(final_path))
    onnx.checker.check_model(onnx.load(str(final_path)), full_check=True)

    report = {
        "raw_model": str(raw_path),
        "final_model": str(final_path),
        "matched_identity_aliases": len(identities),
        "rewired_inputs": rewired_inputs,
        "inserted_nodes": 0,
        "deleted_nodes": len(identities),
        "deleted_initializers": 0,
        "raw_node_count": len(retained_nodes) + len(identities),
        "final_node_count": len(retained_nodes),
        "interface_changed": False,
        "opset_changed": False,
        "custom_domains_added": [],
        "observed_preconditions": {
            f"{op_type}:input_{input_index}:shape_{shape}": count
            for (op_type, input_index, shape), count in sorted(observed.items())
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_model",
        type=Path,
        help="Temporary raw PyTorch-exported ONNX model.",
    )
    parser.add_argument(
        "final_model",
        type=Path,
        help="Distinct output path for the rewritten ONNX model.",
    )
    args = parser.parse_args()
    print(json.dumps(rewrite_initializer_identities(args.raw_model, args.final_model), indent=2))


if __name__ == "__main__":
    main()
