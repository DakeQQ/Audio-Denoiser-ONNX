"""Targeted rewrites for legacy-exporter limitations in DFSMN_AEC.

This is deliberately not a general ONNX optimizer. It exposes two profile-specific
entry points and fails closed unless the complete expected static graph is present.

The SDAEC entry point performs exactly two standard-ONNX rewrites:

1. Omit exporter-generated all-zero initial_h/initial_c inputs from 14 LSTM
   nodes. ONNX LSTM defines omitted initial states as zeros, but the legacy
   PyTorch exporter materializes each state with Shape/Gather/Concat/Expand.
2. Fold 13 static all-zero left-prefix Concat nodes into the consuming 1-D
   Conv's asymmetric ``pads=[left, 0]`` attribute.

The NKF entry point removes only legacy-exporter ``Identity`` aliases of immutable
GRU, Linear, and zero-state initializers after validating their complete consumer
topology for the supplied static frame/batch/model dimensions.

The raw input model is read-only. The rewritten graph and a JSON audit report are
always written to distinct paths.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

EXPECTED_LSTM_COUNT = 14
EXPECTED_PAD_SIGNATURES = Counter({
    (9, 10, 1, (1, 2, 10)): 1,
    (19, 20, 512, (512, 1, 20)): 12,
})


def _attribute(node, name):
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute
    return None


def _int_attribute(node, name, default=None):
    attribute = _attribute(node, name)
    return default if attribute is None else int(attribute.i)


def _ints_attribute(node, name, default=None):
    attribute = _attribute(node, name)
    return default if attribute is None else tuple(int(value) for value in attribute.ints)


def _constant_array(node):
    if node.op_type != "Constant" or node.domain not in ("", "ai.onnx"):
        raise ValueError(f"Expected standard Constant, got {node.domain}:{node.op_type} ({node.name}).")
    value = _attribute(node, "value")
    if value is None:
        raise ValueError(f"Constant {node.name} does not use a TensorProto 'value' attribute.")
    return numpy_helper.to_array(value.t)


def _value_info_map(model):
    values = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        values[value.name] = value
    return values


def _tensor_type_and_shape(value):
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return tensor_type.elem_type, None
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return tensor_type.elem_type, tuple(shape)


def _interface_signature(model):
    def one(value):
        dtype, shape = _tensor_type_and_shape(value)
        return value.name, dtype, shape

    return {
        "inputs": [one(value) for value in model.graph.input],
        "outputs": [one(value) for value in model.graph.output],
        "opsets": [(item.domain, item.version) for item in model.opset_import],
    }


def _op_histogram(model):
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def _model_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_and_omit_lstm_zero_states(model, inferred, producers):
    value_info = _value_info_map(inferred)
    lstms = [
        node for node in model.graph.node
        if node.op_type == "LSTM" and node.domain in ("", "ai.onnx")
    ]
    if len(lstms) != EXPECTED_LSTM_COUNT:
        raise ValueError(f"Expected {EXPECTED_LSTM_COUNT} standard LSTM nodes, found {len(lstms)}.")

    already_omitted = [node for node in lstms if len(node.input) >= 7 and not node.input[5] and not node.input[6]]
    if already_omitted:
        if len(already_omitted) == len(lstms):
            raise ValueError("All LSTM initial states are already omitted; input appears already rewritten.")
        raise ValueError("Only some LSTM initial states are omitted; refusing a partially rewritten graph.")

    candidate_nodes = set()
    rewrites = []
    for node in lstms:
        if len(node.input) != 7 or node.input[4] != "" or not node.input[5] or not node.input[6]:
            raise ValueError(f"Unexpected LSTM input contract at {node.name}: {list(node.input)}")
        hidden_size = _int_attribute(node, "hidden_size")
        direction_attr = _attribute(node, "direction")
        direction = direction_attr.s.decode("utf-8") if direction_attr is not None else "forward"
        num_directions = 2 if direction == "bidirectional" else 1
        if hidden_size is None or hidden_size <= 0 or direction not in ("forward", "reverse", "bidirectional"):
            raise ValueError(f"Unsupported LSTM attributes at {node.name}.")

        removed_inputs = []
        for input_index in (5, 6):
            state_name = node.input[input_index]
            expand = producers.get(state_name)
            if expand is None or expand.op_type != "Expand" or expand.domain not in ("", "ai.onnx"):
                raise ValueError(f"LSTM {node.name} state {input_index} is not produced by standard Expand.")
            if len(expand.input) != 2:
                raise ValueError(f"Expand {expand.name} must have exactly two inputs.")
            zero_constant = producers.get(expand.input[0])
            if zero_constant is None:
                raise ValueError(f"Expand {expand.name} zero input is not produced by Constant.")
            zero = _constant_array(zero_constant)
            if zero.dtype != np.float32 or zero.size == 0 or np.any(zero != 0.0):
                raise ValueError(f"Expand {expand.name} does not expand an all-zero float32 constant.")
            shape_producer = producers.get(expand.input[1])
            if shape_producer is None or shape_producer.op_type != "Concat":
                raise ValueError(f"Expand {expand.name} shape is not produced by Concat.")

            value = value_info.get(state_name)
            if value is None:
                raise ValueError(f"No inferred type/shape for LSTM state {state_name}.")
            dtype, shape = _tensor_type_and_shape(value)
            if dtype != onnx.TensorProto.FLOAT or shape is None or len(shape) != 3:
                raise ValueError(f"LSTM state {state_name} must be rank-3 float32, got dtype={dtype}, shape={shape}.")
            if shape[0] != num_directions or shape[2] != hidden_size:
                raise ValueError(
                    f"LSTM state {state_name} shape {shape} does not match directions={num_directions}, "
                    f"hidden_size={hidden_size}.")

            stack = [state_name]
            while stack:
                tensor_name = stack.pop()
                producer = producers.get(tensor_name)
                if producer is None:
                    continue
                producer_index = next(
                    index for index, candidate in enumerate(model.graph.node) if candidate is producer)
                if producer_index in candidate_nodes:
                    continue
                candidate_nodes.add(producer_index)
                # Shape reads the live recurrent input. Delete Shape if dead, but do not
                # traverse into its data producer and accidentally broaden the rewrite.
                if producer.op_type != "Shape":
                    stack.extend(name for name in producer.input if name)

            removed_inputs.append(state_name)
            node.input[input_index] = ""

        rewrites.append({
            "node": node.name,
            "direction": direction,
            "hidden_size": hidden_size,
            "removed_inputs": removed_inputs,
        })
    return candidate_nodes, rewrites


def _validate_and_fold_zero_prefix_convs(model, inferred, producers, initializers, consumers):
    value_info = _value_info_map(inferred)
    matches = []
    signatures = Counter()
    concat_node_indices = set()
    candidate_initializers = set()

    for node in model.graph.node:
        if node.op_type != "Conv" or node.domain not in ("", "ai.onnx") or not node.input:
            continue
        concat = producers.get(node.input[0])
        if concat is None or concat.op_type != "Concat" or concat.domain not in ("", "ai.onnx"):
            continue
        if len(concat.input) != 2 or len(concat.output) != 1:
            continue
        axis = _int_attribute(concat, "axis")
        if axis not in (2, -1):
            continue
        pad_name, data_name = concat.input
        pad_initializer = initializers.get(pad_name)
        if pad_initializer is None:
            continue
        pad = numpy_helper.to_array(pad_initializer)
        if pad.dtype != np.float32 or pad.ndim != 3 or pad.size == 0 or np.any(pad != 0.0):
            continue
        if len(consumers.get(concat.output[0], ())) != 1 or consumers[concat.output[0]][0] is not node:
            raise ValueError(f"Zero-prefix Concat {concat.name} has consumers other than {node.name}.")

        weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if weight is None:
            raise ValueError(f"Conv {node.name} weight is not an initializer.")
        weight_array = numpy_helper.to_array(weight)
        if weight_array.dtype != np.float32 or weight_array.ndim != 3:
            raise ValueError(f"Conv {node.name} weight must be rank-3 float32.")
        kernel_shape = _ints_attribute(node, "kernel_shape")
        dilations = _ints_attribute(node, "dilations", (1,))
        strides = _ints_attribute(node, "strides", (1,))
        pads = _ints_attribute(node, "pads", (0, 0))
        group = _int_attribute(node, "group", 1)
        auto_pad = _attribute(node, "auto_pad")
        if (
            kernel_shape is None or len(kernel_shape) != 1
            or len(dilations) != 1 or len(strides) != 1 or strides != (1,)
            or pads != (0, 0)
            or (auto_pad is not None and auto_pad.s not in (b"", b"NOTSET"))
        ):
            raise ValueError(f"Conv {node.name} has unsupported padding/stride attributes.")
        pad_length = int(pad.shape[2])
        expected_pad = dilations[0] * (kernel_shape[0] - 1)
        if pad_length != expected_pad or pad.shape[1] != weight_array.shape[1] * group:
            raise ValueError(
                f"Conv {node.name} zero prefix {pad.shape} is incompatible with weight "
                f"{weight_array.shape}, dilation={dilations}, group={group}.")

        data_value = value_info.get(data_name)
        concat_value = value_info.get(concat.output[0])
        if data_value is None or concat_value is None:
            raise ValueError(f"Missing inferred value information for {concat.name}.")
        data_dtype, data_shape = _tensor_type_and_shape(data_value)
        concat_dtype, concat_shape = _tensor_type_and_shape(concat_value)
        if data_dtype != onnx.TensorProto.FLOAT or concat_dtype != onnx.TensorProto.FLOAT:
            raise ValueError(f"Concat {concat.name} must operate on float32 tensors.")
        if data_shape is None or concat_shape is None or len(data_shape) != 3 or len(concat_shape) != 3:
            raise ValueError(f"Concat {concat.name} must operate on inferred rank-3 tensors.")
        if data_shape[:2] != tuple(pad.shape[:2]) or concat_shape[2] != data_shape[2] + pad_length:
            raise ValueError(f"Concat {concat.name} inferred shapes do not match its zero prefix.")

        signature = (pad_length, kernel_shape[0], group, tuple(weight_array.shape))
        signatures[signature] += 1
        matches.append((node, concat, pad_name, data_name, signature))

    if signatures != EXPECTED_PAD_SIGNATURES:
        raise ValueError(
            f"Expected zero-prefix Conv signatures {dict(EXPECTED_PAD_SIGNATURES)}, found {dict(signatures)}.")

    rewrites = []
    for node, concat, pad_name, data_name, signature in matches:
        node.input[0] = data_name
        pads_attribute = _attribute(node, "pads")
        if pads_attribute is None:
            node.attribute.extend([onnx.helper.make_attribute("pads", [signature[0], 0])])
        else:
            del pads_attribute.ints[:]
            pads_attribute.ints.extend([signature[0], 0])
        concat_node_indices.add(next(
            index for index, candidate in enumerate(model.graph.node) if candidate is concat))
        candidate_initializers.add(pad_name)
        rewrites.append({
            "conv": node.name,
            "concat": concat.name,
            "zero_initializer": pad_name,
            "data_input": data_name,
            "pads": [signature[0], 0],
            "kernel_shape": [signature[1]],
            "group": signature[2],
            "weight_shape": list(signature[3]),
        })
    return concat_node_indices, candidate_initializers, rewrites


def _delete_targeted_dead_nodes(model, candidate_indices):
    graph_outputs = {value.name for value in model.graph.output}
    active = set(range(len(model.graph.node)))
    deleted = []
    while True:
        consumers = defaultdict(list)
        for index in active:
            for name in model.graph.node[index].input:
                if name:
                    consumers[name].append(index)
        removable = []
        for index in sorted(candidate_indices & active):
            node = model.graph.node[index]
            if all(output not in graph_outputs and not consumers.get(output) for output in node.output):
                removable.append(index)
        if not removable:
            break
        for index in removable:
            active.remove(index)
            deleted.append(model.graph.node[index].name or f"<{model.graph.node[index].op_type}:{index}>")
    if not deleted:
        raise ValueError("Targeted dead-node deletion removed nothing.")
    kept = [node for index, node in enumerate(model.graph.node) if index in active]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return deleted


def _delete_targeted_dead_initializers(model, candidate_names):
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.input)
    used.update(value.name for value in model.graph.output)
    deleted = []
    kept = []
    for initializer in model.graph.initializer:
        if initializer.name in candidate_names and initializer.name not in used:
            deleted.append(initializer.name)
        else:
            kept.append(initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    if set(deleted) != set(candidate_names):
        raise ValueError(
            f"Expected to delete targeted initializers {sorted(candidate_names)}, deleted {sorted(deleted)}.")
    return deleted


def rewrite_model(raw_model_path, final_model_path, report_path=None):
    raw_model_path = Path(raw_model_path).expanduser().resolve()
    final_model_path = Path(final_model_path).expanduser().resolve()
    if raw_model_path == final_model_path:
        raise ValueError("Raw and final ONNX paths must be distinct.")
    if not raw_model_path.is_file():
        raise FileNotFoundError(raw_model_path)

    model = onnx.load(str(raw_model_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    original_interface = _interface_signature(model)
    before_histogram = _op_histogram(model)
    before_nodes = len(model.graph.node)
    before_initializers = len(model.graph.initializer)

    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}

    state_candidates, lstm_rewrites = _validate_and_omit_lstm_zero_states(
        model, inferred, producers)
    concat_candidates, initializer_candidates, conv_rewrites = _validate_and_fold_zero_prefix_convs(
        model, inferred, producers, initializers, consumers)
    deleted_nodes = _delete_targeted_dead_nodes(
        model, state_candidates | concat_candidates)
    deleted_initializers = _delete_targeted_dead_initializers(
        model, initializer_candidates)

    if len(lstm_rewrites) != EXPECTED_LSTM_COUNT or len(conv_rewrites) != sum(EXPECTED_PAD_SIGNATURES.values()):
        raise AssertionError("Internal rewrite-count mismatch.")
    if _interface_signature(model) != original_interface:
        raise ValueError("Graph interface or opset imports changed during the targeted rewrite.")

    onnx.checker.check_model(model, full_check=True)
    inferred_final = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred_final, full_check=True)

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=final_model_path.name + ".", suffix=".tmp", dir=final_model_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        onnx.save_model(model, str(temporary_path))
        temporary_path.replace(final_model_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    after_histogram = _op_histogram(model)
    report = {
        "raw_model": str(raw_model_path),
        "final_model": str(final_model_path),
        "raw_sha256": _model_sha256(raw_model_path),
        "final_sha256": _model_sha256(final_model_path),
        "raw_nodes": before_nodes,
        "final_nodes": len(model.graph.node),
        "raw_initializers": before_initializers,
        "final_initializers": len(model.graph.initializer),
        "raw_op_histogram": before_histogram,
        "final_op_histogram": after_histogram,
        "lstm_zero_state_rewrites": lstm_rewrites,
        "asymmetric_conv_padding_rewrites": conv_rewrites,
        "deleted_nodes": deleted_nodes,
        "deleted_initializers": deleted_initializers,
        "interface": original_interface,
        "custom_domains_added": [],
        "opset_imports_changed": False,
    }
    if report_path is not None:
        report_path = Path(report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def rewrite_nkf_initializer_identities(
    raw_model_path,
    final_model_path,
    report_path=None,
    *,
    frames,
    model_batch,
    n_freq,
    filter_order,
    fc_dim,
    rnn_dim,
):
    """Remove only legacy-exporter aliases of immutable NKF initializers.

    The static Kalman loop invokes two real-valued GRU modules twice per frame
    and six Linear weights once per frame. The legacy exporter fans repeated
    initializer uses through Identity nodes. Direct initializer fan-out is
    standard ONNX, so this rewrite validates the complete consumer multiset
    before rewiring it.
    """
    raw_model_path = Path(raw_model_path).expanduser().resolve()
    final_model_path = Path(final_model_path).expanduser().resolve()
    if raw_model_path == final_model_path:
        raise ValueError('Raw and final ONNX paths must be distinct.')
    if not raw_model_path.is_file():
        raise FileNotFoundError(raw_model_path)
    if min(frames, model_batch, n_freq, filter_order, fc_dim, rnn_dim) <= 0:
        raise ValueError('All NKF static dimensions must be positive integers.')

    model = onnx.load(str(raw_model_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    original_interface = _interface_signature(model)
    before_histogram = _op_histogram(model)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    graph_outputs = {value.name for value in model.graph.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name:
                consumers[input_name].append((node, input_index))

    gru_aliases_per_initializer = 2 * frames - 1
    linear_aliases_per_initializer = frames - 1
    expected_observed = Counter({
        ('GRU', 1, (1, 3 * rnn_dim, fc_dim)): 2 * gru_aliases_per_initializer,
        ('GRU', 2, (1, 3 * rnn_dim, rnn_dim)): 2 * gru_aliases_per_initializer,
        ('GRU', 3, (1, 6 * rnn_dim)): 2 * gru_aliases_per_initializer,
        ('GRU', 5, (1, model_batch * n_freq, rnn_dim)): 3,
        ('MatMul', 1, (2 * filter_order + 1, fc_dim)): 2 * linear_aliases_per_initializer,
        ('MatMul', 1, (rnn_dim, fc_dim)): 2 * linear_aliases_per_initializer,
        ('MatMul', 1, (fc_dim, filter_order)): 2 * linear_aliases_per_initializer,
        ('Concat', 2, (model_batch, n_freq, filter_order)): 1,
        ('Sub', 1, (model_batch, n_freq, filter_order)): 1,
    })
    expected_identity_count = (
        6 * gru_aliases_per_initializer
        + 6 * linear_aliases_per_initializer
        + 4
    )

    targeted_signatures = set(expected_observed)
    identities = []
    retained_identities = []
    rewrites = {}
    observed = Counter()
    for node in model.graph.node:
        if node.op_type != 'Identity':
            continue
        if node.domain or len(node.input) != 1 or len(node.output) != 1 or node.attribute:
            raise ValueError(f'Unexpected Identity signature: {node.name or node.output[0]}')
        source_name, output_name = node.input[0], node.output[0]
        initializer = initializers.get(source_name)
        if initializer is None:
            raise ValueError(f'Identity source is not an initializer: {node.name or output_name}')
        if initializer.data_type != onnx.TensorProto.FLOAT:
            raise ValueError(f'Identity source is not FLOAT: {node.name or output_name}')
        if output_name in graph_outputs:
            raise ValueError(f'Refusing graph-output Identity: {node.name or output_name}')
        output_consumers = consumers.get(output_name, ())
        if not output_consumers:
            raise ValueError(f'Identity has no consumers: {node.name or output_name}')
        shape = tuple(int(dim) for dim in initializer.dims)
        consumer_signatures = [
            (consumer.op_type, input_index, shape)
            for consumer, input_index in output_consumers
        ]
        targeted_consumers = [
            signature in targeted_signatures for signature in consumer_signatures
        ]
        if any(targeted_consumers) and not all(targeted_consumers):
            raise ValueError(
                f'NKF initializer Identity {node.name or output_name} mixes targeted and '
                f'untargeted consumers: {consumer_signatures}')
        if not any(targeted_consumers):
            retained_identities.append({
                'node': node.name or output_name,
                'source': source_name,
                'output': output_name,
                'shape': list(shape),
                'consumers': [
                    {'op_type': op_type, 'input_index': input_index}
                    for op_type, input_index, _ in consumer_signatures
                ],
            })
            continue
        observed.update(consumer_signatures)
        identities.append(node)
        rewrites[output_name] = source_name

    if not identities:
        raise ValueError('No initializer Identity aliases found; input appears already rewritten.')
    if len(identities) != expected_identity_count:
        raise ValueError(
            f'Expected {expected_identity_count} NKF initializer aliases, found {len(identities)}.')
    if observed != expected_observed:
        raise ValueError(
            'NKF initializer Identity topology mismatch.\n'
            f'Expected: {dict(expected_observed)}\nObserved: {dict(observed)}')

    identity_ids = {id(node) for node in identities}
    retained_nodes = []
    rewired_inputs = 0
    for node in model.graph.node:
        if id(node) in identity_ids:
            continue
        for input_index, input_name in enumerate(node.input):
            replacement = rewrites.get(input_name)
            if replacement is not None:
                node.input[input_index] = replacement
                rewired_inputs += 1
        retained_nodes.append(node)
    expected_rewired_inputs = sum(expected_observed.values())
    if rewired_inputs != expected_rewired_inputs:
        raise AssertionError(
            f'Expected {expected_rewired_inputs} rewired edges, got {rewired_inputs}.')

    before_nodes = len(model.graph.node)
    before_initializers = len(model.graph.initializer)
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    if _interface_signature(model) != original_interface:
        raise ValueError('Graph interface or opset imports changed during NKF rewrite.')
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=final_model_path.name + '.', suffix='.tmp',
        dir=final_model_path.parent, delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        onnx.save_model(model, str(temporary_path))
        temporary_path.replace(final_model_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    report = {
        'raw_model': str(raw_model_path),
        'final_model': str(final_model_path),
        'raw_sha256': _model_sha256(raw_model_path),
        'final_sha256': _model_sha256(final_model_path),
        'raw_nodes': before_nodes,
        'final_nodes': len(model.graph.node),
        'raw_initializers': before_initializers,
        'final_initializers': len(model.graph.initializer),
        'raw_op_histogram': before_histogram,
        'final_op_histogram': _op_histogram(model),
        'nkf_initializer_identity_rewrites': len(identities),
        'retained_untargeted_initializer_identities': retained_identities,
        'rewired_inputs': rewired_inputs,
        'deleted_nodes': [node.name or node.output[0] for node in identities],
        'deleted_initializers': [],
        'interface': original_interface,
        'custom_domains_added': [],
        'opset_imports_changed': False,
        'static_profile': {
            'frames': frames,
            'model_batch': model_batch,
            'n_freq': n_freq,
            'filter_order': filter_order,
            'fc_dim': fc_dim,
            'rnn_dim': rnn_dim,
        },
        'observed_preconditions': {
            f'{op_type}:input_{input_index}:shape_{shape}': count
            for (op_type, input_index, shape), count in sorted(observed.items())
        },
    }
    if report_path is not None:
        report_path = Path(report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_model", nargs="?", default=script_dir / "DFSMN_AEC_ONNX" / "DFSMN_AEC.raw.onnx")
    parser.add_argument(
        "final_model", nargs="?", default=script_dir / "DFSMN_AEC_ONNX" / "DFSMN_AEC.onnx")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report_path = args.report or Path(args.final_model).with_suffix(".rewrite_report.json")
    report = rewrite_model(args.raw_model, args.final_model, report_path)
    print(json.dumps({
        "raw_nodes": report["raw_nodes"],
        "final_nodes": report["final_nodes"],
        "lstm_rewrites": len(report["lstm_zero_state_rewrites"]),
        "conv_padding_rewrites": len(report["asymmetric_conv_padding_rewrites"]),
        "deleted_nodes": len(report["deleted_nodes"]),
        "deleted_initializers": report["deleted_initializers"],
        "report": str(Path(report_path).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
