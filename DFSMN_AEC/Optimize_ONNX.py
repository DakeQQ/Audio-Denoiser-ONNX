"""Optimize the exported DFSMN AEC ONNX model."""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "DFSMN_AEC_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "DFSMN_AEC_Optimized")

ENABLE_FP16 = False   # Mixed FP16/FP32 CUDA graph; see FP16_CUDA_REPORT.md.
UPGRADE_OPSET = 0


def _cast_target(node: onnx.NodeProto) -> int | None:
    return next(
        (
            helper.get_attribute_value(attribute)
            for attribute in node.attribute
            if attribute.name == "to"
        ),
        None,
    )


def _constant_scalars(graph: onnx.GraphProto) -> dict[str, float]:
    values: dict[str, float] = {}
    for tensor in graph.initializer:
        array = np.asarray(numpy_helper.to_array(tensor))
        if array.size == 1:
            values[tensor.name] = float(array.reshape(()))
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attribute in node.attribute:
            value = helper.get_attribute_value(attribute)
            if isinstance(value, onnx.TensorProto):
                array = np.asarray(numpy_helper.to_array(value))
                if array.size == 1:
                    values[node.output[0]] = float(array.reshape(()))
                    break
            if isinstance(value, (float, int)):
                values[node.output[0]] = float(value)
                break
    for node in graph.node:
        if (
            node.op_type == "Identity"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.input[0] in values
        ):
            values[node.output[0]] = values[node.input[0]]
    return values


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Validate scale safety and select two complete FP32 regions.

    All three light-AEC backends operate on normalized waveforms with unit-scale
    STFT/COLA tables. They are individually FP16-safe, but in this composite model
    their small waveform error is strongly amplified by the downstream
    cancellation-sensitive ``near - 1.15 * temporary`` fbank and Log. Consequently
    the light-AEC plus fbank-to-Log frontend is one contiguous FP32 region. The much
    larger DFSMN mask network and mask STFT/ISTFT remain FP16. The terminal PCM
    scale and Clip form a second FP32 region because 32767 rounds to 32768 in FP16.
    """
    model = onnx.load(src_path, load_external_data=False)
    graph = model.graph
    nodes = list(graph.node)
    metadata = {item.key: item.value for item in model.metadata_props}
    backend = metadata.get("light_aec_model", "")
    if backend not in {"SDAEC", "Deep_Echo", "NKF"}:
        raise RuntimeError(
            f"Unsupported or missing DFSMN_AEC light_aec_model metadata: {backend!r}"
        )

    initializers = {
        tensor.name: np.asarray(numpy_helper.to_array(tensor))
        for tensor in graph.initializer
    }
    scalar_values = _constant_scalars(graph)
    expected_scalars = {
        "input_pcm_scale": 1.0 / 32768.0,
        "fbank_power_scale": 32768.0 * 32768.0,
        "output_pcm_scale": 32767.0,
    }
    for name, expected in expected_scalars.items():
        value = initializers.get(name)
        if value is None or value.size != 1 or float(value.reshape(())) != expected:
            raise RuntimeError(
                f"Unsafe DFSMN_AEC FP16 source: expected {name}={expected}, got {value}"
            )

    stft_tables = {
        name: value
        for name, value in initializers.items()
        if name.endswith("stft_kernel")
        and ("custom_stft_A2" in name or "custom_stft_B" in name or "light_aec.custom_stft" in name)
    }
    cola_tables = {
        name: value
        for name, value in initializers.items()
        if name.endswith("inv_win_sum")
    }
    if len(stft_tables) < 2 or len(cola_tables) < 2:
        raise RuntimeError(
            "Could not identify both backend and mask STFT/COLA tables in DFSMN_AEC"
        )
    for name, table in stft_tables.items():
        maximum = float(np.max(np.abs(table)))
        if not np.isfinite(maximum) or not 0.5 <= maximum <= 2.0:
            raise RuntimeError(
                f"Unsafe FP16 STFT table {name}: max abs={maximum:.8g}; "
                "re-export with explicit PCM input scaling"
            )
    for name, table in cola_tables.items():
        maximum = float(np.max(np.abs(table)))
        if not np.isfinite(maximum) or maximum > 256.0:
            raise RuntimeError(
                f"Unsafe FP16 COLA table {name}: max abs={maximum:.8g}; "
                "re-export with output_scale=1.0"
            )

    producers = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for value in node.input:
            consumers.setdefault(value, []).append(node)

    input_scale_users = consumers.get("input_pcm_scale", [])
    if len(input_scale_users) != 1 or input_scale_users[0].op_type != "Mul":
        raise RuntimeError(
            "Unsafe DFSMN_AEC input topology; expected one explicit input_pcm_scale Mul"
        )
    input_scale_node = input_scale_users[0]
    input_scale_data = next(
        (value for value in input_scale_node.input if value != "input_pcm_scale"),
        "",
    )
    input_cast = producers.get(input_scale_data)
    if (
        input_cast is None
        or input_cast.op_type != "Cast"
        or _cast_target(input_cast) != TensorProto.FLOAT
    ):
        raise RuntimeError(
            "Unsafe DFSMN_AEC input topology; expected Cast(FLOAT) -> input scale Mul"
        )

    mask_stft = next(
        (
            node
            for node in nodes
            if node.op_type == "Conv" and "custom_stft_A2.stft_kernel" in node.input
        ),
        None,
    )
    backend_output = (
        producers.get(mask_stft.input[0])
        if mask_stft is not None and mask_stft.input
        else None
    )
    if (
        mask_stft is None
        or backend_output is None
        or backend_output.op_type != "Mul"
        or not any(value.endswith("inv_win_sum") for value in backend_output.input)
    ):
        raise RuntimeError(
            "Could not identify the light-AEC ISTFT output feeding the mask STFT"
        )

    frontend_start = nodes.index(input_cast)
    backend_end = nodes.index(backend_output)
    if backend_end <= frontend_start:
        raise RuntimeError("Unexpected light-AEC frontend node order")
    backend_nodes = nodes[frontend_start:backend_end + 1]

    fbank_conv = next(
        (
            node
            for node in nodes
            if node.op_type == "Conv" and "fbank_conv_weight" in node.input
        ),
        None,
    )
    if fbank_conv is None or len(fbank_conv.output) != 1:
        raise RuntimeError("Could not identify the normalized-domain fbank Conv")

    # Absorb the near/temp layout path before the fbank Conv. Without this
    # expansion, Cast(FP32->FP16) -> Slice/Concat -> Cast(FP16->FP32) is a tiny
    # sandwiched island that adds latency and precision loss for no useful gain.
    backend_node_ids = {id(node) for node in backend_nodes}
    fbank_input_nodes: list[onnx.NodeProto] = []
    pending_values = [fbank_conv.input[0]]
    while pending_values:
        value = pending_values.pop()
        producer = producers.get(value)
        if producer is None or id(producer) in backend_node_ids:
            continue
        if producer is mask_stft:
            raise RuntimeError("The fbank input path unexpectedly traverses the mask STFT")
        if producer not in fbank_input_nodes:
            fbank_input_nodes.append(producer)
            pending_values.extend(producer.input)
    fbank_input_nodes.sort(key=nodes.index)

    # Follow every fbank data descendant until Log so branch/layout nodes remain
    # part of the same frontend rather than creating FP16 cast sandwiches.
    reachable = set(fbank_conv.output)
    fbank_nodes: list[onnx.NodeProto] = [fbank_conv]
    reached_log = False
    for node in nodes[nodes.index(fbank_conv) + 1:]:
        if not any(value in reachable for value in node.input):
            continue
        fbank_nodes.append(node)
        reachable.update(value for value in node.output if value)
        if node.op_type == "Log":
            reached_log = True
            break
    if not reached_log or not fbank_nodes:
        raise RuntimeError("Could not identify the complete DFSMN_AEC fbank-to-Log path")
    if not any("fbank_power_scale" in node.input for node in fbank_nodes):
        raise RuntimeError("The fbank path does not restore int16-domain power")
    allowed_fbank_ops = {
        "Conv", "Reshape", "Transpose", "Split", "Mul", "Sub", "Concat",
        "ReduceSum", "MatMul", "Clip", "Log",
    }
    unexpected = {
        node.op_type for node in fbank_nodes if node.op_type not in allowed_fbank_ops
    }
    if unexpected:
        raise RuntimeError(f"Unexpected operators in the fbank FP32 island: {unexpected}")

    # Deep Echo's two cepstral sum-of-squares reductions are source-scaled by
    # 1/4 (epsilon by 1/16). The composite frontend remains FP32 for downstream
    # quality, but validating this source invariant prevents reintroducing the
    # overflow if the backend policy is narrowed later.
    if backend == "Deep_Echo":
        ceps_prefixes = (
            "/light_aec/cfb_e1/ceps_unit/LN/",
            "/light_aec/cfb_d1/ceps_unit/LN/",
        )
        for prefix in ceps_prefixes:
            prefixed = [node for node in nodes if node.name.startswith(prefix)]
            reductions = [node for node in prefixed if node.op_type == "ReduceSum"]
            quarter_scales = [
                node
                for node in prefixed
                if node.op_type == "Mul"
                and any(
                    value in scalar_values and np.isclose(scalar_values[value], 0.25)
                    for value in node.input
                )
            ]
            if len(reductions) != 1 or len(quarter_scales) != 1:
                raise RuntimeError(
                    f"Could not verify Deep Echo quarter-scaled reduction: {prefix}"
                )

    output_cast = producers.get(graph.output[0].name)
    output_clip = producers.get(output_cast.input[0]) if output_cast is not None else None
    output_layout_nodes: list[onnx.NodeProto] = []
    output_scale = None
    output_value = output_clip.input[0] if output_clip is not None and output_clip.input else ""
    while output_value:
        producer = producers.get(output_value)
        if producer is None:
            break
        if producer.op_type in {"Reshape", "Transpose", "Squeeze", "Unsqueeze"}:
            output_layout_nodes.append(producer)
            output_value = producer.input[0]
            continue
        output_scale = producer
        break
    if (
        output_cast is None
        or output_cast.op_type != "Cast"
        or _cast_target(output_cast) != TensorProto.INT16
        or output_clip is None
        or output_clip.op_type != "Clip"
        or output_scale is None
        or output_scale.op_type != "Mul"
        or "output_pcm_scale" not in output_scale.input
        or len(output_clip.input) < 3
        or output_clip.input[1] not in scalar_values
        or output_clip.input[2] not in scalar_values
        or scalar_values[output_clip.input[1]] != -32768.0
        or scalar_values[output_clip.input[2]] != 32767.0
    ):
        raise RuntimeError(
            "Unexpected output topology; expected Mul(output_pcm_scale) -> Clip -> Cast(INT16)"
        )

    protected_nodes = [
        *backend_nodes,
        *fbank_input_nodes,
        *fbank_nodes,
        output_scale,
        *reversed(output_layout_nodes),
        output_clip,
    ]
    protected = list(dict.fromkeys(node.name for node in protected_nodes if node.name))
    if len(protected) != len({id(node) for node in protected_nodes}):
        raise RuntimeError("DFSMN_AEC FP16 safety nodes require stable non-empty names")

    stft_max = max(float(np.max(np.abs(value))) for value in stft_tables.values())
    cola_max = max(float(np.max(np.abs(value))) for value in cola_tables.values())
    print(
        f"  Verified {backend} scale-safe source (STFT max={stft_max:.6g}, "
        f"COLA max={cola_max:.6g}); retaining frontend={len(backend_nodes) + len(fbank_input_nodes) + len(fbank_nodes)} nodes "
        f"and output={2 + len(output_layout_nodes)} nodes in FP32"
    )
    return protected


MODEL_PLANS = {
    "DFSMN_AEC": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=0,
        hidden_size=0,
        # Preserve the source node names and the contiguous FP32 frontend.
        # Runtime ORT still performs dtype-aware graph optimization on load.
        opt_level=0,
        only_onnxruntime=True,
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
        f16_op_block_list=["Range"],
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    f16_keep_io_types=True,
    # Keep real FP32 constants owned by the two safety islands. In particular,
    # 2**30 and 32767 must not be quantized to FP16 and cast back afterward.
    f16_force_initializers=False,
    # MatMul/Gemm fusion inserts Reshape nodes after node selection and fragments
    # a contiguous blocked region into dozens of one-node FP16 sandwiches. Keep
    # constant folding, dead-node elimination, CSE, and weight tying enabled;
    # disable only graph fusion before and after dtype conversion.
    slim_skip_optimizations=["graph_fusion"],
)

if __name__ == "__main__":
    run_optimizer(CONFIG)
