"""Optimize the exported ZipEnhancer ONNX model."""

from pathlib import Path
import sys

import onnx


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "ZipEnhancer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "ZipEnhancer_Optimized_F16")

ENABLE_FP16 = True      # Mixed FP16/FP32 CUDA graph;
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Select two contiguous FP32 regions around the FP16 Zipformer body.

    Raw-amplitude RMS/STFT arithmetic and dense-block InstanceNormalization need
    FP32 before the first Zipformer projection. Decoder InstanceNormalization,
    phase reconstruction, ISTFT, RMS restoration and sanitization also need FP32.
    Keeping each side contiguous avoids local F32 -> F16 -> F32 cast sandwiches.
    """

    simplified_path = Path(OPTIMIZED_FOLDER_PATH) / Path(src_path).name
    selection_path = simplified_path if simplified_path.exists() else Path(src_path)
    graph = onnx.load(selection_path, load_external_data=False).graph
    nodes = list(graph.node)
    producers = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for input_value in node.input:
            consumers.setdefault(input_value, []).append(node)

    normalizations = [node for node in nodes if node.op_type == "InstanceNormalization"]
    if len(normalizations) != 11:
        raise RuntimeError(
            "Unexpected ZipEnhancer InstanceNormalization topology; refusing "
            f"fragmented FP16 conversion: count={len(normalizations)}"
        )

    encoder_norms = normalizations[:6]
    decoder_norms = normalizations[6:]
    encoder_exit = [
        node
        for node in consumers.get(encoder_norms[-1].output[0], [])
        if node.op_type == "PRelu"
    ]
    decoder_entry = producers.get(decoder_norms[0].input[0])
    if len(encoder_exit) != 1 or decoder_entry is None or decoder_entry.op_type != "Conv":
        raise RuntimeError(
            "Could not identify ZipEnhancer FP32 region boundaries: "
            f"encoder_exit={[node.op_type for node in encoder_exit]}, "
            f"decoder_entry={decoder_entry.op_type if decoder_entry else None}"
        )

    node_indices = {id(node): index for index, node in enumerate(nodes)}
    encoder_end = node_indices[id(encoder_exit[0])]
    decoder_start = node_indices[id(decoder_entry)]
    if encoder_end >= decoder_start:
        raise RuntimeError(
            "Unexpected ZipEnhancer FP32 region ordering; refusing unsafe FP16: "
            f"encoder_end={encoder_end}, decoder_start={decoder_start}"
        )

    output_names = {value.name for value in graph.output}
    decoder_outputs = {
        value
        for node in nodes[decoder_start:]
        for value in node.output
    }
    if not output_names.issubset(decoder_outputs):
        raise RuntimeError("ZipEnhancer decoder FP32 region does not reach every graph output")

    protected = [
        node.name
        for node in nodes[:encoder_end + 1]
        if node.name and node.op_type != "Constant"
    ]
    protected.extend(
        node.name
        for node in nodes[decoder_start:]
        if node.name and node.op_type != "Constant"
    )
    print(
        "  Keeping contiguous ZipEnhancer frontend/decoder regions in FP32: "
        f"{encoder_end + 1} frontend nodes, {len(nodes) - decoder_start} decoder nodes"
    )
    return protected

# InstanceNormalization is covered by the two contiguous node regions above.
FP16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]


MODEL_PLANS = {
    "ZipEnhancer": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=4,
        hidden_size=112,
        # Preserve pass-1 node boundaries until after precision conversion.
        opt_level=0,
        only_onnxruntime=True,
        first_slim_no_shape_infer="auto",
        second_slim_no_shape_infer="auto",
        fp16_symbolic_shape_infer="auto",
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
        f16_op_block_list=FP16_OP_BLOCK_LIST,
    ),
}


CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    # Initializers used only by blocked FP32 nodes must stay FP32. Forcing every
    # initializer through FP16 changes 1e-9 -> 1.19e-7 and 0.15 -> 0.150024,
    # which visibly perturbs sparse/silent STFT bins despite casting back.
    f16_force_initializers=False,
)


def _collapse_sanitization_casts(model_path: Path) -> None:
    """Remove converter hops before float-compatible IsInf checks."""

    model = onnx.load(model_path, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for values in (inferred.graph.input, inferred.graph.output, inferred.graph.value_info)
        for value in values
    }
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_value in node.input:
            consumers.setdefault(input_value, []).append(node)

    removed: set[int] = set()
    collapsed = 0
    for is_inf in (node for node in model.graph.node if node.op_type == "IsInf"):
        double_cast = producers.get(is_inf.input[0])
        fp16_cast = producers.get(double_cast.input[0]) if double_cast is not None else None
        if (
            double_cast is None
            or double_cast.op_type != "Cast"
            or fp16_cast is None
            or fp16_cast.op_type != "Cast"
        ):
            continue
        double_to = next((attr.i for attr in double_cast.attribute if attr.name == "to"), None)
        fp16_to = next((attr.i for attr in fp16_cast.attribute if attr.name == "to"), None)
        source_value = fp16_cast.input[0]
        if (
            double_to != onnx.TensorProto.DOUBLE
            or fp16_to != onnx.TensorProto.FLOAT16
            or value_types.get(source_value) != onnx.TensorProto.FLOAT
            or consumers.get(fp16_cast.output[0]) != [double_cast]
            or consumers.get(double_cast.output[0]) != [is_inf]
        ):
            continue
        is_inf.input[0] = source_value
        removed.update((id(fp16_cast), id(double_cast)))
        collapsed += 1

    if collapsed != 2:
        raise RuntimeError(
            "Unexpected ZipEnhancer sanitization cast topology; refusing rewrite: "
            f"collapsed={collapsed}"
        )
    kept_nodes = [node for node in model.graph.node if id(node) not in removed]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    print("  Removed 2 redundant F32 -> F16 -> F64 sanitization cast chains.")


if __name__ == "__main__":
    run_optimizer(CONFIG)
    if ENABLE_FP16:
        _collapse_sanitization_casts(Path(OPTIMIZED_FOLDER_PATH) / "ZipEnhancer.onnx")

