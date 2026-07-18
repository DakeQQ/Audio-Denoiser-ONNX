"""Optimize the exported H-GTCRN ONNX model."""

from collections import defaultdict, deque
from pathlib import Path
import sys

import onnx
from onnx import TensorProto, helper

_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer

ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "H_GTCRN_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "H_GTCRN_Optimized")

ENABLE_FP16 = False   # Mixed FP16/FP32 CUDA graph; see FP16_CUDA_Test/README.md.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Keep numerically wide chains in FP32 without precision sandwiches.

    Naive FP16 fails independently in WPE, AuxIVA and the GTCRN temporal
    recurrent-attention (TRA) energy paths. Protect each complete chain instead
    of individual reduction nodes so the converter creates only genuine region
    boundaries, not FP32 -> FP16 -> FP32 islands around layout-only operators.
    """
    graph = onnx.load(src_path, load_external_data=False).graph
    nodes = list(graph.node)
    producers = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(index)

    core_prefix = "/gtcrn/"
    frontend_seeds: set[int] = set()
    for node in nodes:
        if not node.name.startswith(core_prefix):
            continue
        for input_name in node.input:
            producer_index = producers.get(input_name)
            if producer_index is not None and not nodes[producer_index].name.startswith(core_prefix):
                frontend_seeds.add(producer_index)
    if not frontend_seeds:
        raise RuntimeError("Could not identify the H-GTCRN frontend-to-network boundary")

    # Walk backwards from every GTCRN feature input. This retains input
    # normalization, STFT, WPE, AuxIVA, source selection and feature assembly as
    # one contiguous FP32 region. WPE/AuxIVA are iterative complex solvers and
    # both independently produce NaN when converted to FP16.
    frontend_indices: set[int] = set()
    queue = deque(frontend_seeds)
    while queue:
        index = queue.popleft()
        if index in frontend_indices:
            continue
        frontend_indices.add(index)
        for input_name in nodes[index].input:
            producer_index = producers.get(input_name)
            if producer_index is not None:
                queue.append(producer_index)

    frontend_names = {
        nodes[index].name
        for index in frontend_indices
        if nodes[index].name
    }
    for required_prefix in ("/stft_model/", "/wpe/", "/iva/"):
        if not any(name.startswith(required_prefix) for name in frontend_names):
            raise RuntimeError(f"H-GTCRN FP16 guard did not capture {required_prefix}")

    # onnxslim inserts these two layout-only transposes in the protected feature
    # assembly chain. Selecting them closes two FP32 -> FP16 -> Transpose -> FP32
    # sandwiches; unknown names in a block list are harmless on other versions.
    protected = frontend_names | {"Transpose_token_10", "Transpose_token_11"}

    # TRA computes x^2 before a mean and GRU. These energy features become wide
    # (measured up to 44,048), and the final decoder TRA GRU is the first node that
    # produces NaN under CUDA FP16 for quiet speech and full-scale tones. Keep all
    # six complete square -> mean -> GRU -> gate chains in FP32, with one cast at
    # entry and one at exit per block.
    tra_nodes = [
        node
        for node in nodes
        if node.name.startswith(core_prefix)
        and "/tra/" in node.name
        and node.op_type != "Constant"
    ]
    if len([node for node in tra_nodes if node.op_type == "GRU"]) != 6:
        raise RuntimeError("Expected six H-GTCRN TRA GRU paths")
    protected.update(node.name for node in tra_nodes if node.name)

    # ORT rewrites each rank-3 TRA MatMul to Reshape -> Gemm -> Reshape. Promote
    # those generated shape-only nodes into the surrounding FP32 chain so they do
    # not create reciprocal cast pairs.
    for node in tra_nodes:
        if node.op_type == "MatMul" and node.name:
            protected.update((f"{node.name}_pre_reshape", f"{node.name}_post_reshape"))

    # Keep PCM scaling, NaN replacement and clipping in FP32. In FP16, +32767 is
    # rounded to +32768 and can wrap to -32768 in the final int16 Cast.
    output_scales = [
        index
        for index, node in enumerate(nodes)
        if node.op_type == "Mul" and "output_pcm_scale" in node.input
    ]
    if len(output_scales) != 1:
        raise RuntimeError("Could not identify the H-GTCRN output PCM scale")
    output_cast_index = producers.get(graph.output[0].name)
    if output_cast_index is None:
        raise RuntimeError("Could not identify the H-GTCRN output Cast")
    output_cast = nodes[output_cast_index]
    cast_to = next(
        (
            helper.get_attribute_value(attribute)
            for attribute in output_cast.attribute
            if attribute.name == "to"
        ),
        None,
    )
    if output_cast.op_type != "Cast" or cast_to != TensorProto.INT16:
        raise RuntimeError("Expected the H-GTCRN graph to end in Cast(INT16)")

    output_chain: set[int] = set()
    queue = deque(output_scales)
    while queue:
        index = queue.popleft()
        if index == output_cast_index or index in output_chain:
            continue
        output_chain.add(index)
        for output_name in nodes[index].output:
            queue.extend(consumers.get(output_name, ()))
    if not any(
        output_cast_index in consumers.get(output_name, ())
        for index in output_chain
        for output_name in nodes[index].output
    ):
        raise RuntimeError("H-GTCRN output FP32 chain does not reach Cast(INT16)")
    protected.update(
        nodes[index].name
        for index in output_chain
        if nodes[index].name
    )

    print(
        "  H-GTCRN FP16 guards: "
        f"frontend={len(frontend_names)}, TRA={len(tra_nodes)}, output={len(output_chain)} nodes"
    )
    return sorted(protected)


MODEL_PLANS = {
    "H_GTCRN": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=0,
        hidden_size=0,
        opt_level=1,
        only_onnxruntime=True,
        first_slim_no_shape_infer="auto",
        second_slim_no_shape_infer="auto",
        fp16_symbolic_shape_infer="auto",
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    # FP32-only constants must not be rounded to FP16 and cast back inside a
    # protected solver/TRA chain.
    f16_force_initializers=False,
)

if __name__ == "__main__":
    run_optimizer(CONFIG)

