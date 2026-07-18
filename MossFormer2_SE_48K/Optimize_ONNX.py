"""Optimize the exported MossFormer2 SE 48K ONNX model."""

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


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "MossFormer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "MossFormer_Optimized")

ENABLE_FP16 = False   # Mixed FP16/FP32: CUDA-friendly while protecting the wide fbank and FSMN paths.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Select the numerically wide paths that must remain in FP32.

    The exporter normalises PCM before the frontend, so both Conv outputs fit in FP16.
    Squaring and accumulating the low-level fbank bins still needs FP32 for quiet-audio
    dynamic range. The trained FSMN gates also intentionally create very large temporary
    values (up to billions for this checkpoint) immediately before ``norm2``; retain each
    complete gate/memory path in FP32 and cast its normalised result back to FP16. Select
    both regions structurally so harmless exporter name changes cannot disable protection.
    """
    graph = onnx.load(src_path, load_external_data=False).graph
    nodes = list(graph.node)
    frontend_split = next(
        (index for index, node in enumerate(nodes) if node.op_type == "Split"),
        None,
    )
    first_log = next(
        (index for index, node in enumerate(nodes) if node.op_type == "Log"),
        None,
    )
    if frontend_split is None or first_log is None or frontend_split >= first_log:
        raise RuntimeError("Could not identify the MossFormer2 SE log-fbank frontend path")

    protected = [
        node.name
        for node in nodes[frontend_split + 1:first_log + 1]
        if node.name and node.op_type not in {"Constant", "Reshape"}
    ]
    log_output = nodes[first_log].output[0]
    for node in nodes[first_log + 1:]:
        if log_output in node.input and node.op_type == "Add" and node.name:
            protected.append(node.name)
            break

    # For each FSMN block, the nearest LayerNormalization before ``norm2`` is the
    # affine-free shared normalisation feeding the fused u/v projection. Everything from
    # there through norm2 must use FP32; norm2 reduces the wide gate output back to the
    # small range consumed by the next FLASH layer.
    previous_norm = None
    for index, node in enumerate(nodes):
        if node.op_type != "LayerNormalization":
            continue
        if any(".norm2.weight" in value for value in node.input):
            if previous_norm is None:
                raise RuntimeError(f"Could not identify the FSMN input normalization for {node.name}")
            protected.extend(
                candidate.name
                for candidate in nodes[previous_norm:index + 1]
                if candidate.name and candidate.op_type != "Constant"
            )
        previous_norm = index

    # ORT rewrites rank-3 MatMul nodes as Reshape -> Gemm -> Reshape while preserving
    # the original MatMul name on the Gemm. Without selecting the generated Reshapes,
    # each one becomes an FP16 island inside an otherwise contiguous FP32 FSMN path:
    # FP32 -> FP16 -> Reshape -> FP32. Promote both Reshapes so onnxslim can remove
    # those reciprocal casts. This removes 192 dtype switches without moving any
    # compute-heavy FLASH operation out of FP16.
    protected_set = set(protected)
    protected.extend(
        generated_name
        for node in nodes
        if node.name in protected_set and node.op_type == "MatMul"
        for generated_name in (f"{node.name}_pre_reshape", f"{node.name}_post_reshape")
    )

    return protected


MODEL_PLANS = {
    "MossFormer2_SE_48K": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=8,
        hidden_size=512,
        opt_level=1,
        only_onnxruntime=True,
        f16_node_block_list=_fp16_sensitive_nodes,
        # ReduceL2 remains FP16: all 48 ScaleNorm reductions stayed finite on CUDA
        # (maximum 2,003 across representative and adversarial full-scale inputs), and
        # keeping them FP32 would add 96 isolated FP16<->FP32 switches.
        f16_op_block_list=["Range", "ReduceMean", "ReduceSum", "Sqrt"],
    ),
}


CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    f16_force_initializers=False,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)


