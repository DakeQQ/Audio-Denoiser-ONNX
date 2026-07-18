"""Optimize the exported ZipEnhancer ONNX model."""

from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists() and (_candidate / "audio_onnx_metadata.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "ZipEnhancer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "ZipEnhancer_Optimized")

ENABLE_FP16 = False      # Mixed FP16/FP32 CUDA graph;
UPGRADE_OPSET = 0


# A blanket FP16 conversion changes the INT16 input Cast to FP16, so the first
# audio * audio overflows for |sample| > sqrt(65504). Keep the complete input RMS
# path and final waveform rescale/sanitize path in FP32. Node names are from the
# checked static ZipEnhancer export and remain stable through the first slim/ORT
# optimization passes.
FP16_NODE_BLOCK_LIST = [
    "/Cast",
    "/Reshape",
    "/Mul",
    "/ReduceMean",
    "/Add",
    "/Sqrt",
    "/Div",
    # Sparse/zero-padded windows can normalize an STFT coefficient above
    # sqrt(65504). Compute real^2 + imag^2 and magnitude compression in FP32.
    "/Mul_1",
    "/Mul_2",
    "/Add_1",
    "/Add_2",
    "/Pow",
    "/Mul_9",
    "/Reshape_41",
    "/IsNaN",
    "/Where_4",
    "/Clip",
    "/Cast_1",
]

# Dense encoder/decoder activations can exceed 20,000 before normalization.
# CUDA's true FP16 InstanceNormalization variance path then overflows or loses
# enough precision to collapse output quality. Keep only these reductions FP32;
# Conv/Gemm/attention and the rest of the network remain FP16.
FP16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
    "InstanceNormalization",
]


MODEL_PLANS = {
    "ZipEnhancer": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=4,
        hidden_size=112,
        opt_level=2,
        first_slim_no_shape_infer="auto",
        second_slim_no_shape_infer="auto",
        fp16_symbolic_shape_infer="auto",
        f16_node_block_list=FP16_NODE_BLOCK_LIST,
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


def _validate_fp16_source_contract() -> None:
    """Fail closed if an exporter change invalidates an exact node exclusion."""
    if not ENABLE_FP16:
        return
    source_path = Path(ORIGINAL_FOLDER_PATH) / "ZipEnhancer.onnx"
    if not source_path.is_file():
        return  # Preserve the common optimizer's skip-missing behavior.

    import onnx

    expected_ops = {
        "/Cast": "Cast",
        "/Reshape": "Reshape",
        "/Mul": "Mul",
        "/ReduceMean": "ReduceMean",
        "/Add": "Add",
        "/Sqrt": "Sqrt",
        "/Div": "Div",
        "/Mul_1": "Mul",
        "/Mul_2": "Mul",
        "/Add_1": "Add",
        "/Add_2": "Add",
        "/Pow": "Pow",
        "/Mul_9": "Mul",
        "/Reshape_41": "Reshape",
        "/IsNaN": "IsNaN",
        "/Where_4": "Where",
        "/Clip": "Clip",
        "/Cast_1": "Cast",
    }
    if set(expected_ops) != set(FP16_NODE_BLOCK_LIST):
        raise RuntimeError("FP16 safety-node validation is out of sync with its block list.")
    nodes = {node.name: node.op_type for node in onnx.load(str(source_path)).graph.node}
    mismatches = {
        name: {"expected": op_type, "actual": nodes.get(name)}
        for name, op_type in expected_ops.items()
        if nodes.get(name) != op_type
    }
    if mismatches:
        raise RuntimeError(
            "ZipEnhancer FP16 safety-node contract changed; refusing an unsafe "
            f"conversion: {mismatches}"
        )


if __name__ == "__main__":
    _validate_fp16_source_contract()
    run_optimizer(CONFIG)

