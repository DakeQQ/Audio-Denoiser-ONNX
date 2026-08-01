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
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "MossFormer_Optimized_F16")

ENABLE_FP16 = True   # Mixed FP16/FP32: CUDA-friendly while protecting the wide fbank and FSMN paths.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Select the numerically wide paths that must remain in FP32.

    Float inputs use int16-scale PCM, so the fused DFT Conv and the raw STFT branch can
    exceed the FP16 finite range. Squaring and accumulating the low-level fbank bins also
    needs FP32 for quiet-audio dynamic range. The trained FSMN gates intentionally create
    very large temporary values immediately before ``norm2``; retain each complete
    gate/memory path in FP32 and cast its normalised result back to FP16. Select all regions
    structurally so harmless exporter name changes cannot disable protection.
    """
    # run_optimizer resolves this selector after pass-1 onnxslim has written the
    # destination model. Select that exact graph so generated MatMul reshape shells
    # and fused transpose tokens inherit the surrounding FP32 body.
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

    split_node = nodes[frontend_split]
    frontend_conv = producers.get(split_node.input[0])
    if frontend_conv is None or frontend_conv.op_type != "Conv" or len(split_node.output) != 2:
        raise RuntimeError("Could not identify the MossFormer2 SE fused spectral frontend")

    protected = [frontend_conv.name, split_node.name]
    protected.extend(
        node.name
        for node in nodes[frontend_split + 1:first_log + 1]
        if node.name and node.op_type not in {"Constant", "Reshape"}
    )
    log_output = nodes[first_log].output[0]
    for node in nodes[first_log + 1:]:
        if log_output in node.input and node.op_type == "Add" and node.name:
            protected.append(node.name)
            break

    graph_output = next(
        (value.name for value in graph.output if value.name == "denoised_audio"),
        None,
    )

    def path_from_ancestor(value: str, ancestor: str) -> list[onnx.NodeProto] | None:
        if value == ancestor:
            return []
        producer = producers.get(value)
        if producer is None:
            return None
        for input_value in producer.input:
            path = path_from_ancestor(input_value, ancestor)
            if path is not None:
                return path + [producer]
        return None

    stft_path = (
        path_from_ancestor(graph_output, split_node.output[1])
        if graph_output is not None
        else None
    )
    expected_stft_ops = {
        ("Reshape", "Mul", "Reshape", "ConvTranspose", "Slice", "Div", "Resize"),
        ("Reshape", "Mul", "Reshape", "ConvTranspose", "Div", "Resize"),
    }
    if stft_path is None or tuple(node.op_type for node in stft_path) not in expected_stft_ops:
        raise RuntimeError(
            "Unexpected MossFormer2 SE raw-STFT output path; refusing unsafe FP16: "
            f"ops={[node.op_type for node in stft_path] if stft_path else None}"
        )
    protected.extend(node.name for node in stft_path if node.name)

    # The nearest LayerNormalization before each ``norm2`` is the affine-free shared
    # normalisation feeding a fused u/v projection. Keep one continuous FP32 region from
    # the first such normalization through the final norm2, including the intervening
    # FLASH layers. This avoids 23 F32 -> F16 -> F32 layer sandwiches.
    previous_norm = None
    fp32_body_start = None
    fp32_body_end = None
    for index, node in enumerate(nodes):
        if node.op_type != "LayerNormalization":
            continue
        if any(".norm2.weight" in value for value in node.input):
            if previous_norm is None:
                raise RuntimeError(f"Could not identify the FSMN input normalization for {node.name}")
            if fp32_body_start is None:
                fp32_body_start = previous_norm
            fp32_body_end = index
        previous_norm = index

    if fp32_body_start is None or fp32_body_end is None:
        raise RuntimeError("Could not identify the MossFormer2 SE continuous FP32 body")
    protected.extend(
        candidate.name
        for candidate in nodes[fp32_body_start:fp32_body_end + 1]
        if candidate.name and candidate.op_type != "Constant"
    )

    return protected


MODEL_PLANS = {
    "MossFormer2_SE_48K": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=8,
        hidden_size=512,
        # Preserve source node names until after precision conversion. ORT level 1
        # creates fused Transpose tokens that fragment the continuous FP32 body.
        opt_level=0,
        only_onnxruntime=True,
        f16_node_block_list=_fp16_sensitive_nodes,
        # The initial pre-body ScaleNorm remains FP16. ScaleNorms inside the merged body
        # inherit FP32 so they do not create isolated reciprocal casts.
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


