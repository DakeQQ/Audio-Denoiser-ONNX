"""Optimize the exported FLA-TFLocoformer SS 8K ONNX model."""

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


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "FLA_TFLocoformer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "FLA_TFLocoformer_Optimized")

ENABLE_FP16 = False     # Mixed FP16/FP32 with the wide normalization and focusing paths guarded below.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Keep the PCM-scale variance normalization, waveform restoration and focusing powers in FP32.

    The input variance sums squared PCM-scale samples, the restored waveform is PCM scale
    before the final 1/32768, and ||q^3|| in focused linear attention exceeds 65504 for
    moderate activations.
    """

    simplified_path = Path(OPTIMIZED_FOLDER_PATH) / Path(src_path).name
    selection_path = simplified_path if simplified_path.exists() else Path(src_path)
    graph = onnx.load(selection_path, load_external_data=False).graph
    nodes = list(graph.node)
    stft_index = next((index for index, node in enumerate(nodes) if node.op_type == "Conv"), None)
    istft_index = next(
        (index for index in range(len(nodes) - 1, -1, -1) if nodes[index].op_type == "ConvTranspose"),
        None,
    )
    if stft_index is None or istft_index is None or stft_index >= istft_index:
        raise RuntimeError("Could not identify the FLA-TFLocoformer STFT / iSTFT boundaries")

    head_nodes = [node for node in nodes[:stft_index] if node.op_type != "Constant"]
    head_ops = {node.op_type for node in head_nodes}
    if not {"ReduceSum", "Sqrt", "Where"}.issubset(head_ops):
        raise RuntimeError(
            "Unexpected FLA-TFLocoformer input-normalization topology; refusing unsafe FP16: "
            f"ops={sorted(head_ops)}"
        )

    tail_nodes = [node for node in nodes[istft_index + 1:] if node.op_type != "Constant"]
    output_names = {value.name for value in graph.output}
    if not output_names.issubset({value for node in tail_nodes for value in node.output}):
        raise RuntimeError("Unexpected FLA-TFLocoformer output topology; refusing cast islands")

    pow_outputs = {node.output[0] for node in nodes if node.op_type == "Pow"}
    focus_nodes = [
        node for node in nodes
        if node.op_type == "Pow" or (node.op_type == "ReduceL2" and node.input[0] in pow_outputs)
    ]
    if not focus_nodes or len(focus_nodes) % 4 != 0:
        raise RuntimeError(
            "Unexpected FLA-TFLocoformer focusing topology; refusing unsafe FP16: "
            f"count={len(focus_nodes)}"
        )

    selected = head_nodes + tail_nodes + focus_nodes
    if any(not node.name for node in selected):
        raise RuntimeError("FP16 guard requires named ONNX nodes")
    return [node.name for node in selected]


MODEL_PLANS = {
    "FLA_TFLocoformer_SS_8K": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=4,
        hidden_size=128,
        # Preserve the selected nodes until precision conversion.
        opt_level=0,
        only_onnxruntime=True,
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
    ),
}


CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
