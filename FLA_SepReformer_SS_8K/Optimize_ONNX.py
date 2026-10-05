"""Optimize the exported FLA-SepReformer SS 8K ONNX model."""

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


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "FLA_SepReformer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "FLA_SepReformer_Optimized")

ENABLE_FP16 = False     # Mixed FP16/FP32 with the focused linear attention core and the waveform tail guarded below.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Keep the focused linear attention core and the waveform decoder / peak normalization in FP32.

    ||q^3|| exceeds 65504 for moderate activations, 1 / (q . mean(k) + 1e-6) overflows FP16 on
    near-silent frames, and the raw decoder output peaks far above full scale before the in-graph
    peak normalization. The q / kv / proj / dwc projections stay FP16.
    """

    simplified_path = Path(OPTIMIZED_FOLDER_PATH) / Path(src_path).name
    selection_path = simplified_path if simplified_path.exists() else Path(src_path)
    graph = onnx.load(selection_path, load_external_data=False).graph
    nodes = list(graph.node)
    decoder_index = next(
        (index for index in range(len(nodes) - 1, -1, -1) if nodes[index].op_type == "ConvTranspose"),
        None,
    )
    if decoder_index is None:
        raise RuntimeError("Could not identify the FLA-SepReformer waveform decoder")

    tail_nodes = [node for node in nodes[decoder_index:] if node.op_type != "Constant"]
    output_names = {value.name for value in graph.output}
    if not output_names.issubset({value for node in tail_nodes for value in node.output}):
        raise RuntimeError("Unexpected FLA-SepReformer output topology; refusing cast islands")

    projections = ("/attn/q/", "/attn/kv/", "/attn/proj/", "/attn/dwc/")
    core_nodes = [
        node for node in nodes
        if "/ega/attn/" in node.name and node.op_type != "Constant" and not any(p in node.name for p in projections)
    ]
    scopes = {node.name.split("/ega/attn/")[0] for node in core_nodes}
    pow_outputs = {node.output[0] for node in core_nodes if node.op_type == "Pow"}
    # Each attention has two norms of the q**3 / k**3 powers and one division (Reciprocal or Div).
    focus_norms = [node for node in core_nodes if node.op_type == "ReduceL2" and node.input[0] in pow_outputs]
    reciprocals = [node for node in core_nodes if node.op_type in ("Reciprocal", "Div")]
    if not scopes or len(focus_norms) != 2 * len(scopes) or len(reciprocals) != len(scopes):
        raise RuntimeError(
            "Unexpected FLA-SepReformer focusing topology; refusing unsafe FP16: "
            f"scopes={len(scopes)}, norms={len(focus_norms)}, reciprocals={len(reciprocals)}"
        )

    selected = tail_nodes + core_nodes
    if any(not node.name for node in selected):
        raise RuntimeError("FP16 guard requires named ONNX nodes")
    return [node.name for node in selected]


MODEL_PLANS = {
    "FLA_SepReformer_SS_8K": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=8,
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
