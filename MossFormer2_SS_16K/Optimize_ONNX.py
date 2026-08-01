"""Optimize the exported MossFormer2 SS 16K ONNX model."""

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

ENABLE_FP16 = False     # Mixed FP16/FP32 with the wide output RMS path guarded below.
UPGRADE_OPSET = 0


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Keep decoded-waveform RMS restoration in FP32.

    Squared waveform samples fit FP16 individually, but CUDA's reduction over a
    full window can overflow. The gain division can also exceed 65504 before the
    final 1/32768 output scaling brings samples safely back into float range.
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
        raise RuntimeError("Could not identify the MossFormer2 SS waveform decoder")

    square_index = next(
        (
            index
            for index in range(decoder_index + 1, len(nodes))
            if nodes[index].op_type == "Mul"
            and len(nodes[index].input) == 2
            and nodes[index].input[0] == nodes[index].input[1]
        ),
        None,
    )
    output_scale_index = next(
        (index for index in range(len(nodes) - 1, decoder_index, -1) if nodes[index].op_type == "Mul"),
        None,
    )
    if square_index is None or output_scale_index is None or square_index >= output_scale_index:
        raise RuntimeError("Could not identify the MossFormer2 SS output RMS path")

    rms_nodes = [
        node
        for node in nodes[square_index:output_scale_index + 1]
        if node.op_type != "Constant"
    ]
    expected_ops = ["Mul", "ReduceMean", "Sqrt", "Greater", "Div", "Where", "Mul", "Mul"]
    ops = [node.op_type for node in rms_nodes]
    if ops != expected_ops or any(not node.name for node in rms_nodes):
        raise RuntimeError(
            "Unexpected MossFormer2 SS output RMS topology; refusing unsafe FP16: "
            f"ops={ops}"
        )

    output_nodes = [
        node
        for node in nodes[output_scale_index + 1:]
        if node.op_type != "Constant"
    ]
    output_names = {value.name for value in graph.output}
    produced_outputs = {value for node in output_nodes for value in node.output}
    if (
        not output_names.issubset(produced_outputs)
        or any(node.op_type not in {"Reshape", "Squeeze", "Split", "Transpose"} for node in output_nodes)
        or any(not node.name for node in output_nodes)
    ):
        raise RuntimeError(
            "Unexpected MossFormer2 SS output-routing topology; refusing cast islands: "
            f"ops={[node.op_type for node in output_nodes]}"
        )
    return [node.name for node in rms_nodes + output_nodes]


MODEL_PLANS = {
    "MossFormer2_SS_16K": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=8,
        hidden_size=512,
        # Preserve the selected output-routing nodes until precision conversion.
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
