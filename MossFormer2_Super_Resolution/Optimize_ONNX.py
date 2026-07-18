"""Optimize the exported MossFormer2 Super Resolution ONNX model."""

from pathlib import Path
import sys

import onnx
from onnx import TensorProto


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

ENABLE_FP16 = False    # CUDA-safe mixed FP16/FP32 graph; validated by FP16_CUDA_Test.
UPGRADE_OPSET = 0      # 0 = Keep the exported opset;


def _value_consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    return consumers


def _cast_target(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    return next((attribute.i for attribute in node.attribute if attribute.name == "to"), None)


def _fp16_sensitive_nodes(src_path: str) -> list[str]:
    """Keep overflow-sensitive arithmetic in two contiguous FP32 regions.

    The normalized resampler/STFT, mask tail, crossover, and complete HiFi-GAN
    generator fit FP16. The mel power/log path needs FP32 for quiet-signal dynamic
    range, while the 24-layer MossFormer contains attention, ScaleNorm, and FSMN
    intermediates that exceed 65,504. Keeping the complete recurrent core in one
    region is both safer and much cheaper than inserting casts around individual
    reductions. The level-0 pre-conversion policy preserves this region until dtype
    assignment; the post-conversion simplifier can then fuse MatMul chains without
    introducing FP16 sandwiches.
    """
    graph = onnx.load(src_path, load_external_data=False).graph
    nodes = list(graph.node)
    producers = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    consumers = _value_consumers(graph)

    log_index = next((i for i, node in enumerate(nodes) if node.op_type == "Log"), None)
    if log_index is None:
        raise RuntimeError("Could not identify the MossFormer2 SR mel Log node")
    square_index = next(
        (
            i
            for i, node in enumerate(nodes[:log_index])
            if node.op_type == "Mul"
            and len(node.input) >= 2
            and node.input[0] == node.input[1]
            and any(candidate.op_type == "ReduceSum" for candidate in consumers.get(node.output[0], ()))
        ),
        None,
    )
    if square_index is None:
        raise RuntimeError("Could not identify the MossFormer2 SR mel power path")

    front_conv = next(
        (
            node
            for node in nodes[log_index + 1:]
            if node.op_type == "Conv" and "front_w" in node.input
        ),
        None,
    )
    if front_conv is None:
        raise RuntimeError("Could not identify the MossFormer2 SR front projection")
    front_add = next(
        (
            node
            for node in consumers.get(front_conv.output[0], ())
            if node.op_type == "Add"
        ),
        None,
    )
    if front_add is None:
        raise RuntimeError("Could not identify the MossFormer2 SR positional Add")
    front_value = front_add.output[0]
    core_start = next(
        (
            i
            for i, node in enumerate(nodes)
            if node.op_type == "Transpose" and front_value in node.input
        ),
        None,
    )
    core_end_candidates = [
        i
        for i, node in enumerate(nodes)
        if node.op_type == "Add"
        and front_value in node.input
        and any(candidate.op_type == "LeakyRelu" for candidate in consumers.get(node.output[0], ()))
    ]
    if core_start is None or len(core_end_candidates) != 1:
        raise RuntimeError(
            "Could not identify one contiguous MossFormer2 SR recurrent core: "
            f"start={core_start}, ends={core_end_candidates}"
        )
    core_end = core_end_candidates[0]
    core = nodes[core_start:core_end + 1]
    if sum(node.op_type == "ReduceL2" for node in core) != 48:
        raise RuntimeError("Unexpected MossFormer2 SR core: expected 48 ScaleNorm ReduceL2 nodes")
    if sum(node.op_type == "LayerNormalization" for node in core) != 73:
        raise RuntimeError("Unexpected MossFormer2 SR core: expected 73 LayerNormalization nodes")

    # The export must use the FP16-safe PCM boundary:
    # float clip[-1, 1] -> *32768 -> int32 -> integer clip -> int16.
    output_cast = producers.get(graph.output[0].name)
    int_clip = producers.get(output_cast.input[0]) if output_cast is not None else None
    int32_cast = producers.get(int_clip.input[0]) if int_clip is not None else None
    pcm_mul = producers.get(int32_cast.input[0]) if int32_cast is not None else None
    normalized_clip = producers.get(pcm_mul.input[0]) if pcm_mul is not None else None
    pcm_ops = [
        output_cast.op_type if output_cast is not None else None,
        int_clip.op_type if int_clip is not None else None,
        int32_cast.op_type if int32_cast is not None else None,
        pcm_mul.op_type if pcm_mul is not None else None,
        normalized_clip.op_type if normalized_clip is not None else None,
    ]
    if (
        pcm_ops != ["Cast", "Clip", "Cast", "Mul", "Clip"]
        or _cast_target(output_cast) != TensorProto.INT16
        or _cast_target(int32_cast) != TensorProto.INT32
    ):
        raise RuntimeError(
            "Unsafe or stale PCM output chain. Re-run Export_MossFormer_SR.py before FP16 optimization; "
            f"found ops={pcm_ops}."
        )

    protected = [
        node.name
        for node in [*nodes[square_index:log_index + 1], *core]
        if node.name and node.op_type != "Constant"
    ]
    print(
        f"  Keeping mel power/log ({log_index - square_index + 1} nodes) and the complete "
        f"MossFormer core ({core_end - core_start + 1} nodes) in FP32."
    )
    return protected


MODEL_PLANS = {
    "MossFormer2_SR": Plan(
        method="F16" if ENABLE_FP16 else "F32",
        num_heads=4,
        hidden_size=128,
        # ORT level 2 serializes the Generator's Conv -> Tanh tail as a
        # com.microsoft::FusedConv node, but the CUDA FusedConv kernel does not
        # support Tanh. Level 1 keeps the portable standard Conv and Tanh nodes.
        # For FP16, assign dtypes before ORT rewrites MatMul into
        # Reshape -> Gemm -> Reshape. Otherwise those generated layout nodes form
        # hundreds of F16 islands inside the selected FP32 core.
        opt_level=0 if ENABLE_FP16 else 1,
        only_onnxruntime=True,
        use_gpu=False,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
        fp16_symbolic_shape_infer=False,
        f16_node_block_list=_fp16_sensitive_nodes if ENABLE_FP16 else None,
        f16_op_block_list=["Range"],
    ),
}


CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    # FP32-region weights and small quiet-signal constants must not be quantized
    # to FP16 and cast back. That would defeat the protected mel/core arithmetic.
    f16_force_initializers=False,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
