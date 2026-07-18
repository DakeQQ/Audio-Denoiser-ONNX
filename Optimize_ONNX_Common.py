"""Shared ONNX optimization pipeline for Audio-Denoiser export scripts."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import onnx
import onnx.version_converter
from onnx import TensorProto
from onnxruntime.quantization import QuantType, quant_utils, quantize_dynamic
from onnxslim import slim

from audio_onnx_metadata import preserve_optimized_metadata, read_source_metadata


NodeSelector = list[str] | Callable[[str], list[str] | None] | None
IntValue = int | Callable[[str], int]
NoShapeInfer = bool | Literal["auto"]
SymbolicShapeInfer = bool | Literal["auto"]

_DYNAMIC_METHODS = {"DYNAMIC", "DYNAMIC_Q8"}
_VALID_METHODS = {"F32", "F16", *_DYNAMIC_METHODS}
_DYNAMIC_WEIGHT_TYPES = {"QUINT8": QuantType.QUInt8, "QINT8": QuantType.QInt8}
_DEFAULT_F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
    # "Pow",
    # "Sqrt",
    # "ReduceMean",
    # "ReduceSum",
    # "ReduceL2",
    # "Softmax",
    # "Exp",
    # "Log",
]


@dataclass
class Plan:
    """Per-model optimization recipe; ``None`` fields inherit ``OptimizerConfig`` defaults."""

    method: str = "F32"  # F32 | F16 | DYNAMIC | DYNAMIC_Q8
    num_heads: IntValue = 0
    hidden_size: IntValue = 0
    opt_level: int | None = None
    only_onnxruntime: bool | None = None
    provider: str | None = None
    use_gpu: bool | None = None
    optimize: bool = True
    transformer: bool = True
    fp16: bool = False
    external: bool | None = None
    first_slim_no_shape_infer: NoShapeInfer | None = None
    second_slim_no_shape_infer: NoShapeInfer | None = None
    fp16_symbolic_shape_infer: SymbolicShapeInfer | None = None
    upgrade_opset: int | None = None
    nodes_to_exclude: NodeSelector = None
    nodes_to_include: NodeSelector = None
    dynamic_weight_type: str | None = None
    per_channel: bool | None = None
    reduce_range: bool | None = None
    default_tensor_type: int | None = None
    f16_node_block_list: list[str] | None = None
    f16_op_block_list: list[str] | None = None


@dataclass
class OptimizerConfig:
    """Global defaults shared by one thin ``Optimize_ONNX.py`` wrapper."""

    original_folder_path: str
    optimized_folder_path: str
    model_plans: dict[str, Plan]
    force_external_data: bool = False
    upgrade_opset: int = 0
    skip_missing: bool = True
    # graph optimizer
    optimizer_level: int = 2
    optimizer_model_type: str = "bert"
    optimizer_only_onnxruntime: bool = False
    optimizer_provider: str | None = None
    optimizer_use_gpu: bool = False
    optimizer_fusion_options: dict | None = None
    # onnxslim
    first_slim_no_shape_infer: NoShapeInfer = False
    second_slim_no_shape_infer: NoShapeInfer | None = None
    slim_skip_fusion_patterns: object = False
    slim_skip_optimizations: object = None
    slim_size_threshold: int | None = None
    slim_no_constant_folding: bool = False
    # dynamic INT8 defaults, used only for explicit DYNAMIC/DYNAMIC_Q8 plans
    dynamic_weight_type: str = "QInt8"
    dynamic_per_channel: bool = True
    dynamic_reduce_range: bool = False
    dynamic_default_tensor_type: int | None = None
    dynamic_nodes_to_exclude: NodeSelector = None
    dynamic_nodes_to_include: NodeSelector = None
    # float16 defaults: opt-in only; models can add targeted waveform/norm guards
    f16_keep_io_types: bool = False
    f16_force_initializers: bool = True
    f16_min_positive_val: float = 1e-7
    f16_max_finite_val: float = 32767.0
    f16_symbolic_shape_infer: SymbolicShapeInfer = True
    f16_node_block_list: list[str] | None = None
    f16_op_block_list: list[str] | None = None


@dataclass
class ResolvedPlan:
    method: str
    num_heads: IntValue
    hidden_size: IntValue
    opt_level: int | None
    only_onnxruntime: bool
    provider: str | None
    use_gpu: bool
    optimize: bool
    transformer: bool
    fp16: bool
    external: bool
    first_slim_no_shape_infer: NoShapeInfer
    second_slim_no_shape_infer: NoShapeInfer | None
    fp16_symbolic_shape_infer: SymbolicShapeInfer
    upgrade_opset: int
    nodes_to_exclude: NodeSelector
    nodes_to_include: NodeSelector
    dynamic_weight_type: str
    per_channel: bool
    reduce_range: bool
    default_tensor_type: int | None
    f16_node_block_list: list[str] | None
    f16_op_block_list: list[str]


def _pick(value, default):
    return default if value is None else value


def _uses_fp16(plan: Plan) -> bool:
    return bool(plan.fp16 or plan.method.upper() == "F16")


def resolve_plan(plan: Plan, config: OptimizerConfig) -> ResolvedPlan:
    return ResolvedPlan(
        method=plan.method.upper(),
        num_heads=plan.num_heads,
        hidden_size=plan.hidden_size,
        opt_level=plan.opt_level,
        only_onnxruntime=_pick(plan.only_onnxruntime, config.optimizer_only_onnxruntime),
        provider=_pick(plan.provider, config.optimizer_provider),
        use_gpu=_pick(plan.use_gpu, config.optimizer_use_gpu),
        optimize=plan.optimize,
        transformer=plan.transformer,
        fp16=plan.fp16,
        external=_pick(plan.external, config.force_external_data),
        first_slim_no_shape_infer=_pick(plan.first_slim_no_shape_infer, config.first_slim_no_shape_infer),
        second_slim_no_shape_infer=_pick(plan.second_slim_no_shape_infer, config.second_slim_no_shape_infer),
        fp16_symbolic_shape_infer=_pick(plan.fp16_symbolic_shape_infer, config.f16_symbolic_shape_infer),
        upgrade_opset=config.upgrade_opset if plan.upgrade_opset is None else plan.upgrade_opset,
        nodes_to_exclude=_pick(plan.nodes_to_exclude, config.dynamic_nodes_to_exclude),
        nodes_to_include=_pick(plan.nodes_to_include, config.dynamic_nodes_to_include),
        dynamic_weight_type=_pick(plan.dynamic_weight_type, config.dynamic_weight_type).upper(),
        per_channel=_pick(plan.per_channel, config.dynamic_per_channel),
        reduce_range=_pick(plan.reduce_range, config.dynamic_reduce_range),
        default_tensor_type=_pick(plan.default_tensor_type, config.dynamic_default_tensor_type),
        f16_node_block_list=_pick(plan.f16_node_block_list, config.f16_node_block_list),
        f16_op_block_list=_pick(plan.f16_op_block_list, config.f16_op_block_list) or list(_DEFAULT_F16_OP_BLOCK_LIST),
    )


def validate_plan(name: str, rp: ResolvedPlan) -> None:
    if rp.method not in _VALID_METHODS:
        raise ValueError(
            f"[{name}] unknown method {rp.method!r}; use 'F32', 'F16', 'DYNAMIC', or 'DYNAMIC_Q8'. "
            "Weight-only Q2/Q4/Q8 quantization is intentionally not part of the audio-denoiser default path."
        )
    if rp.method in _DYNAMIC_METHODS and rp.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
        raise ValueError(f"[{name}] unknown dynamic_weight_type; choose 'QUInt8' or 'QInt8'.")


def _remove_external_files(model_path: str) -> None:
    for path in (model_path, model_path + ".data"):
        if os.path.exists(path):
            os.remove(path)


def _save_model(model, model_path: str, external: bool) -> None:
    _remove_external_files(model_path)
    if external:
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(model_path) + ".data",
        )
    else:
        onnx.save(model, model_path)


def _iter_all_data_tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _iter_all_data_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_all_data_tensors(subgraph)


def _retarget_external_location(model_path: str, new_location: str) -> None:
    model = onnx.load(model_path, load_external_data=False)
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location == TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = new_location
    onnx.save(model, model_path)
    del model
    gc.collect()


def resave(src_path: str, dst_path: str, external: bool) -> None:
    model = onnx.load(src_path)
    _save_model(model, dst_path, external)
    del model
    gc.collect()


def _shape_value_is_dynamic(value) -> bool:
    return value is None or isinstance(value, str)


def detect_dynamic_axes(model_path: str) -> bool:
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        try:
            return any(_shape_value_is_dynamic(dim) for meta in session._inputs_meta for dim in meta.shape)
        finally:
            del session
            gc.collect()
    except Exception:
        model = onnx.load(model_path, load_external_data=False)
        try:
            for value_info in model.graph.input:
                shape = value_info.type.tensor_type.shape
                for dim in shape.dim:
                    if dim.dim_param or not dim.HasField("dim_value"):
                        return True
            return False
        finally:
            del model
            gc.collect()


def _resolve_no_shape_infer(setting: NoShapeInfer, dynamic_axes: bool) -> bool:
    if setting == "auto":
        return dynamic_axes
    return bool(setting)


def _resolve_symbolic_shape_infer(setting: SymbolicShapeInfer, dynamic_axes: bool) -> bool:
    if setting == "auto":
        return not dynamic_axes
    return bool(setting)


def run_onnxslim(model_path: str, external: bool, config: OptimizerConfig, no_shape_infer: bool) -> None:
    def _slim() -> None:
        slim(
            model=model_path,
            output_model=model_path,
            no_shape_infer=no_shape_infer,
            skip_fusion_patterns=config.slim_skip_fusion_patterns,
            skip_optimizations=config.slim_skip_optimizations,
            no_constant_folding=config.slim_no_constant_folding,
            size_threshold=config.slim_size_threshold,
            save_as_external_data=external,
            verbose=False,
        )

    data_path = model_path + ".data"
    if not external or not os.path.exists(data_path):
        _slim()
        return

    stash_path = model_path + ".stash.data"
    if os.path.exists(stash_path):
        os.remove(stash_path)
    os.replace(data_path, stash_path)
    _retarget_external_location(model_path, os.path.basename(stash_path))
    try:
        _slim()
    except BaseException:
        if not os.path.exists(data_path):
            os.replace(stash_path, data_path)
            _retarget_external_location(model_path, os.path.basename(data_path))
        raise
    finally:
        if os.path.exists(stash_path):
            os.remove(stash_path)


def build_fusion_options(config: OptimizerConfig):
    if not config.optimizer_fusion_options:
        return None
    from onnxruntime.transformers.fusion_options import FusionOptions

    options = FusionOptions(config.optimizer_model_type)
    for key, value in config.optimizer_fusion_options.items():
        setattr(options, key, value)
    return options


def _deduplicate_node_names(graph) -> int:
    used_names, next_name_suffix, used_values, next_value_suffix, remap, renamed = set(), {}, set(), {}, {}, 0
    used_values.update(i.name for i in graph.input)
    used_values.update(i.name for i in graph.initializer)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in remap:
                node.input[i] = remap[name]

        name = node.name
        if name:
            if name not in used_names:
                used_names.add(name)
            else:
                suffix = next_name_suffix.get(name, 1)
                while f"{name}_{suffix}" in used_names:
                    suffix += 1
                node.name = f"{name}_{suffix}"
                used_names.add(node.name)
                next_name_suffix[name] = suffix + 1
                renamed += 1

        for i, output in enumerate(node.output):
            if not output:
                continue
            if output not in used_values:
                used_values.add(output)
                continue
            suffix = next_value_suffix.get(output, 1)
            while f"{output}_{suffix}" in used_values:
                suffix += 1
            new_output = f"{output}_{suffix}"
            node.output[i] = new_output
            used_values.add(new_output)
            next_value_suffix[output] = suffix + 1
            remap[output] = new_output
            renamed += 1
    return renamed


def _resolve_int(value: IntValue, src_path: str) -> int:
    return int(value(src_path)) if callable(value) else int(value)


def _resolve_nodes(selector: NodeSelector, src_path: str) -> list[str] | None:
    nodes = selector(src_path) if callable(selector) else selector
    return nodes or None


def optimize_onnx_model(
    model_path: str,
    rp: ResolvedPlan,
    config: OptimizerConfig,
    src_path: str,
    use_fp16: bool,
    external: bool,
    dynamic_axes: bool,
) -> None:
    from onnxruntime.transformers.optimizer import optimize_model

    opt_level = config.optimizer_level if rp.opt_level is None else rp.opt_level
    kwargs = {
        "use_gpu": rp.use_gpu,
        "opt_level": opt_level,
        "num_heads": _resolve_int(rp.num_heads, src_path),
        "hidden_size": _resolve_int(rp.hidden_size, src_path),
        "optimization_options": build_fusion_options(config),
        "model_type": config.optimizer_model_type,
        "only_onnxruntime": rp.only_onnxruntime,
        "verbose": False,
    }
    if rp.provider is not None:
        kwargs["provider"] = rp.provider

    if opt_level == 0 and rp.only_onnxruntime:
        # Some mixed-precision policies select one contiguous FP32 region by the
        # exporter's node names. ORT level 1 rewrites rank-3 MatMul nodes into
        # Reshape -> Gemm -> Reshape before float16 conversion, fragmenting that
        # region into hundreds of avoidable FP16 islands. Wrap the graph directly
        # when level 0 is requested; the second onnxslim pass and runtime ORT
        # optimizer still apply dtype-aware rewrites after conversion.
        from onnxruntime.transformers.onnx_model import OnnxModel

        model = OnnxModel(onnx.load(model_path))
    else:
        model = optimize_model(model_path, **kwargs)
    if use_fp16:
        model.convert_float_to_float16(
            keep_io_types=config.f16_keep_io_types,
            force_fp16_initializers=config.f16_force_initializers,
            use_symbolic_shape_infer=_resolve_symbolic_shape_infer(rp.fp16_symbolic_shape_infer, dynamic_axes),
            max_finite_val=config.f16_max_finite_val,
            min_positive_val=config.f16_min_positive_val,
            op_block_list=rp.f16_op_block_list,
            node_block_list=_resolve_nodes(rp.f16_node_block_list, src_path),
        )
        renamed = _deduplicate_node_names(model.model.graph)
        if renamed:
            print(f"  Renamed {renamed} duplicate node names after float16 conversion.")
    model.save_model_to_file(model_path, use_external_data_format=external)
    del model
    gc.collect()


def quantize_dynamic_int8(src_path: str, dst_path: str, rp: ResolvedPlan, external: bool) -> None:
    weight_type = _DYNAMIC_WEIGHT_TYPES[rp.dynamic_weight_type]
    extra_options = {
        "ActivationSymmetric": False,
        "WeightSymmetric": False,
        "EnableSubgraph": True,
        "ForceQuantizeNoInputCheck": False,
        "MatMulConstBOnly": True,
    }
    if rp.default_tensor_type is not None:
        extra_options["DefaultTensorType"] = rp.default_tensor_type
    print(
        f"  Quantizing weights (dynamic INT8, {rp.dynamic_weight_type}, "
        f"per_channel={rp.per_channel}, reduce_range={rp.reduce_range})..."
    )
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    quantize_dynamic(
        model_input=model,
        model_output=dst_path,
        per_channel=rp.per_channel,
        reduce_range=rp.reduce_range,
        weight_type=weight_type,
        extra_options=extra_options,
        nodes_to_quantize=_resolve_nodes(rp.nodes_to_include, src_path),
        nodes_to_exclude=_resolve_nodes(rp.nodes_to_exclude, src_path),
        use_external_data_format=external,
    )
    del model
    gc.collect()


def upgrade_opset_version(model_path: str, version: int, external: bool) -> None:
    print(f"  Upgrading opset to {version}...")
    try:
        model = onnx.version_converter.convert_version(onnx.load(model_path), version)
        _save_model(model, model_path, external)
        del model
        gc.collect()
    except Exception as exc:
        print(f"  Opset upgrade failed: {exc}. Keeping current version.")
        resave(model_path, model_path, external)


def get_model_paths(config: OptimizerConfig, name: str) -> tuple[str, str]:
    return (
        os.path.join(config.original_folder_path, f"{name}.onnx"),
        os.path.join(config.optimized_folder_path, f"{name}.onnx"),
    )


def process_model(name: str, rp: ResolvedPlan, config: OptimizerConfig) -> bool:
    src_path, dst_path = get_model_paths(config, name)
    if not os.path.exists(src_path):
        message = f"  Skipping - file not found: {src_path}"
        if config.skip_missing:
            print(message)
            return False
        raise FileNotFoundError(message)

    source_metadata = read_source_metadata(src_path)
    _remove_external_files(dst_path)

    external = rp.external
    use_fp16 = rp.fp16 or rp.method == "F16"
    dynamic_axes = detect_dynamic_axes(src_path)
    first_no_shape = _resolve_no_shape_infer(rp.first_slim_no_shape_infer, dynamic_axes)
    second_setting = rp.second_slim_no_shape_infer
    if second_setting is None:
        second_setting = rp.first_slim_no_shape_infer
    second_no_shape = _resolve_no_shape_infer(second_setting, dynamic_axes)

    print(f"  Source: {src_path}")
    print(f"  Output: {dst_path}")
    if dynamic_axes:
        print("  Dynamic axes detected; using dynamic-safe shape-inference settings where configured.")

    resave(src_path, dst_path, external)

    if rp.optimize:
        print("  Simplifying (onnxslim, pass 1)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=first_no_shape)

    if rp.method in _DYNAMIC_METHODS:
        quantize_dynamic_int8(dst_path, dst_path, rp, external)

    if rp.optimize and (rp.transformer or use_fp16):
        print("  Optimizing (ORT transformers optimizer)...")
        optimize_onnx_model(dst_path, rp, config, src_path, use_fp16, external, dynamic_axes)
        print("  Simplifying (onnxslim, pass 2)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=second_no_shape)

    if rp.upgrade_opset > 0:
        upgrade_opset_version(dst_path, rp.upgrade_opset, external)

    if not external and os.path.exists(dst_path + ".data"):
        os.remove(dst_path + ".data")

    preserve_optimized_metadata(src_path, dst_path, source_metadata)
    return True


def run_optimizer(config: OptimizerConfig) -> None:
    os.makedirs(config.optimized_folder_path, exist_ok=True)

    resolved = {name: resolve_plan(plan, config) for name, plan in config.model_plans.items()}
    for name, rp in resolved.items():
        validate_plan(name, rp)

    for name in resolved:
        _, dst_path = get_model_paths(config, name)
        _remove_external_files(dst_path)

    if any(_uses_fp16(plan) for plan in config.model_plans.values()):
        print(
            "FP16 conversion is enabled explicitly. Configured node/operator guards remain in FP32; "
            "validate each model with realistic audio before use."
        )

    for name, rp in resolved.items():
        print(f"\n{'=' * 60}\nProcessing: {name}  [{rp.method}{' + FP16' if rp.fp16 else ''}]\n{'=' * 60}")
        process_model(name, rp, config)

    print("\n--- All models processed successfully! ---")