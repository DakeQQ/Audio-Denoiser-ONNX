import gc
import os
import subprocess

import onnx.version_converter
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim


# Path Setting
original_folder_path = "/home/DakeQQ/Downloads/MelBandRoformer_ONNX"                    # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/MelBandRoformer_Optimized"              # The optimized folder.
model_path = os.path.join(original_folder_path, "MelBandRoformer.onnx")                 # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, "MelBandRoformer.onnx")      # The optimized model name.
do_quantize = True
use_gpu_fp16 = False                                                                    # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                       # ['CPUExecutionProvider', 'CUDAExecutionProvider']
target_platform = "amd64"                                                               # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.


# ONNX Model Optimizer
slim(
    model=model_path,
    output_model=optimized_model_path,
    no_shape_infer=True,                                    # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


if do_quantize:
    quantize_dynamic(
        model_input=optimized_model_path,
        model_output=optimized_model_path,
        per_channel=True,                                   # True for model accuracy but cost a lot of time during quanting process.
        reduce_range=False,                                 # True for some x86_64 platform.
        weight_type=QuantType.QUInt8,                       # It is recommended using uint8 + Symmetric False
        extra_options={'ActivationSymmetric': False,        # True for inference speed. False may keep more accuracy.
                       'WeightSymmetric': False,            # True for inference speed. False may keep more accuracy.
                       'EnableSubgraph': True,              # True for more quant.
                       'ForceQuantizeNoInputCheck': False,  # True for more quant.
                       'MatMulConstBOnly': True             # False for more quant. Sometime, the inference speed may get worse.
                       },
        nodes_to_exclude=None,                              # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
        use_external_data_format=False                      # Save the model into two parts.
    )


# transformers.optimizer
model = optimize_model(optimized_model_path,
                       use_gpu=use_gpu_fp16,
                       opt_level=2,
                       num_heads=8,
                       hidden_size=1536,
                       provider=provider,
                       verbose=False,
                       model_type='bert')
if use_gpu_fp16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=True,
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(optimized_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=optimized_model_path,
    output_model=optimized_model_path,
    no_shape_infer=True,                                   # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# Upgrade the Opset version. (optional process)
model = onnx.load(optimized_model_path)
model = onnx.version_converter.convert_version(model, 18)
onnx.save(model, optimized_model_path, save_as_external_data=False)
del model
gc.collect()


if not use_gpu_fp16:
    # Convert the simplified model to ORT format.
    if provider == 'CPUExecutionProvider':
        optimization_style = "Fixed"
    else:
        optimization_style = "Runtime"  # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {optimized_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {optimized_folder_path}'], shell=True)
