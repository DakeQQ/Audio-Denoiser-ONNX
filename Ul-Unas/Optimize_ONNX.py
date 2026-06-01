import gc
import os

import onnx.version_converter
import onnxruntime
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim

# Path Setting
original_folder_path = "/home/DakeQQ/Downloads/Ul_Unas_ONNX"                        # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/Ul_Unas_Optimized"                  # The optimized folder.
model_path = os.path.join(original_folder_path, "Ul_Unas.onnx")                     # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, "Ul_Unas.onnx")          # The optimized model name.
use_fp16 = False                                                                    # Set true for fp16 quant.
target_opset = 0                                                                    # Upgrade the ONNX Opset version. Set it to 0 to keep the same version as the exported model.


# Check model
if isinstance(onnxruntime.InferenceSession(model_path)._inputs_meta[0].shape[-1], str):
    DYNAMIC_AXES = True
else:
    DYNAMIC_AXES = False


# ONNX Model Optimizer
slim(
    model=model_path,
    output_model=optimized_model_path,
    no_shape_infer=True if DYNAMIC_AXES else False,   # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# transformers.optimizer
model = optimize_model(optimized_model_path,
                       use_gpu=False,
                       opt_level=1,
                       num_heads=0,
                       hidden_size=0,
                       verbose=False,
                       model_type='bert')
if use_fp16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=False if DYNAMIC_AXES else True,  # True for more optimize but may get errors.
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(optimized_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=optimized_model_path,
    output_model=optimized_model_path,
    no_shape_infer=True if DYNAMIC_AXES else False,   # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# Upgrade the Opset version. (optional process)
if target_opset != 0:
    model = onnx.load(optimized_model_path)
    model = onnx.version_converter.convert_version(model, target_opset)
    onnx.save(model, optimized_model_path, save_as_external_data=False)
    del model
    gc.collect()

