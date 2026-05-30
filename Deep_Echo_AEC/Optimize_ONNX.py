import gc
import os

import onnx.version_converter
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim

# Path Setting
original_folder_path = "/home/DakeQQ/Downloads/Deep_Echo_AEC_ONNX"                        # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/Deep_Echo_AEC_Optimized"                  # The optimized folder.
model_path = os.path.join(original_folder_path, "Deep_Echo_AEC.onnx")                     # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, "Deep_Echo_AEC.onnx")          # The optimized model name.
use_fp16 = False
target_opset = 0

# ONNX Model Optimizer
slim(
    model=model_path,
    output_model=optimized_model_path,
    no_shape_infer=False,   # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# transformers.optimizer
model = optimize_model(optimized_model_path,
                       use_gpu=False,
                       opt_level=1,  # 1 perform much better in CUDA and OpenVINO
                       num_heads=0,
                       hidden_size=0,
                       verbose=False,
                       model_type='bert')
if use_fp16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(optimized_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=optimized_model_path,
    output_model=optimized_model_path,
    no_shape_infer=False,   # False for more optimize but may get errors.
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
