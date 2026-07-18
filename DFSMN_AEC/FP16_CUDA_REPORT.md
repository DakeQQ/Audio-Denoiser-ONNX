# DFSMN_AEC mixed-FP16 CUDA report

Validated 2026-07-18 on NVIDIA GeForce RTX 5060 Ti, ONNX Runtime 1.27.0, CUDAExecutionProvider, `use_tf32=0`.

## Scope

The checked artifact is the static, batch-folded SDAEC composite profile:

- Inputs: `near_end_audio`, `far_end_audio`, INT16 `[1, 1, 48000]`
- Output: `aec_audio`, INT16 `[1, 1, 48000]`
- Internal fold: two independent 24,000-sample windows
- Model metadata: `light_aec_model=SDAEC`

The exporter and fail-closed selector also recognize `Deep_Echo` and `NKF`. The scale policy is shared by all three branches, and the Deep Echo source includes its separately validated quarter-scaled cepstral reduction. The quantitative results below are for the active SDAEC artifact; separate fixed Deep Echo/NKF artifacts should run the same regression test after changing `project_path_B` and exporting them.

## Diagnosis before conversion changes

The original FP32 graph and a blanket FP16 conversion were run on the same 305,783-sample bundled near/far pair. The blanket graph returned all-zero audio while FP32 had RMS 2275.93.

Layer-output comparison on CUDA found two independent failures:

1. **PCM scale folded into static ISTFT normalization.**
   - `light_aec.custom_istft.inv_win_sum`: max 5,108,596
   - `custom_istft_A2.inv_win_sum`: max 5,119,842.5
   - FP16 conversion clipped these constants to roughly 32,768.
   - At `/light_aec/custom_istft/Mul`, relative error jumped to 0.2564, RMS ratio fell to 0.8070, and correlation fell to 0.9823.
2. **Int16-domain fbank square/reduction overflow.**
   - The first non-finite tensor was `/ReduceSum_1` in the fbank power path.
   - FP32 max magnitude was 1.37858e11; the following Mel `MatMul` reached 5.24952e11.
   - The FP16 tensor became `Inf`, then propagated `NaN` through the DFSMN mask and final ISTFT.

The 41 `LayerNormalization` nodes were finite in the diagnostic run. There is no Softmax/attention subgraph. The failures were the oversized COLA multiplication and fbank power accumulation, not Softmax or GELU.

## Exporter correction

`Export_DFSMN_AEC.py` now applies one scale-safe policy to SDAEC, Deep Echo, and NKF:

- Explicitly multiply INT16 PCM by exact `2^-15` before the light-AEC backend.
- Keep backend and mask STFT kernels at unit scale instead of storing approximately `3e-5` DFT weights.
- Keep both ISTFT COLA reciprocals unscaled; current maxima are 155.902 and 156.250.
- Run the fbank convolution on normalized waveforms, then restore the trained int16-domain power with exact `2^30` in the selected FP32 frontend.
- Apply `32767` only at the terminal FP32 PCM boundary before `Clip` and `Cast(INT16)`.
- For Deep Echo, scale the two cepstral LayerNorm centered inputs by `1/4` and epsilon by `1/16`. This is algebraically neutral but prevents sum-of-squares overflow if that backend is later narrowed to FP16.

The scale-safe FP32 export remains equivalent to the prior FP32 graph on the complete pair:

- CPU: max 1 LSB, 7 changed samples, 0 samples over 1 LSB
- CUDA: max 1 LSB, 6 changed samples, 0 samples over 1 LSB

## Final mixed-precision policy

A fully FP16 light-AEC backend is finite, but its temporary waveform relative error was about 0.006. The cancellation-sensitive `near - 1.15 * temporary` feature and `Log` amplified this: fbank-convolution relative error 0.0029 became 0.1113 after `Log`, and complete output quality fell to correlation 0.99916 / SNR 27.75 dB.

The accepted graph therefore uses two merged FP32 regions:

1. **Frontend FP32 region:** input cast/scale, selected light-AEC backend, near/temp/echo fbank convolution, power restoration, Mel projection, clamp, and log.
2. **Output FP32 region:** terminal `32767` multiply, batch-fold reshape, and clamp.

The DFSMN mask network, mask STFT, mask application, and final ISTFT remain FP16. This preserves the bulk of the model-size and CUDA-compute benefit while preventing the cascaded model from amplifying light-AEC FP16 noise.

Only onnxslim `graph_fusion` is disabled during conversion. Otherwise its inserted `Reshape -> Gemm -> Reshape` nodes fragment one blocked region into dozens of small FP16 islands. Constant folding, dead-node elimination, common-subexpression elimination, and weight tying remain enabled; ORT applies runtime graph optimization after loading.

## Cast audit

Final graph: exactly 5 Cast nodes, no adjacent `Cast -> Cast`, and no short FP16 sandwich:

1. INT16 -> FP32 at the merged frontend entry
2. FP32 -> FP16 for the temporary waveform entering the mask STFT
3. FP32 -> FP16 after fbank `Log` entering the DFSMN mask network
4. FP16 -> FP32 after final ISTFT normalization
5. FP32 -> INT16 at output

Comparison:

| Policy | Casts | Result |
|---|---:|---|
| Blanket FP16 | 2 | Invalid; correlation 0.7150, SNR 2.95 dB |
| Broad suspect-op block list | 90 | Excessive FP32/FP16 sandwiches |
| Final merged policy | 5 | Valid; 94.4% fewer casts than broad blocking |

The blanket graph has the unavoidable two I/O casts and is the theoretical lower bound, but it is numerically invalid. A valid mixed graph necessarily adds precision-boundary casts; the meaningful cast-overhead comparison is therefore 90 versus 5.

## CUDA quality and placement

Permanent test: `FP16_CUDA_Test/test_fp16_cuda.py`

Thresholds:

- Correlation >= 0.99999
- SNR >= 50 dB
- Max absolute error <= 64 INT16 LSB
- RMS ratio in `[0.999, 1.001]`
- Stress cases <= 1 LSB
- Exactly five casts and no adjacent cast pair
- Every profiled node event on CUDA

Measured on the complete 305,783-sample pair:

| Metric | Result |
|---|---:|
| Correlation | 0.9999998035 |
| SNR | 64.05 dB |
| Mean absolute error | 0.463 LSB |
| Max absolute error | 46 LSB |
| FP32 RMS | 2275.9295 |
| Mixed RMS | 2275.8760 |
| RMS ratio | 0.9999765 |

Stress results:

- Silence: exact
- Opposing DC extrema: exact
- Alternating extrema: max 1 LSB
- Full-range random: max 1 LSB
- 1%-gain bundled audio: exact for this quantized output profile

ORT profile: 677/677 executable node events on `CUDAExecutionProvider`, zero CPU execution events.

Production `Inference_DFSMN_ONNX_AEC.py` completed through CUDA I/O binding with TF32 disabled and RTF 0.0315 on this machine. Its saved WAV matched the controlled test at correlation 0.99999981, SNR 64.09 dB, MAE 0.461 LSB, and max error 46 LSB. This is a smoke result, not a controlled benchmark claim.

## Artifact summary

- FP32 source: 36,126,484 bytes
- Mixed model: 21,078,330 bytes
- Size reduction: 41.65%
- Mixed initializers: 229 FP32, 55 FP16, 29 INT64
- ONNX checker and strict shape inference: pass
- Metadata carrier preserved beside both source and optimized models

The latest machine-readable results are written by the test to `FP16_CUDA_Test/last_validation.json`.
