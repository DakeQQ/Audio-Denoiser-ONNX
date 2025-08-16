---

## Audio-Denoiser-ONNX  
Audio denoising powered by ONNX Runtime for enhanced clarity.  

### Features  
1. **Supported Models**:  
   - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)  
   - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
   - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal/summary)
   - [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)
   - [MossFormerGAN-SE-16K](https://www.modelscope.cn/models/alibabasglab/MossFormerGAN_SE_16K)
   - [MossFormer2-Super-Resolution](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SR_48K)
   - [SDAEC - Acoustic Echo Cancellation](https://github.com/ZhaoF-i/SDAEC)
   - [DFSMN - Acoustic Echo Cancellation](https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k)

2. **Dynamic Quantization**:  
   - Dynamic quantization is **not recommended** for Denoiser as it significantly reduces performance due to increased computational overhead. Except, Mel-Band-Roformer. 

3. **End-to-End Processing**:  
   - The solution includes internal `STFT/ISTFT` processing.  
   - Input: Noisy audio  
   - Output: Crystal-clear denoised audio  

4. **Resources**:  
   - [Explore More Projects](https://github.com/DakeQQ?tab=repositories)  

5. **Note**
   - Please note that the denoiser model optimized (`opt_level=99`) on Windows cannot be used on Linux, and vice versa.
---

### 性能 Performance  
| OS           | Device       | Backend           | Model        | Real-Time Factor <br> (Chunk Size: 4000 or 250ms) |
|:------------:|:------------:|:-----------------:|:------------:|:------------------------------------------------:|
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | ZipEnhancer <br> f32 | 0.32                                              |
| Ubuntu-24.04 | Desktop      | OpenVINO-CPU <br> i3-12300 | ZipEnhancer <br> f32 | 0.25                                     |
| macOS 15     | MacBook Air  | CPU <br> M3       | ZipEnhancer <br> f32 | 0.25                                              |
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | GTCRN <br> f32       | 0.0036                                            |
| macOS 15     | MacBook Air  | CPU <br> M3       | GTCRN <br> f32       | 0.0013 ~<br> 0.0019                               |  
| Ubuntu-24.04 | Laptop       | CPU <br> i5-7300HQ | DFSMN <br> f32      | 0.0068 ~<br> 0.012                                |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | MelBandRofomer <br> q8f32 | 1.40 <br> (Chunk Size: 1000ms)              |
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | MossFormerGAN_SE_16K <br> f32 | 1.085                                    |
| Ubuntu-24.04 | Desktop      | OpenVINO-CPU <br> i3-12300 | MossFormerGAN_SE_16K <br> f32 | 0.95                            |
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | MossFormer2-SR <br> f32 | 1.49                                           |
| Ubuntu-24.04 | Desktop      | CPU <br> i7-1165G7 | SDAEC <br> f32 | 0.105                                                  |
| Ubuntu-24.04 | Desktop      |  OpenVINO-CPU <br> i7-1165G7 | SDAEC <br> f32 | 0.095                                        |
| Ubuntu-24.04 | Desktop      | CPU <br> i7-1165G7 | DFSMN_AEC <br> f32 | 0.11                                               |

---

### To-Do List  
- [ ] [Denoiser-MossFormer2-48K](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SE_48K)
- [ ] [ExNet-BF-PF](https://github.com/AdiCohen501/ExNet-BF-PF)
- [ ] [MP-SENet](https://github.com/yxlu-0102/MP-SENet)
---

## Audio-Denoiser-ONNX  
通过 ONNX Runtime 实现音频降噪，提升音质清晰度。

### 功能  
1. **支持的模型**：  
   - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
   - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
   - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal/summary)
   - [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)
   - [MossFormerGAN-SE-16K](https://www.modelscope.cn/models/alibabasglab/MossFormerGAN_SE_16K)
   - [MossFormer2-Super-Resolution](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SR_48K)
   - [SDAEC - Acoustic Echo Cancellation](https://github.com/ZhaoF-i/SDAEC)
   - [DFSMN - Acoustic Echo Cancellation](https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k)


2. **动态量化**：  
   - 除了 Mel-Band-Roformer 之外，**不建议**对其餘 Denoiser 应用动态量化，因为这会由于计算负载增加而显著降低性能。

3. **端到端处理**：  
   - 解决方案内置 `STFT/ISTFT` 处理。  
   - 输入：带噪音的音频  
   - 输出：清晰无噪音的音频  

4. **资源**：  
   - [探索更多项目](https://github.com/DakeQQ?tab=repositories)  

5. **Note**
   - 请注意，在 Windows 系统上优化的(`opt_level=99`)降噪模型无法在 Linux 系统上使用，反之亦然。
---
