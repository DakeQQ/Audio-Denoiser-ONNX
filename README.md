---

## Audio-Denoiser-ONNX  
Audio denoising powered by ONNX Runtime for enhanced clarity.  

### Features  
1. **Supported Models**:  
   - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)  
   - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
   - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal/summary)

2. **Dynamic Quantization**:  
   - Dynamic quantization is **not recommended** for ZipEnhancer as it significantly reduces performance due to increased computational overhead.  

3. **End-to-End Processing**:  
   - The solution includes internal `STFT/ISTFT` processing.  
   - Input: Noisy audio  
   - Output: Crystal-clear denoised audio  

4. **Resources**:  
   - [Download Models](https://drive.google.com/drive/folders/1L13BJRqdBrPX8jQj3wwCiI67xC5QIT3S?usp=drive_link)  
   - [Explore More Projects](https://dakeqq.github.io/overview/)  

5. **Note**
   - Please note that the denoiser model optimized on Windows cannot be used on Linux, and vice versa.
---

### 性能 Performance  
| OS           | Device       | Backend           | Model        | Real-Time Factor <br> (Chunk Size: 4000 or 250ms) |
|:------------:|:------------:|:-----------------:|:------------:|:------------------------------------------------:|
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | ZipEnhancer <br> f32 | 0.32                                              |
| macOS 15     | MacBook Air  | CPU <br> M3       | ZipEnhancer <br> f32 | 0.25                                              |
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300 | GTCRN <br> f32       | 0.0036                                            |
| macOS 15     | MacBook Air  | CPU <br> M3       | GTCRN <br> f32       | 0.0013 ~<br> 0.0019                               |  
| Ubuntu-24.04 | Laptop       | CPU <br> i5-7300HQ | DFSMN <br> f32      | 0.0068 ~<br> 0.012                                |
---

### To-Do List  

---

## Audio-Denoiser-ONNX  
通过 ONNX Runtime 实现音频降噪，提升音质清晰度。

### 功能  
1. **支持的模型**：  
   - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
   - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
   - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal/summary)

2. **动态量化**：  
   - **不建议**对 ZipEnhancer 应用动态量化，因为这会由于计算负载增加而显著降低性能。

3. **端到端处理**：  
   - 解决方案内置 `STFT/ISTFT` 处理。  
   - 输入：带噪音的音频  
   - 输出：清晰无噪音的音频  

4. **资源**：  
   - [下载模型](https://drive.google.com/drive/folders/1L13BJRqdBrPX8jQj3wwCiI67xC5QIT3S?usp=drive_link)  
   - [探索更多项目](https://dakeqq.github.io/overview/)  

5. **Note**
   - 请注意，在 Windows 系统上优化的降噪模型无法在 Linux 系统上使用，反之亦然。
---
