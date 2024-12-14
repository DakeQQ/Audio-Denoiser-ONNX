# Audio-Denoiser-ONNX
Utilizes ONNX Runtime for audio denoising.
1. Now support:
   - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
   - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
3. It is not recommended to apply dynamic quantization to ZipEnhancer, as it significantly slows down inference due to the increased computational load.
4. This end-to-end version includes internal `STFT/ISTFT` processing. Input noisy audio; output is crystal clear.
5. [Download](https://drive.google.com/drive/folders/1L13BJRqdBrPX8jQj3wwCiI67xC5QIT3S?usp=drive_link)
6. See more -> https://dakeqq.github.io/overview/

# Audio-Denoiser-ONNX
1. 现在支持:
   - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
   - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
3. 不建议对 ZipEnhancer 应用动态量化，因为这会由于计算负载的增加而显著减慢推理速度。
4. 这个端到端的版本包含 `STFT/ISTFT` 处理。简单的输入噪声音频，输出则是清澈明了的音频。
5. [下载](https://drive.google.com/drive/folders/1L13BJRqdBrPX8jQj3wwCiI67xC5QIT3S?usp=drive_link)
6. See more -> https://dakeqq.github.io/overview/

# 性能 Performance
| OS | Device | Backend | Model | Real-Time Factor<br>( Chunk_Size: 4000 or 250ms ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | ZipEnhancer<br>f32 | 0.32 |
| macOS 15  | MacBook Air | CPU<br>M3 | ZipEnhancer<br>f32 | 0.25 |
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | GTCRN<br>f32 | 0.0036 |
| macOS 15  | MacBook Air | CPU<br>M3 | GTCRN<br>f32 | 0.0013 ~<br>0.0019 |
