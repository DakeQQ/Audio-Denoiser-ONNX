# Audio-Denoiser-ONNX
Utilizes ONNX Runtime for audio denoising.
1. Add [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) model.
2. It is not recommended to apply dynamic quantization to ZipEnhancer, as it significantly slows down inference due to the increased computational load.
3. This end-to-end version includes internal STFT/ISTFT processing. Input noisy audio; output is crystal clear.
4. [Download](https://drive.google.com/drive/folders/1L13BJRqdBrPX8jQj3wwCiI67xC5QIT3S?usp=drive_link)
5. See more -> https://dakeqq.github.io/overview/

# Audio-Denoiser-ONNX
1. 添加 [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) 模型。
2. 不建议对 ZipEnhancer 应用动态量化，因为这会由于计算负载的增加而显著减慢推理速度。
3. 这个端到端的版本包含 `STFT/ISTFT` 处理。简单的输入噪声音频，输出则是清澈明了的音频。
4. [下载](https://drive.google.com/drive/folders/1L13BJRqdBrPX8jQj3wwCiI67xC5QIT3S?usp=drive_link)
5. See more -> https://dakeqq.github.io/overview/

# 性能 Performance
| OS | Device | Backend | Model | Real-Time Factor<br>( Chunk_Size: 4000 ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | ZipEnhancer<br>f32 | 0.33 |
