import gc
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime
import soundfile as sf
import torch
from pydub import AudioSegment
from STFT_Process import STFT_Process  
from modeling_modified.ulunas_optimized import ULUNAS


model_path          = r"/home/DakeQQ/Downloads/ul-unas-main"                   # The UL-UNAS download path.
onnx_model_A        = r"/home/DakeQQ/Downloads/Ul_Unas_ONNX/Ul_Unas.onnx"      # The exported onnx model path.
test_noisy_audio    = r"./example/0174.wav"                                    # The noisy audio path.
save_denoised_audio = r"./denoised.wav"                                        # The output denoised audio path.


ORT_Accelerate_Providers = []                       # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                    # else keep empty.
DYNAMIC_AXES       = False                          # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
IN_SAMPLE_RATE     = 16000                          # UL-UNAS is designed for 16kHz only.
OUT_SAMPLE_RATE    = 16000                          # UL-UNAS is designed for 16kHz only.
INPUT_AUDIO_LENGTH = 32000                          # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 4096. Higher values yield better quality. It is better to set an integer multiple of the HOP_LENGTH value.
MAX_SIGNAL_LENGTH  = 4096 if DYNAMIC_AXES else 128  # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE        = 'hann'                         # Type of window function used in the STFT. UL-UNAS uses standard hann.
STFT_PAD_MODE      = 'constant'                     # ["constant", "reflect"]
N_MELS             = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT               = 512                            # Number of FFT components for the STFT process
WINDOW_LENGTH      = 512                            # Length of windowing, edit it carefully.
HOP_LENGTH         = 256                            # Number of samples between successive frames in the STFT
MAX_THREADS        = 4                              # Number of parallel threads for test audio denoising.
OPSET              = 17                             # ONNX opset version. Set it to 17 for better performance and compatibility. You can adjust it if you encounter issues with certain providers.
NORMALIZE_AUDIO    = False                          # Normalize the input audio to a target RMS level (e.g., 8192) before processing. It can help improve the performance of the model, especially for low-volume audio. Set it to True if you want to enable it.
REMOVE_DC_OFFSET   = False                          # Keep disabled for parity with the original UL-UNAS inference path.


def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-7))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio
  

def convert_state_dict(original_sd, types):
    """
    Convert state dict keys from the original ULUNAS model (which uses nn.Sequential)
    to the optimized model format (which uses flat named attributes).

    The original model uses:
      - XConvBlock:  ops.{0=pad,1=conv,2=bn,3=act,4=ctfa,5=shuffle}
      - XDWSBlock:   pconv.{0=conv,1=bn,2=act,3=shuffle} / dconv.{0=pad,1=conv,2=bn,3=act,4=ctfa}
      - XMBBlocks:   pconv1.{0=conv,1=bn,2=act,3=shuffle} / dconv.{0=pad,1=conv,2=bn,3=act} / pconv2.{0=conv,1=bn,2=ctfa}

    The optimized model uses:
      - XConvBlock:  pad, conv, bn, act, ctfa, shuffle
      - XDWSBlock:   pconv_conv, pconv_bn, pconv_act, pconv_shuffle / dconv_pad, dconv_conv, dconv_bn, dconv_act, dconv_ctfa
      - XMBBlocks:   pconv1_conv, pconv1_bn, pconv1_act, pconv1_shuffle / dconv_pad, dconv_conv, dconv_bn, dconv_act / pconv2_conv, pconv2_bn, pconv2_ctfa
    """
    # Key mappings for each block type
    xconv_map = [
        ('ops.1.', 'conv.'),
        ('ops.2.', 'bn.'),
        ('ops.3.', 'act.'),
        ('ops.4.', 'ctfa.'),
    ]
    xdws_map = [
        ('pconv.0.', 'pconv_conv.'),
        ('pconv.1.', 'pconv_bn.'),
        ('pconv.2.', 'pconv_act.'),
        ('dconv.1.', 'dconv_conv.'),
        ('dconv.2.', 'dconv_bn.'),
        ('dconv.3.', 'dconv_act.'),
        ('dconv.4.', 'dconv_ctfa.'),
    ]
    xmb_map = [
        ('pconv1.0.', 'pconv1_conv.'),
        ('pconv1.1.', 'pconv1_bn.'),
        ('pconv1.2.', 'pconv1_act.'),
        ('dconv.1.', 'dconv_conv.'),
        ('dconv.2.', 'dconv_bn.'),
        ('dconv.3.', 'dconv_act.'),
        ('pconv2.0.', 'pconv2_conv.'),
        ('pconv2.1.', 'pconv2_bn.'),
        ('pconv2.2.', 'pconv2_ctfa.'),
    ]
    block_maps = {0: xconv_map, 1: xdws_map, 2: xmb_map}

    # Decoder block types (reversed order from encoder)
    n_blocks = len(types)
    decoder_types = [types[i] for i in range(n_blocks - 1, 0, -1)] + [types[0]]

    new_sd = {}
    for key, value in original_sd.items():
        new_key = key

        # Handle encoder blocks
        if key.startswith('encoder.en_convs.'):
            parts = key.split('.', 3)  # ['encoder', 'en_convs', idx, remainder]
            block_idx = int(parts[2])
            remainder = parts[3]
            block_type = types[block_idx]
            for old_prefix, new_prefix in block_maps[block_type]:
                if remainder.startswith(old_prefix):
                    new_key = f'encoder.en_convs.{block_idx}.{new_prefix}{remainder[len(old_prefix):]}'
                    break

        # Handle decoder blocks
        elif key.startswith('decoder.de_convs.'):
            parts = key.split('.', 3)  # ['decoder', 'de_convs', idx, remainder]
            block_idx = int(parts[2])
            remainder = parts[3]
            block_type = decoder_types[block_idx]
            for old_prefix, new_prefix in block_maps[block_type]:
                if remainder.startswith(old_prefix):
                    new_key = f'decoder.de_convs.{block_idx}.{new_prefix}{remainder[len(old_prefix):]}'
                    break

        if new_key.endswith('affine_weight') or new_key.endswith('affine_bias'):
            value = value.view(1, value.shape[0], 1, value.shape[1])
        elif new_key.endswith('slope_weight'):
            value = value.view(1, value.shape[0], 1, 1)

        new_sd[new_key] = value

    return new_sd


class ULUNAS_CUSTOM(torch.nn.Module):
    def __init__(self, ulunas, stft_model, istft_model, in_sample_rate, out_sample_rate, remove_dc_offset=False):
        super(ULUNAS_CUSTOM, self).__init__()
        self.ulunas = ulunas
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.inv_int16 = float(1.0 / 32768.0)
        self.output_pcm_scale = 32767.0
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.out_sample_rate_scale = out_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before_centering = self.in_sample_rate_scale > 1.0
        self.resample_after_centering = self.in_sample_rate_scale < 1.0
        self.output_resample_before_pcm = self.out_sample_rate_scale > 1.0
        self.output_resample_after_pcm = self.out_sample_rate_scale < 1.0
        self.remove_dc_offset = remove_dc_offset

    def forward(self, audio):
        audio = audio.float()
        if self.resample_before_centering:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        audio = audio * self.inv_int16
        if self.resample_after_centering:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        if self.remove_dc_offset:
            audio = audio - torch.mean(audio, dim=-1, keepdim=True)
        real_part, imag_part = self.stft_model(audio)
        magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part + 1e-12)
        # UL-UNAS returns a real-valued sigmoid mask (1, F, T)
        mask = self.ulunas(magnitude)
        # Apply mask to both real and imag (simple magnitude masking)
        s_real = real_part * mask
        s_imag = imag_part * mask
        audio = self.istft_model(s_real, s_imag)
        if self.output_resample_before_pcm:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        audio *= self.output_pcm_scale
        if self.output_resample_after_pcm:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode=STFT_PAD_MODE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode=STFT_PAD_MODE).eval()
    ulunas = ULUNAS().eval()
    ckpt = torch.load(model_path + "/checkpoints/model_trained_on_dns3.tar", map_location='cpu', weights_only=False)
    converted_sd = convert_state_dict(ckpt['model'], types=[0, 2, 1, 2, 1])
    ulunas.load_state_dict(converted_sd, strict=False)
    ulunas.fuse_bn_()  # Fuse BatchNorm into Conv weights for optimized inference
    ulunas = ULUNAS_CUSTOM(ulunas.float(), custom_stft, custom_istft, IN_SAMPLE_RATE, OUT_SAMPLE_RATE, remove_dc_offset=REMOVE_DC_OFFSET)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        ulunas,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del ulunas
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
print('\nExport done!\n\nStart to run Ul-Unas by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 1       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 1       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


# Load the input audio
print(f"\nTest Input Audio: {test_noisy_audio}")
audio = np.array(AudioSegment.from_file(test_noisy_audio).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
if NORMALIZE_AUDIO:
    audio = normalise_audio(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(30 * IN_SAMPLE_RATE, audio_len)  # Default to slice in 30 seconds. You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int) & (OUT_SAMPLE_RATE == IN_SAMPLE_RATE):
        stride_step = shape_value_out
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]
inv_audio_len = float(100.0 / aligned_len)


audio_len = int(audio_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Start to run Ul-Unas
print("\nRunning the Ul-Unas by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel denoised the audio.
    futures = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, audio))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.3f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:audio_len]
end_time = time.time()
print(f"Complete: 100.00%")

# Save the denoised wav.
sf.write(save_denoised_audio, denoised_wav, OUT_SAMPLE_RATE, format='WAVEX')
elapsed_time = end_time - start_time
audio_duration = audio_len / OUT_SAMPLE_RATE
rtf = elapsed_time / audio_duration
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {elapsed_time:.3f} Seconds\nAudio Duration: {audio_duration:.3f} Seconds\nRTF: {rtf:.4f}")
