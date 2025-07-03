import gc
import time
import yaml
from ml_collections import ConfigDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime
import torch
import soundfile as sf
from pydub import AudioSegment
from modeling_modified.mel_band_roformer import MelBandRoformer
from STFT_Process import STFT_Process                                                           # The custom STFT/ISTFT can be exported in ONNX format.


project_path = "/home/DakeQQ/Downloads/Mel-Band-Roformer-Vocal-Model-main"                        # The Mel-Band-Roformer GitHub project path.
model_path = "/home/DakeQQ/Downloads/Mel-Band-Roformer-Vocal-Model-main/MelBandRoformer.ckpt"     # The model download path.
onnx_model_A = "/home/DakeQQ/Downloads/MelBandRoformer_ONNX/MelBandRoformer.onnx"                 # The exported onnx model path.
test_noisy_audio = "./test.wav"                                                                 # The noisy audio path.
save_denoised_audio = "./test_denoised.wav"                                                     # The output denoised audio path.

ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 44100              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 22050 and less than 362800. Higher values yield better quality but time consume. It is better to set an integer multiple of the HOP_LENGTH value.
MAX_SIGNAL_LENGTH = 1024 if DYNAMIC_AXES else (INPUT_AUDIO_LENGTH // 100 + 1)  # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
NFFT = 2048                             # Number of FFT components for the STFT process
WINDOW_LENGTH = 1920                     # Length of windowing, edit it carefully.
HOP_LENGTH = 441                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 44100                     # The MelBandRoformer parameter, do not edit the value.
MAX_THREADS = 4                         # Number of parallel threads for test audio denoising.


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)
  

class MelBandRoformer_Modified(torch.nn.Module):
    def __init__(self, mel_band_roformer, stft_model, istft_model, nfft, max_signal_len):
        super(MelBandRoformer_Modified, self).__init__()
        self.mel_band_roformer = mel_band_roformer
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.inv_int16 = float(1.0 / 32768.0)
        self.zeros = torch.zeros((1, 1, (nfft // 2 + 1) * 2, max_signal_len, 2), dtype=torch.int8)

    def forward(self, audio):
        audio = audio * self.inv_int16
        audio_L, audio_R = torch.split(audio, [1, 1], dim=1)
        real_L, imag_L = self.stft_model(audio_L, 'constant')
        real_R, imag_R = self.stft_model(audio_R, 'constant')
        stft_repr_L = torch.stack((real_L, imag_L), dim=-1)
        stft_repr_R = torch.stack((real_R, imag_R), dim=-1)
        stft_repr = torch.stack((stft_repr_L, stft_repr_R), dim=1)
        real_part, imag_part = self.mel_band_roformer(stft_repr, self.zeros)
        real_L, real_R = torch.split(real_part, [1, 1], dim=0)
        imag_L, imag_R = torch.split(imag_part, [1, 1], dim=0)
        audio_L = custom_istft(real_L, imag_L)
        audio_R = custom_istft(real_R, imag_R)
        audio = torch.cat((audio_L, audio_R), dim=1)
        return (audio * 32768.0).clamp(min=-32768.0, max=32767.0).to(torch.int16)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    with open(project_path + "/configs/config_vocals_mel_band_roformer.yaml") as f:
      config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    model = MelBandRoformer(**dict(config.model))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    mel_band_roformer = MelBandRoformer_Modified(model.eval(), custom_stft, custom_istft, NFFT, MAX_SIGNAL_LENGTH)
    audio = torch.ones((1, 2, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        mel_band_roformer,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del mel_band_roformer
    del model
    del audio
    del custom_stft
    del custom_istft
    gc.collect()
print('\nExport done!\n\nStart to run MelBandRoformer by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
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
audio = AudioSegment.from_file(test_noisy_audio).set_frame_rate(SAMPLE_RATE)
audio_channels = audio.channels
audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
if audio_channels == 1:
    audio = audio.reshape(1, 1, -1)
    audio = np.concatenate((audio, audio), axis=1)
else:
    audio = audio.reshape(1, -1, 2).transpose(0, 2, 1)
audio_len = audio.shape[-1]
inv_audio_len = float(100.0 / audio_len)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_in_channel = ort_session_A._inputs_meta[0].shape[1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(SAMPLE_RATE, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int):
        stride_step = shape_value_out
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, shape_value_in_channel, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, shape_value_in_channel, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]


# Start to run Mel-Band-Roformer
print("\nRunning the MelBandRoformer by ONNX Runtime.")
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
denoised_wav = np.concatenate(saved, axis=-1)[0, :, :audio_len].transpose()
end_time = time.time()
print(f"Complete: 100.00%")

# Save the denoised wav.
sf.write(save_denoised_audio, denoised_wav, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
