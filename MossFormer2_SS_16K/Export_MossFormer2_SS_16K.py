import gc
import shutil
import time
import site
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
import torch
from pydub import AudioSegment

model_path = "/home/DakeQQ/Downloads/MossFormer2_SS_16K"                              # The MossFormer2_SS_16K download folder.
onnx_model_A = "/home/DakeQQ/Downloads/MossFormer_ONNX/MossFormer2_SS_16K.onnx"       # The exported onnx model path.
test_mixed_audio = "./examples/test.wav"                                              # The mixed audio path.
save_separated_0 = "separated_0.wav"                                                  # The output separated audio path.
save_separated_1 = "separated_1.wav"                                                  # The output separated audio path.


DYNAMIC_AXES = False                    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
KEEP_ORIGINAL_SAMPLE_RATE = True        # If False, the model outputs audio at 48kHz; otherwise, it uses the original sample rate.
SAMPLE_RATE = 16000                     # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH = 16000              # Maximum input audio length: the length of the audio input signal (in samples) is recommended to be greater than 8000 and less than 96000. Higher values yield better quality but time consume.
MAX_THREADS = 4                         # Number of parallel threads for inference.
PAD_HEAD = 8000                         # ~0.5 Seconds


SAMPLE_RATE_SCALE = float(16000.0 / SAMPLE_RATE)

site_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/__init__.py", site_package_path + "/clearvoice/__init__.py")
shutil.copyfile("./modeling_modified/network_wrapper.py", site_package_path + "/clearvoice/network_wrapper.py")
shutil.copyfile("./modeling_modified/mossformer2.py", site_package_path + "/clearvoice/models/mossformer2_ss/mossformer2.py")
from clearvoice import ClearVoice


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class MOSSFORMER_SS(torch.nn.Module):
    def __init__(self, mossformer_ss, input_audio_len, sample_rate):
        super(MOSSFORMER_SS, self).__init__()
        self.mossformer_ss = mossformer_ss
        self.sample_rate = sample_rate
        self.inv_int16 = 1.0 / 32768.0
        self.norm_factor = float(10.0 ** (-25.0 / 20.0))

        t = torch.arange(int(2000.0 * input_audio_len / sample_rate), dtype=torch.float32)
        sinu = t.unsqueeze(-1) * self.mossformer_ss.mask_net.pos_enc.inv_freq
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        self.emb_pos = (emb * self.mossformer_ss.mask_net.pos_enc.scale).transpose(0, -1)
        self.emb_pos = self.emb_pos.unsqueeze(0).half()

    def norm_audio(self, x, EPS=1e-6):
        rms = torch.sqrt((x ** 2).mean())
        scalar = self.norm_factor / (rms + EPS)
        x = x * scalar
        pow_x = x ** 2
        avg_pow_x = pow_x.mean()
        rmsx = torch.sqrt(pow_x[pow_x > avg_pow_x].mean())
        scalarx = self.norm_factor / (rmsx + EPS)
        x = x * scalarx
        return x, 1.0 / (scalar * scalarx + EPS)

    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        if SAMPLE_RATE_SCALE < 1.0:
            audio, scale = self.norm_audio(audio)
            if self.sample_rate != 16000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
        else:
            if self.sample_rate != 16000:
                audio = torch.nn.functional.interpolate(
                    audio,
                    scale_factor=SAMPLE_RATE_SCALE,
                    mode='linear',
                    align_corners=True
                )
            audio, scale = self.norm_audio(audio)
        rms_in = torch.sqrt((audio ** 2).mean()) * (scale * 32767.0)
        x = self.mossformer_ss.enc(audio)
        mask = self.mossformer_ss.mask_net.norm(x)
        mask = self.mossformer_ss.mask_net.conv1d_encoder(mask)
        x_len = x.shape[-1].unsqueeze(0)
        mask = mask + self.emb_pos[..., :x_len].float()
        mask = self.mossformer_ss.mask_net.mdl(mask)
        mask = self.mossformer_ss.mask_net.prelu(mask)
        mask = self.mossformer_ss.mask_net.conv1d_out(mask)
        mask = mask.view(self.mossformer_ss.mask_net.num_spks, -1, x_len)
        mask = self.mossformer_ss.mask_net.output(mask) * self.mossformer_ss.mask_net.output_gate(mask)
        mask = self.mossformer_ss.mask_net.conv1_decoder(mask)
        mask = mask.view(1, self.mossformer_ss.mask_net.num_spks, -1, x_len)
        mask = self.mossformer_ss.mask_net.activation(mask)
        x = torch.stack([x, x], dim=1)
        sep_x = x * mask
        audio_out = [self.mossformer_ss.dec.forward(sep_x[:, i]) for i in range(self.mossformer_ss.num_spks)]
        for i, wav in enumerate(audio_out):
            rms_out = torch.sqrt((wav ** 2).mean())
            audio_out[i] = (wav * rms_in / rms_out).clamp(min=-32768.0, max=32767.0).to(torch.int16)
        return audio_out


print('Export start ...')
with torch.inference_mode():
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'], model_path=model_path)
    mossformer = myClearVoice.models[0].model.eval().float().to("cpu")
    mossformer = MOSSFORMER_SS(mossformer, INPUT_AUDIO_LENGTH, SAMPLE_RATE)
    audio = torch.randint(low=-32768, high=32767, size=(1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)

    torch.onnx.export(
        mossformer,
        (audio,),
        onnx_model_A,
        input_names=['mix_audio'],
        output_names=['separated_0', 'separated_1'],
        do_constant_folding=True,
        dynamic_axes={
            'mix_audio': {2: 'audio_len'},
            'separated_0': {1: 'audio_len'},
            'separated_1': {1: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del mossformer
    del audio
    gc.collect()
print('\nExport done!\n\nStart to run MossFormer_SS by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = in_name_A[0].name
out_name_A = [i.name for i in out_name_A]


# Load the input audio
print(f"\nTest Input Audio: {test_mixed_audio}")
audio = np.array(AudioSegment.from_file(test_mixed_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio = np.concatenate([np.zeros(PAD_HEAD, dtype=np.int16), audio], axis=0)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = max(SAMPLE_RATE * 6, audio_len)
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int) & KEEP_ORIGINAL_SAMPLE_RATE:
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


if SAMPLE_RATE != 16000 and not KEEP_ORIGINAL_SAMPLE_RATE:
    SAMPLE_RATE = 16000
    audio_len = int(audio_len * SAMPLE_RATE_SCALE)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    separated_0, separated_1 = ort_session_A.run(out_name_A, {in_name_A: _audio[:, :, _slice_start: _slice_end]})
    return _slice_start * _inv_audio_len, separated_0, separated_1


# Start to run MossFormer_SS
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel separate the audio.
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
saved_0 = [result[1] for result in results]
saved_1 = [result[2] for result in results]
separated_0 = np.concatenate(saved_0, axis=-1).reshape(-1)[PAD_HEAD:audio_len]
separated_1 = np.concatenate(saved_1, axis=-1).reshape(-1)[PAD_HEAD:audio_len]
end_time = time.time()
print(f"Complete: 100.00%")

# Save the separated wav.
sf.write(save_separated_0, separated_0, SAMPLE_RATE, format='WAVEX')
sf.write(save_separated_1, separated_1, SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_separated_0} & {save_separated_1}.\n\nTime Cost: {end_time - start_time:.3f} Seconds")
