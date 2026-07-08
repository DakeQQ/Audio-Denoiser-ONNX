from pathlib import Path


EXAMPLE_AUDIO_ROOT = Path(__file__).resolve().parent / "Test_Examples"

_MODEL_AUDIO_FILES = {
    "deep_echo_aec": (("near_end", "aec/nearend_mic1.wav"), ("far_end", "aec/farend_speech1.wav")),
    "dfsmn": (("noisy", "denoise/speech_with_noise_48k.wav"),),
    "dfsmn_aec": (("near_end", "aec/nearend_mic1.wav"), ("far_end", "aec/farend_speech1.wav")),
    "gtcrn": (("noisy", "denoise/gtcrn_mix.wav"),),
    "h_gtcrn": (("noisy", "denoise/h_gtcrn_noisy.wav"),),
    "mel_band_roformer": (("noisy", "denoise/mel_band_roformer.wav"),),
    "mossformer2_se_48k": (("noisy", "denoise/speech_with_noise1.wav"),),
    "mossformer2_ss_16k": (("mixed", "separation/mixed_speech.wav"),),
    "mossformer2_super_resolution": (("source", "super_resolution/basic_ref_zh.wav"),),
    "mossformergan_se_16k": (("noisy", "denoise/speech_with_noise1.wav"),),
    "nkf_aec": (("near_end", "aec/nearend_mic1.wav"), ("far_end", "aec/farend_speech1.wav")),
    "sdaec": (("near_end", "aec/nearend_mic1.wav"), ("far_end", "aec/farend_speech1.wav")),
    "ul_unas": (("noisy", "denoise/ul_unas_0174.wav"),),
    "zipenhancer": (("noisy", "denoise/speech_with_noise1.wav"),),
}


def example_audio_path(relative_path):
    return str(EXAMPLE_AUDIO_ROOT / relative_path)


def model_audio_cases(model_name):
    try:
        audio_files = _MODEL_AUDIO_FILES[model_name]
    except KeyError as exc:
        names = ", ".join(sorted(_MODEL_AUDIO_FILES))
        raise ValueError(f"Unknown demo audio model {model_name!r}. Available models: {names}") from exc
    return [(example_audio_path(relative_path), case_name) for case_name, relative_path in audio_files]


def model_audio_paths(model_name):
    return [path for path, _case_name in model_audio_cases(model_name)]


def model_audio_path(model_name, case_name=None):
    cases = model_audio_cases(model_name)
    if case_name is None:
        if len(cases) != 1:
            names = ", ".join(case for _path, case in cases)
            raise ValueError(f"Model {model_name!r} has multiple demo audio cases: {names}")
        return cases[0][0]

    for path, name in cases:
        if name == case_name:
            return path

    names = ", ".join(name for _path, name in cases)
    raise ValueError(f"Unknown demo audio case {case_name!r} for model {model_name!r}. Available cases: {names}")