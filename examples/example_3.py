import json

import numpy as np
import torch

from audio_metrics import AudioPromptAdherence


def make_signal(sr, audio_len, beat_freq, tone_freq):
    beat_period = int(sr // beat_freq)
    signal = np.zeros(audio_len).astype(np.float32)
    onset_period = beat_period // 10
    for start in range(0, audio_len, beat_period):
        end = min(start + beat_period // 2, audio_len)
        length = end - start
        # tone
        signal[start:end] = np.sin(
            tone_freq * 2 * np.pi * np.arange(length) / sr
        )
        # onset
        end = min(start + onset_period // 2, audio_len)
        length = end - start
        signal[start:end] += np.random.random(length) - 0.5
    peak = np.max(np.abs(signal))
    signal *= 0.5 / peak
    return signal


def cfg_pair(matching=True):
    beat_min = 0.5
    beat_max = 2.0
    rnd1 = np.random.random()
    rnd2 = np.random.random()
    beat_freq_mix = beat_min + rnd1 * (beat_max - beat_min)

    tone_min = 100
    tone_max = 500
    tone_freq_mix = tone_min + rnd2 * (tone_max - tone_min)

    mix_cfg = {"beat_freq": beat_freq_mix, "tone_freq": tone_freq_mix}

    if not matching:
        rnd1 = np.random.random()
        rnd2 = np.random.random()
        beat_freq_mix = beat_min + rnd1 * (beat_max - beat_min)
        tone_freq_mix = tone_min + rnd2 * (tone_max - tone_min)

    beat_freq_stem = 2 ** np.random.randint(-3, 4) * beat_freq_mix
    tone_freq_stem = 2 ** np.random.randint(-3, 4) * tone_freq_mix

    stem_cfg = {"beat_freq": beat_freq_stem, "tone_freq": tone_freq_stem}

    return mix_cfg, stem_cfg


def mix_stem_pair(sr, audio_len, matching=True):
    mix_cfg, stem_cfg = cfg_pair(matching)
    mix = make_signal(sr, audio_len, **mix_cfg)
    stem = make_signal(sr, audio_len, **stem_cfg)
    return (mix, stem, sr)


def audio_pair_generator(n_items, sr, audio_len, matching=True):
    for _ in range(n_items):
        yield mix_stem_pair(sr, audio_len, matching)


def main():
    dev = torch.device("cuda")
    n_items = 300
    sr = 48000
    audio_len = 10 * sr
    win_len = 5 * sr

    real_data = audio_pair_generator(n_items, sr, audio_len, matching=True)
    fake_data_1 = audio_pair_generator(n_items, sr, audio_len, matching=True)
    fake_data_2 = audio_pair_generator(n_items, sr, audio_len, matching=False)
    # import soundfile as sf

    # for i, (mix, stem, sr) in enumerate(real_data):
    #     print(i)
    #     fp = f"/tmp/mg/ex/real_{i}.flac"
    #     sf.write(fp, np.column_stack((mix, stem)), sr)
    # for i, (mix, stem, sr) in enumerate(fake_data_2):
    #     print(i)
    #     fp = f"/tmp/mg/ex/fake2_{i}.flac"
    #     sf.write(fp, np.column_stack((mix, stem)), sr)
    # return
    metric = AudioPromptAdherence(dev, win_len)
    metric.set_background(real_data)
    result = metric.compare_to_background(fake_data_1)
    print("matching")
    print(json.dumps(result, indent=2))
    result = metric.compare_to_background(fake_data_2)
    print("non-matching")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
