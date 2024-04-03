from pathlib import Path
import numpy as np
import soundfile as sf


def make_signal(sr, audio_len, beat_rate, tone_freq):
    """Make a signal with a that combines a noise onset and a tone at the
    specified rate and frequency.
    """

    beat_period = int(sr // beat_rate)
    signal = np.zeros(audio_len).astype(np.float32)
    onset_period = beat_period // 10
    tone_length = beat_period // 2
    onset_length = onset_period // 2
    tone_sig = np.sin(tone_freq * 2 * np.pi * np.arange(tone_length) / sr)
    onset_sig = np.random.random(onset_length) - 0.5
    for start in range(0, audio_len, beat_period):
        end = min(start + beat_period // 2, audio_len)
        length = end - start
        # tone
        signal[start:end] = tone_sig[:length]
        end = min(start + onset_period // 2, audio_len)
        length = end - start
        # onset
        signal[start:end] += onset_sig[:length]
    peak = np.max(np.abs(signal))
    signal *= 0.5 / peak
    return signal


def cfg_pair(matching=True):
    """
    Create two pairs of beat_rate and tone_freq.  When `matching=True` both rate
    and freq have a harmonic relationship across the pairs, otherwise they are
    unrelated.
    """

    beat_min = 0.5
    beat_max = 2.0
    rnd1 = np.random.random()
    rnd2 = np.random.random()
    beat_rate_mix = beat_min + rnd1 * (beat_max - beat_min)

    tone_min = 100
    tone_max = 500
    tone_freq_mix = tone_min + rnd2 * (tone_max - tone_min)

    mix_cfg = {"beat_rate": beat_rate_mix, "tone_freq": tone_freq_mix}

    if not matching:
        rnd1 = np.random.random()
        rnd2 = np.random.random()
        beat_rate_mix = beat_min + rnd1 * (beat_max - beat_min)
        tone_freq_mix = tone_min + rnd2 * (tone_max - tone_min)

    beat_rate_stem = 2 ** np.random.randint(-3, 4) * beat_rate_mix
    tone_freq_stem = 2 ** np.random.randint(-3, 4) * tone_freq_mix

    stem_cfg = {"beat_rate": beat_rate_stem, "tone_freq": tone_freq_stem}

    return mix_cfg, stem_cfg


def mix_stem_pair(sr, audio_len, matching=True):
    """Create a pair of audio signals of the specified length."""
    mix_cfg, stem_cfg = cfg_pair(matching)
    mix = make_signal(sr, audio_len, **mix_cfg)
    stem = make_signal(sr, audio_len, **stem_cfg)
    return (mix, stem, sr)


def audio_pair_generator(n_items, sr, audio_len, matching=True):
    for _ in range(n_items):
        yield mix_stem_pair(sr, audio_len, matching)


def generate_audio_samples(audio_dir, n_items=100, sr=48000, audio_len=None):
    if audio_len is None:
        audio_len = 10 * sr
    real_data = audio_pair_generator(n_items, sr, audio_len, matching=True)
    fake_data = audio_pair_generator(n_items, sr, audio_len, matching=False)
    audio_dir = Path(audio_dir)
    real_dir = audio_dir / "real"
    fake_dir = audio_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    for i, (mix, stem, sr) in enumerate(real_data):
        fp = real_dir / f"sample_{i:02d}.wav"
        sf.write(fp, np.column_stack((mix, stem)), sr)

    for i, (mix, stem, sr) in enumerate(fake_data):
        fp = fake_dir / f"sample_{i:02d}.wav"
        sf.write(fp, np.column_stack((mix, stem)), sr)
