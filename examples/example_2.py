import json

import numpy as np
import torch

from audio_metrics import AudioPromptAdherence


def random_audio_pair_generator(n_items, sr, audio_len):
    for _ in range(n_items):
        mix = np.random.random(audio_len).astype(np.float32) - 0.5
        stem = np.random.random(audio_len).astype(np.float32) - 0.5
        yield (mix, stem, sr)


def main():
    dev = torch.device("cuda")
    n_items = 150
    sr = 48000
    audio_len = 10 * sr
    win_len = 5 * sr
    metric = AudioPromptAdherence(dev, win_len)

    real_data = random_audio_pair_generator(n_items, sr, audio_len)
    fake_data = random_audio_pair_generator(n_items, sr, audio_len)

    metric.set_background(real_data)
    res = metric.compare_to_background(fake_data)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
