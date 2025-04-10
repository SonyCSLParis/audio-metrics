import random

import numpy as np
import musdb

from audio_metrics.util.cpu_parallel import cpu_parallel
from audio_metrics import AudioMetrics

# the sample rate of the clap encoder we will use below
sr = 48000


# data preparation helper functions:
def create_ctx_stem_pair(song, rng=None):
    # pick a random stem from the song, and take the average of the remaining
    # stems as context; convert stereo to mono
    rng = rng or random
    n_stems, n_samples, _ = song.stems.shape
    stem_idx = rng.randrange(n_stems)
    mix_idx = list(i for i in range(n_stems) if i != stem_idx)
    out = np.empty((n_samples, 2), dtype=song.stems.dtype)
    # context
    out[:, 0] = np.mean(song.stems[mix_idx], axis=0).mean(axis=1)
    # stem
    out[:, 1] = song.stems[stem_idx].mean(axis=-1)
    return out


def misalign_pairs(pairs):
    N = len(pairs)
    idx = list(range(N))
    random.shuffle(idx)
    for i in idx:
        j = (i + 1) % N
        yield np.stack((pairs[i][:, 0], pairs[j][:, 1]), axis=-1)


musdb_train = musdb.DB(subsets="train", download=True, sample_rate=sr)
musdb_test = musdb.DB(subsets="test", download=True, sample_rate=sr)
reference = cpu_parallel(musdb_train, create_ctx_stem_pair, n_workers=16)
candidate_good = list(cpu_parallel(musdb_test, create_ctx_stem_pair, n_workers=16))
candidate_bad = list(misalign_pairs(candidate_good))

am = AudioMetrics(
    embedder="laion_clap_music",
    mix_function="L0",
    metrics=[
        # prdc and fad evaluated the stems only (contexts are ignored)
        "prdc",  # precision, reacll, density, coverage
        "fad",  # frechet audio distance
        # apa evaluates how well contexts and stems fit together
        "apa",  # accompaniment prompt adherence
    ],
    # Choose a low dimensionality because the MUSDB reference set is quite small (<100)
    n_pca=10,
)
am.add_reference(reference)
print()
result = am.evaluate(candidate_good)
print("Metrics for MUSDB test set")
print(result)
print()

result = am.evaluate(candidate_bad)
print("Metrics for MUSDB test set with misaligned (context, stem) pairs")
print(result)
