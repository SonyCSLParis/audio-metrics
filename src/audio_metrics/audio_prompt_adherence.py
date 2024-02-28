import random
import numpy as np
import torch
from tqdm import tqdm

from audio_metrics import AudioMetrics
from audio_metrics.clap import CLAP
from audio_metrics.embed_pipeline import EmbedderPipeline

# from audio_metrics.vggish import VGGish
# from audio_metrics.openl3 import OpenL3


def mix_tracks(audio):
    """Mix channels preserving peak amplitude.

    audio: samples x channels

    """

    assert len(audio.shape) == 2
    # n_ch = audio.shape[1]
    if audio.shape[1] == 1:
        return audio[:, 0]
    vmax_orig = np.abs(audio).max()
    if vmax_orig <= 0:
        return audio[:, 0]
    mix = np.mean(audio, 1)
    vmax_new = np.abs(mix).max()
    gain = vmax_orig / vmax_new
    mix *= gain
    return mix


def misalign_pairs(pairs):
    N = len(pairs)
    perm = np.random.permutation(N)
    for i in range(N):
        k = perm[i]
        l = perm[(i + 1) % N]
        mix = pairs[k][0]
        stem = pairs[l][1]
        sr = pairs[k][2]
        yield (mix, stem, sr)


class AudioPromptAdherence:
    def __init__(self, dev):
        self.n_pca = 100
        embedders = {
            # "vggish": VGGish(dev),
            "clap": CLAP(dev),
            # "openl3": OpenL3(dev),
        }
        self.embed_kwargs = {
            "combine_mode": "average",  # 'stack'
            "batch_size": 10,
            "max_workers": 10,
        }
        self.pipeline = EmbedderPipeline(embedders)
        self.good_metrics = AudioMetrics(metrics=["fad"])
        self.bad_metrics = AudioMetrics(metrics=["fad"])

    def set_background(self, audio_pairs):
        # NOTE: we load all audio into memory
        audio_pairs = list(audio_pairs)
        n_items = len(audio_pairs)
        self._check_minimum_data_size(n_items)
        progress = tqdm(
            total=2 * n_items, desc="computing background embeddings"
        )
        good_pairs = [
            (mix_tracks(np.column_stack((mix, stem))), sr)
            for mix, stem, sr in audio_pairs
        ]
        embeddings = self.pipeline.embed_join(
            good_pairs, **self.embed_kwargs, progress=progress
        )
        del good_pairs
        self.good_metrics.set_background_data(embeddings)
        self.good_metrics.set_pca_projection(self.n_pca)
        del embeddings
        bad_pairs = [
            (mix_tracks(np.column_stack((mix, stem))), sr)
            for mix, stem, sr in misalign_pairs(audio_pairs)
        ]
        del audio_pairs
        embeddings = self.pipeline.embed_join(
            bad_pairs, **self.embed_kwargs, progress=progress
        )
        del bad_pairs
        self.bad_metrics.set_background_data(embeddings)
        self.bad_metrics.set_pca_projection(self.n_pca)
        del embeddings

    def compare_to_background(self, audio_pairs):
        pairs = [
            (mix_tracks(np.column_stack((mix, stem))), sr)
            for mix, stem, sr in audio_pairs
        ]
        embeddings = self.pipeline.embed_join(pairs, **self.embed_kwargs)
        good = self.good_metrics.compare_to_background(embeddings)
        bad = self.bad_metrics.compare_to_background(embeddings)
        # result = {}
        # for k in keys:
        #     if k in copy_keys:
        #         result[k] = good[k]
        #     else:
        #         score = (bad[k] - good[k]) / (bad[k] + good[k])
        #         result[k] = score
        key = "fad_clap_output"
        score = (bad[key] - good[key]) / (bad[key] + good[key])
        return {
            "audio_prompt_adherence": score,
            "n_real": good["n_real"],
            "n_fake": good["n_fake"],
        }

    def _check_minimum_data_size(self, n_items):
        msg = f"The number of PCA components ({self.n_pca}) cannot be larger than the number of embedding vectors ({n_items})"
        assert self.n_pca <= n_items, msg
