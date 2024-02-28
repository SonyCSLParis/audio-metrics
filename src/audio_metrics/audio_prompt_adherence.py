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


def mix_pairs(pairs):
    return [
        (mix_tracks(np.column_stack((mix, stem))), sr)
        for mix, stem, sr in pairs
    ]


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def windowed_view(arr, window_size, hop_size=None):
    if hop_size is None:
        hop_size = window_size
    i = 0
    while i + window_size <= len(arr):
        yield arr[i : i + window_size]
        i += hop_size


def maybe_slice_audio(audio_sr_pairs, win_len=None):
    if win_len is None:
        yield from audio_sr_pairs
        return
    for audio, sr in audio_sr_pairs:
        if len(audio) <= win_len:
            yield (audio, sr)
        else:
            for win in windowed_view(audio, win_len):
                yield (win, sr)


class AudioPromptAdherence:
    def __init__(self, device=None, win_len=None):
        self.n_pca = 100
        if device is None:
            device = get_device()
        embedders = {"clap": CLAP(device)}
        self.embed_kwargs = {
            "combine_mode": "average",
            "batch_size": 10,
            "max_workers": 10,
        }
        self.pipeline = EmbedderPipeline(embedders)
        self.good_metrics = AudioMetrics(metrics=["fad"])
        self.bad_metrics = AudioMetrics(metrics=["fad"])
        self.win_len = win_len

    def set_background(self, audio_pairs):
        # NOTE: we load all audio into memory
        audio_pairs = list(audio_pairs)
        n_items = len(audio_pairs)
        if self.win_len is None:
            self._check_minimum_data_size(n_items)
            total = 2 * n_items
        else:
            total = None

        emb_kwargs = dict(
            self.embed_kwargs,
            progress=tqdm(
                total=total,
                desc="computing background embeddings",
            ),
        )
        good_pairs = mix_pairs(audio_pairs)
        good_pairs = maybe_slice_audio(good_pairs, self.win_len)
        embeddings = self.pipeline.embed_join(good_pairs, **emb_kwargs)
        del good_pairs
        self.good_metrics.set_background_data(embeddings)
        self.good_metrics.set_pca_projection(self.n_pca)
        del embeddings
        bad_pairs = mix_pairs(misalign_pairs(audio_pairs))
        bad_pairs = maybe_slice_audio(bad_pairs, self.win_len)
        del audio_pairs
        embeddings = self.pipeline.embed_join(bad_pairs, **emb_kwargs)
        del bad_pairs
        self.bad_metrics.set_background_data(embeddings)
        self.bad_metrics.set_pca_projection(self.n_pca)
        del embeddings

    def compare_to_background(self, audio_pairs):
        pairs = mix_pairs(audio_pairs)
        pairs = maybe_slice_audio(pairs, self.win_len)
        emb_kwargs = dict(
            self.embed_kwargs,
            progress=tqdm(
                total=None,
                desc="computing candidate embeddings",
            ),
        )
        embeddings = self.pipeline.embed_join(pairs, **emb_kwargs)
        good = self.good_metrics.compare_to_background(embeddings)
        bad = self.bad_metrics.compare_to_background(embeddings)
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
