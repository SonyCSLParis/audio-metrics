import random
import numpy as np
import torch
from tqdm import tqdm

from audio_metrics import AudioMetrics
from audio_metrics.embed_pipeline import EmbedderPipeline

from audio_metrics.clap import CLAP
from audio_metrics.vggish import VGGish
from audio_metrics.openl3 import OpenL3


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
    return (
        (mix_tracks(np.column_stack((mix, stem))), sr)
        for mix, stem, sr in pairs
    )


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
        self.adh1_metrics = AudioMetrics(metrics=["fad"])
        self.adh2_metrics = AudioMetrics(metrics=["fad"])
        self.stem_metrics = AudioMetrics(metrics=["fad"])
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
        adh1_pairs = mix_pairs(audio_pairs)
        adh1_pairs = maybe_slice_audio(adh1_pairs, self.win_len)
        embeddings = self.pipeline.embed_join(adh1_pairs, **emb_kwargs)
        del adh1_pairs
        self.adh1_metrics.set_background_data(embeddings)
        self.adh1_metrics.set_pca_projection(self.n_pca)
        del embeddings
        adh2_pairs = mix_pairs(misalign_pairs(audio_pairs))
        adh2_pairs = maybe_slice_audio(adh2_pairs, self.win_len)
        embeddings = self.pipeline.embed_join(adh2_pairs, **emb_kwargs)
        del adh2_pairs
        self.adh2_metrics.set_background_data(embeddings)
        self.adh2_metrics.set_pca_projection(self.n_pca)
        del embeddings

        stems_only = ((stem, sr) for _, stem, sr in audio_pairs)
        stems_only = maybe_slice_audio(stems_only, self.win_len)
        # del audio_pairs
        embeddings = self.pipeline.embed_join(stems_only, **emb_kwargs)
        del stems_only
        del audio_pairs
        self.stem_metrics.set_background_data(embeddings)
        self.stem_metrics.set_pca_projection(self.n_pca)

    def compare_to_background(self, audio_pairs):
        audio_pairs = list(audio_pairs)
        mixed = mix_pairs(audio_pairs)
        mixed = maybe_slice_audio(mixed, self.win_len)
        emb_kwargs = dict(
            self.embed_kwargs,
            progress=tqdm(
                total=None,
                desc="computing candidate embeddings",
            ),
        )
        mixed_embeddings = self.pipeline.embed_join(mixed, **emb_kwargs)
        adh1 = self.adh1_metrics.compare_to_background(mixed_embeddings)
        adh2 = self.adh2_metrics.compare_to_background(mixed_embeddings)
        del mixed_embeddings
        stems_only = ((stem, sr) for _, stem, sr in audio_pairs)
        stems_only = maybe_slice_audio(stems_only, self.win_len)
        stem_embeddings = self.pipeline.embed_join(stems_only, **emb_kwargs)
        stem = self.stem_metrics.compare_to_background(stem_embeddings)
        key = "fad_clap_output"
        score = (adh2[key] - adh1[key]) / (adh2[key] + adh1[key])
        return {
            "audio_prompt_adherence": score,
            "stem_fad_clap": stem[key],
            "n_real": adh1["n_real"],
            "n_fake": adh1["n_fake"],
        }

    def _check_minimum_data_size(self, n_items):
        msg = f"The number of PCA components ({self.n_pca}) cannot be larger than the number of embedding vectors ({n_items})"
        assert self.n_pca <= n_items, msg
