import random
import numpy as np
import torch
import enum
from tqdm import tqdm

from audio_metrics import AudioMetrics
from audio_metrics.embed_pipeline import EmbedderPipeline
from audio_metrics.kid import KEY_METRIC_KID_MEAN


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


def maybe_slice_audio(audio_sr_pairs, win_dur=None):
    if win_dur is None:
        yield from audio_sr_pairs
        return
    for audio, sr in audio_sr_pairs:
        win_len = int(sr * win_dur)
        if len(audio) <= win_len:
            yield (audio, sr)
        else:
            for win in windowed_view(audio, win_len):
                yield (win, sr)


class Embedder(enum.Enum):
    VGGISH: enum.auto()
    OPENL3: enum.auto()
    CLAP: enum.auto()


class Distance(enum.Enum):
    FAD: enum.auto()
    MMD: enum.auto()


Embedder = enum.Enum("Embedder", {k: k for k in ("vggish", "openl3", "clap")})
Metric = enum.Enum("Metric", {k: k for k in ("fad", "mmd2")})


class AudioPromptAdherence:
    def __init__(
        self,
        device: str | torch.device | None = None,
        win_dur: float | None = None,
        n_pca: int | None = None,
        embedder: str = Embedder.vggish,
        metric: str = Metric.fad,
    ):
        self.n_pca = n_pca
        embedders = {"emb": self._get_embedder(embedder, device)}
        self.embed_kwargs = {
            "combine_mode": "average",
            "batch_size": 10,
            "max_workers": 10,
        }
        self.pipeline = EmbedderPipeline(embedders)
        _metric, self.metric_key = self._get_metric(metric)
        self.adh1_metrics = AudioMetrics(metrics=[_metric])
        self.adh2_metrics = AudioMetrics(metrics=[_metric])
        self.stem_metrics = AudioMetrics(metrics=[_metric])
        self.win_dur = win_dur

        # hacky: build the key to get the metric value from the AuioMetrics results
        self._key = "_".join(
            [self.metric_key, "emb", embedders["emb"].names[0]]
        )

    def _get_metric(self, name):
        metric = Metric(name)
        # TODO: use enum in AudioMetrics as well, for now it uses strings
        if metric == Metric.fad:
            key = "fad"
            return "fad", key
        if metric == Metric.mmd2:
            key = KEY_METRIC_KID_MEAN
            return "kd", key
        raise NotImplementedError(f"Unsupported metric {metric}")

    def _get_embedder(self, name, device):
        if device is None:
            device = get_device()
        emb = Embedder(name)
        if emb == Embedder.vggish:
            from audio_metrics.vggish import VGGish

            return VGGish(device)
        if emb == Embedder.openl3:
            from audio_metrics.openl3 import OpenL3

            return OpenL3(device)
        if emb == Embedder.clap:
            from audio_metrics.clap import CLAP

            return CLAP(device, intermediate_layers=False)
        raise NotImplementedError(f"Unsupported embedder {emb}")

    def set_background(self, audio_pairs):
        # NOTE: we load all audio into memory
        audio_pairs = list(audio_pairs)
        n_items = len(audio_pairs)
        if self.win_dur is None:
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
        adh1_pairs = maybe_slice_audio(adh1_pairs, self.win_dur)
        embeddings = self.pipeline.embed_join(adh1_pairs, **emb_kwargs)
        del adh1_pairs
        self.adh1_metrics.set_background_data(embeddings)
        self.adh1_metrics.set_pca_projection(self.n_pca)
        del embeddings
        adh2_pairs = mix_pairs(misalign_pairs(audio_pairs))
        adh2_pairs = maybe_slice_audio(adh2_pairs, self.win_dur)
        embeddings = self.pipeline.embed_join(adh2_pairs, **emb_kwargs)
        del adh2_pairs
        self.adh2_metrics.set_background_data(embeddings)
        self.adh2_metrics.set_pca_projection(self.n_pca)
        del embeddings

        stems_only = ((stem, sr) for _, stem, sr in audio_pairs)
        stems_only = maybe_slice_audio(stems_only, self.win_dur)
        # del audio_pairs
        embeddings = self.pipeline.embed_join(stems_only, **emb_kwargs)
        del stems_only
        del audio_pairs
        self.stem_metrics.set_background_data(embeddings)
        self.stem_metrics.set_pca_projection(self.n_pca)

    def compare_to_background(self, audio_pairs):
        audio_pairs = list(audio_pairs)
        mixed = mix_pairs(audio_pairs)
        mixed = maybe_slice_audio(mixed, self.win_dur)
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
        stems_only = maybe_slice_audio(stems_only, self.win_dur)
        stem_embeddings = self.pipeline.embed_join(stems_only, **emb_kwargs)
        stem = self.stem_metrics.compare_to_background(stem_embeddings)
        key = self._key
        m_x_y = max(0, adh1[key])
        m_xp_y = max(0, adh2[key])
        score = (m_xp_y - m_x_y) / (m_xp_y + m_x_y)
        return {
            "audio_prompt_adherence": score,
            "stem_distance": max(0, stem[key]),
            "n_real": adh1["n_real"],
            "n_fake": adh1["n_fake"],
        }

    def _check_minimum_data_size(self, n_items):
        msg = f"The number of PCA components ({self.n_pca}) cannot be larger than the number of embedding vectors ({n_items})"
        assert self.n_pca <= n_items, msg
