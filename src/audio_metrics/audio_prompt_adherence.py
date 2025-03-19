import warnings
import pickle
import enum
import numpy as np
import torch
from tqdm import tqdm

from . import AudioMetrics, Embedder
from .embed_pipeline import EmbedderPipeline
from .kid import KEY_METRIC_KID_MEAN
from .mix_functions import MIX_FUNCTIONS


def mix_pairs(pairs, mix_func):
    return (
        # (mix_tracks_loudness(np.column_stack((mix, stem)), sr), sr)
        (mix_func(np.column_stack((mix, stem)), sr), sr)
        for mix, stem, sr in pairs
    )


def misalign(pairs):
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


Metric = enum.Enum("Metric", {k: k for k in ("fad", "mmd")})


class AudioPromptAdherence:
    def __init__(
        self,
        device: str | torch.device | None = None,
        win_dur: float | None = None,
        n_pca: int | None = None,
        pca_whiten: bool = True,
        embedder: str = Embedder.clap_music,
        layer: str | None = None,
        metric: str = Metric.fad,
        mix_func: str = "L2",
    ):
        self.n_pca = n_pca
        self.pca_whiten = pca_whiten
        self.mix_func = MIX_FUNCTIONS.get(mix_func, "L2")
        self._emb_key = "emb"
        embedders = {
            self._emb_key: self._get_embedder(embedder, layer=layer, device=device)
        }
        self.embed_kwargs = {
            "combine_mode": "average",
            "batch_size": 10,
            "max_workers": 10,
        }
        self.pipeline = EmbedderPipeline(embedders)
        _metric, self.metric_key = self._get_metric(metric)
        self.metrics_1 = AudioMetrics(metrics=[_metric])
        self.metrics_2 = AudioMetrics(metrics=[_metric])
        self.m_x_xp = None
        self.win_dur = win_dur
        # hacky: build the key to get the metric value from the AuioMetrics results
        self.layer = layer or embedders["emb"].names[0]
        self._key = "_".join([self.metric_key, "emb", self.layer])

    def save_state(self, fp):
        joint = {}
        d1 = self.metrics_1.__getstate__()
        d2 = self.metrics_2.__getstate__()
        for k, v in d1.items():
            joint["metrics_1/" + k] = v
        for k, v in d2.items():
            joint["metrics_2/" + k] = v
        joint["m_x_xp"] = self.m_x_xp
        with open(fp, "wb") as f:
            pickle.dump(joint, f)

    def load_state(self, fp):
        with open(fp, "rb") as f:
            joint = pickle.load(f)
        prefix1 = "metrics_1/"
        prefix2 = "metrics_2/"
        d1 = {
            k.removeprefix(prefix1): v for k, v in joint.items() if k.startswith(prefix1)
        }
        d2 = {
            k.removeprefix(prefix2): v for k, v in joint.items() if k.startswith(prefix2)
        }
        self.metrics_1.__setstate__(d1)
        self.metrics_2.__setstate__(d2)
        self.m_x_xp = joint["m_x_xp"]

    def _get_metric(self, name):
        metric = Metric(name)
        # TODO: use enum in AudioMetrics as well, for now it uses strings
        if metric == Metric.fad:
            key = "fad"
            return "fad", key
        if metric == Metric.mmd:
            key = KEY_METRIC_KID_MEAN
            return "kd", key
        raise NotImplementedError(f"Unsupported metric {metric}")

    def _get_embedder(self, name, layer=None, device=None):
        if device is None:
            device = get_device()
        emb = Embedder(name)
        if emb == Embedder.vggish:
            from audio_metrics.vggish import VGGish

            return VGGish(device)
        if "openl3" in Embedder.__members__ and emb == Embedder.openl3:
            from audio_metrics.openl3 import OpenL3

            return OpenL3(device)
        if emb in (Embedder.clap, Embedder.clap_music, Embedder.clap_music_speech):
            from audio_metrics.clap import (
                CLAP,
                CLAP_MUSIC_SPEECH_CHECKPOINT_URL,
                CLAP_MUSIC_CHECKPOINT_URL,
            )

            clap_url = CLAP_MUSIC_SPEECH_CHECKPOINT_URL
            if emb == Embedder.clap_music:
                clap_url = CLAP_MUSIC_CHECKPOINT_URL
            return CLAP(
                device,
                intermediate_layers=layer is not None,
                checkpoint_url=clap_url,
                layer=layer,
            )
        raise NotImplementedError(f"Unsupported embedder {emb}")

    def _make_emb(self, audio_pairs, progress=None):
        # early fusion
        items = mix_pairs(audio_pairs, self.mix_func)
        items = maybe_slice_audio(items, self.win_dur)
        return self.pipeline.embed_join(items, **self.embed_kwargs, progress=progress)

    def set_background(self, audio_pairs):
        # NOTE: we load all audio into memory
        pairs = list(audio_pairs)
        n_items = len(pairs)
        if self.win_dur is None:
            self._check_minimum_data_size(n_items)
            total = 2 * n_items
        else:
            total = None
        prog = tqdm(total=total, desc="computing background embeddings")
        self.metrics_1.set_background_data(self._make_emb(pairs, prog))
        emb_2 = self._make_emb(misalign(pairs), prog)
        self.metrics_2.set_background_data(emb_2)
        self.metrics_1.set_pca_projection(self.n_pca, self.pca_whiten)
        self.metrics_2.set_pca_projection(self.n_pca, self.pca_whiten)
        # compare non-matching to matching and save distance value for APA computation
        result = self.metrics_1.compare_to_background(emb_2)
        self.m_x_xp = result[self._key]

    def compare_to_background(self, audio_pairs):
        pairs = list(audio_pairs)
        prog = tqdm(total=None, desc="computing candidate embeddings")
        embeddings = self._make_emb(pairs, prog)
        return self.compare_embeddings_to_background(embeddings)

    def compare_embeddings_to_background(self, embeddings):
        if not isinstance(embeddings, dict):
            embeddings = {(self._emb_key, self.layer): embeddings}

        d1 = self.metrics_1.compare_to_background(embeddings)
        d2 = self.metrics_2.compare_to_background(embeddings)
        key = self._key
        m_x_y = max(0, d1[key])
        m_xp_y = max(0, d2[key])
        # score = (m_xp_y - m_x_y) / (m_xp_y + m_x_y)
        # score = max(0, (m_xp_y - m_x_y) / self.m_x_xp)
        score = apa(m_x_y, m_xp_y, self.m_x_xp)
        if abs(m_x_y - m_xp_y) >= self.m_x_xp:
            warnings.warn("Triangle inequality not satisfied")
        return {
            "audio_prompt_adherence": score,
            "n_real": d1["n_real"],
            "n_fake": d1["n_fake"],
        }

    def _check_minimum_data_size(self, n_items):
        msg = f"The number of PCA components ({self.n_pca}) cannot be larger than the number of embedding vectors ({n_items})"
        assert self.n_pca <= n_items, msg


def apa(d_y_x, d_y_xp, d_x_xp):
    d_y_x = max(0, d_y_x)
    d_y_xp = max(0, d_y_xp)
    d_x_xp = max(0, d_x_xp)
    numerator = d_y_xp - d_y_x
    denominator = d_x_xp
    if abs(numerator) > denominator:
        # msg1 = f" y_x={d_y_x:.3f} y_xp={d_y_xp:.3f} x_xp={d_x_xp:.3f}"
        # msg2 = f" a+b={abs(d_y_x - d_y_xp)} c={d_x_xp}"
        # warnings.warn("Triangle inequality not satisfied:" + msg1 + msg2)
        denominator = abs(numerator)
    if denominator <= 0:
        return 0.0
    return 1 / 2 + numerator / (2 * denominator)
