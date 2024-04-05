import warnings
import enum
import warnings
import numpy as np
import torch
from tqdm import tqdm
import pyloudnorm as pyln

from audio_metrics import AudioMetrics
from audio_metrics.embed_pipeline import EmbedderPipeline
from audio_metrics.kid import KEY_METRIC_KID_MEAN


def mix_tracks_loudness(audio, sr, stem_db_red=-4.0, out_db=-20.0):
    """Mix channels with fixed loudness relationship

    audio: samples x channels

    """

    assert len(audio.shape) == 2
    if audio.shape[1] == 1:
        return audio[:, 0]
    vmax = np.abs(audio).max(0)
    eps = 1e-5
    silent = vmax < eps
    if np.all(silent):
        warnings.warn("Both channels silent")
        return audio[:, 0]

    meter = pyln.Meter(sr)  # create BS.1770 meter
    if np.any(silent):
        warnings.warn("One channel silent")
        mix = audio[:, ~silent][:, 0]
    else:
        with warnings.catch_warnings():
            s0, s1 = audio.T
            warnings.simplefilter("ignore")
            l0 = meter.integrated_loudness(s0)
            l1 = meter.integrated_loudness(s1)
            # set the loudness of s1 w.r.t. that of s0
            s1 = pyln.normalize.loudness(s1, l1, l0 + stem_db_red)
            mix = s0 + s1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l_mix = meter.integrated_loudness(mix)  # measure loudness
        mix = pyln.normalize.loudness(mix, l_mix, out_db)
    l_mix = meter.integrated_loudness(mix)  # measure loudness
    vmax = np.max(np.abs(mix))
    if vmax > 1.0:
        warnings.warn(f"Reducing gain to prevent clipping ({vmax:.2f})")
        mix /= vmax
    return mix


def mix_pairs(pairs):
    return (
        (mix_tracks_loudness(np.column_stack((mix, stem)), sr), sr)
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


class Embedder(enum.Enum):
    VGGISH: enum.auto()
    OPENL3: enum.auto()
    CLAP: enum.auto()


class Distance(enum.Enum):
    FAD: enum.auto()
    MMD: enum.auto()


Embedder = enum.Enum("Embedder", {k: k for k in ("vggish", "openl3", "clap")})
Metric = enum.Enum("Metric", {k: k for k in ("fad", "mmd")})


class AudioPromptAdherence:
    def __init__(
        self,
        device: str | torch.device | None = None,
        win_dur: float | None = None,
        n_pca: int | None = None,
        pca_whiten: bool = True,
        embedder: str = Embedder.vggish,
        metric: str = Metric.fad,
    ):
        self.n_pca = n_pca
        self.pca_whiten = pca_whiten
        embedders = {"emb": self._get_embedder(embedder, device)}
        self.embed_kwargs = {
            "combine_mode": "average",
            "batch_size": 10,
            "max_workers": 10,
        }
        self.pipeline = EmbedderPipeline(embedders)
        _metric, self.metric_key = self._get_metric(metric)
        self.metrics_1 = AudioMetrics(metrics=[_metric])
        self.metrics_2 = AudioMetrics(metrics=[_metric])
        self.win_dur = win_dur
        # hacky: build the key to get the metric value from the AuioMetrics results
        self._key = "_".join([self.metric_key, "emb", embedders["emb"].names[0]])

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

    def _make_emb(self, audio_pairs, progress=None):
        # early fusion
        items = mix_pairs(audio_pairs)
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
        emb = self._make_emb(pairs, prog)
        d1 = self.metrics_1.compare_to_background(emb)
        d2 = self.metrics_2.compare_to_background(emb)
        key = self._key
        m_x_y = max(0, d1[key])
        m_xp_y = max(0, d2[key])
        # score = (m_xp_y - m_x_y) / (m_xp_y + m_x_y)
        score = 1 / 2 + (m_xp_y - m_x_y) / (2 * self.m_x_xp)
        # print(f"a={d1[key]:.3f} b={d2[key]:.3f} c={self.c_result[key]:.3f}")
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
