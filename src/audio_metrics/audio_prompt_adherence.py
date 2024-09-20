import warnings
from functools import partial
import pickle
import enum
import numpy as np
import torch
from tqdm import tqdm
import pyloudnorm as pyln
from cylimiter import Limiter

from audio_metrics import AudioMetrics
from audio_metrics.embed_pipeline import EmbedderPipeline
from audio_metrics.kid import KEY_METRIC_KID_MEAN


def mix_tracks_peak_preserve(audio, sr):
    """Mix channels as as, and normalize to peak amplitude of the original waveforms

    audio: samples x channels

    """
    assert len(audio.shape) == 2
    # n_ch = audio.shape[1]
    if audio.shape[1] == 1:
        return audio[:, 0]
    vmax_orig = np.abs(audio).max()
    eps = 1e-5
    if vmax_orig <= eps:
        return audio[:, 0]
    mix = np.mean(audio, 1)
    vmax_new = np.abs(mix).max()
    gain = vmax_orig / vmax_new
    mix *= gain
    return mix


def mix_tracks_peak_normalize(audio, sr, stem_db_red=0.0, out_db=0.0):
    """Mix by-peak normalizing channels, and then peak normalizing the mix.

    audio: samples x channels

    """
    # gain = np.power(10.0, target/20.0) / current_peak
    out_gain = np.power(10.0, out_db / 20.0)
    stem_gain = np.power(10.0, stem_db_red / 20.0)
    assert len(audio.shape) == 2
    # n_ch = audio.shape[1]
    if audio.shape[1] == 1:
        mix = audio[:, 0]
    else:
        peaks = np.abs(audio).max(0, keepdims=True)
        peaks[0, 1] *= stem_gain
        mix = (audio / peaks).sum(1)

    mix *= out_gain / np.abs(mix).max()
    return mix


def mix_preserve_loudness(audio, sr):
    # audio: nsamples x 2
    meter = pyln.Meter(sr)  # create BS.1770 meter
    s0, s1 = audio.T
    s2 = s0 + s1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l0 = meter.integrated_loudness(s0)
        l1 = meter.integrated_loudness(s1)
        l2 = meter.integrated_loudness(s2)

        l_trg = max(l0, l1)
        if not np.isinf(l_trg) and not np.isinf(l2):
            s2 = pyln.normalize.loudness(s2, l2, l_trg)

    vmax = np.max(np.abs(s2))
    if vmax > 1.0:
        warnings.warn(f"Reducing gain (peak amp: {vmax:.2f})")
        limiter = Limiter()
        s2 = limiter.apply(s2)

    return s2


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
            l1_trg = l0 + stem_db_red
            if not np.isinf(l1) and not np.isinf(l1_trg):
                s1 = pyln.normalize.loudness(s1, l1, l1_trg)
            mix = s0 + s1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l_mix = meter.integrated_loudness(mix)  # measure loudness
        if not np.isinf(l_mix) and not np.isinf(out_db):
            mix = pyln.normalize.loudness(mix, l_mix, out_db)
    l_mix_check = meter.integrated_loudness(mix)  # measure loudness
    vmax = np.max(np.abs(mix))
    if vmax > 1.0:
        # warnings.warn(f"Reducing gain to prevent clipping ({vmax:.2f})")
        limiter = Limiter()
        mix = limiter.apply(mix)

    if np.any(np.isnan(mix)):
        print(f"NaN with vmax={vmax}")
        print(f"l_mix={l_mix} l_mix_check={l_mix_check} l_out={out_db}")
        print(f"l0={l0} l1={l1}")
    return mix


MIX_FUNCS = dict(
    PP=mix_tracks_peak_preserve,
    P0=partial(mix_tracks_peak_normalize, stem_db_red=-0, out_db=-3),
    P1=partial(mix_tracks_peak_normalize, stem_db_red=-3, out_db=-3),
    P2=partial(mix_tracks_peak_normalize, stem_db_red=-6, out_db=-3),
    L0=partial(mix_tracks_loudness, stem_db_red=0, out_db=-20),
    L1=partial(mix_tracks_loudness, stem_db_red=-3, out_db=-20),
    L2=partial(mix_tracks_loudness, stem_db_red=-6, out_db=-20),
)


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


Embedder = enum.Enum(
    "Embedder",
    {
        k: k
        for k in (
            "vggish",
            "openl3",
            "clap",  # legacy -> clap_music_speech
            "clap_music",
            "clap_music_speech",
        )
    },
)
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
        self.mix_func = MIX_FUNCS.get(mix_func, "L2")
        embedders = {"emb": self._get_embedder(embedder, layer=layer, device=device)}
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
        if emb == Embedder.openl3:
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
                device, intermediate_layers=layer is not None, checkpoint_url=clap_url
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
        emb = self._make_emb(pairs, prog)
        d1 = self.metrics_1.compare_to_background(emb)
        d2 = self.metrics_2.compare_to_background(emb)
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
