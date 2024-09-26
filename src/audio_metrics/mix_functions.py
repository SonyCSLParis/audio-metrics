import warnings
from functools import partial
import numpy as np

import pyloudnorm as pyln
from cylimiter import Limiter


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


MIX_FUNCTIONS = dict(
    PP=mix_tracks_peak_preserve,
    P0=partial(mix_tracks_peak_normalize, stem_db_red=-0, out_db=-3),
    P1=partial(mix_tracks_peak_normalize, stem_db_red=-3, out_db=-3),
    P2=partial(mix_tracks_peak_normalize, stem_db_red=-6, out_db=-3),
    L0=partial(mix_tracks_loudness, stem_db_red=0, out_db=-20),
    L1=partial(mix_tracks_loudness, stem_db_red=-3, out_db=-20),
    L2=partial(mix_tracks_loudness, stem_db_red=-6, out_db=-20),
)
