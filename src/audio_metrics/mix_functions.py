import warnings
from functools import partial
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view, as_strided
import scipy
import pyloudnorm as pyln
import numpy_audio_limiter
import opt_einsum


class Meter(pyln.Meter):
    def __init__(self, sr):
        super().__init__(sr)
        self.G = np.array([1.0, 1.0, 1.0, 1.41, 1.41])

    def integrated_loudness_fast(self, data):
        """Measure the integrated gated loudness of a signal.

        Uses the weighting filters and block size defined by the meter the
        integrated loudness is measured based upon the gating algorithm defined
        in the ITU-R BS.1770-4 specification.

        Input data must have shape (samples, ch) or (samples,) for mono audio.
        Supports up to 5 channels and follows the channel ordering: [Left,
        Right, Center, Left surround, Right surround]

        This is an equivalent implementation to the original integrated_loudness
        method, optimized for speed.  It achieves about a 1.20x speedup over the
        original for stereo tracks ans around 1.45x for mono tracks.

        Params
        ------

        data : ndarray Input multichannel audio data.

        Returns
        -------

        LUFS : float Integrated gated loudness of the input measured in dB LUFS.
        """
        input_data = data
        pyln.util.valid_audio(input_data, self.rate, self.block_size)

        if data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        numSamples, numChannels = input_data.shape
        einsum = opt_einsum.contract  # or np.einsum
        # Apply frequency weighting filters - account for the acoustic response of the head and auditory system
        for filter_class, filter_stage in self._filters.items():
            input_data = filter_stage.passband_gain * scipy.signal.lfilter(
                filter_stage.b, filter_stage.a, input_data, axis=0
            )
        G = self.G[:numChannels]  # channel gains
        T_g = self.block_size  # 400 ms gating block standard
        Gamma_a = -70.0  # -70 LKFS = absolute loudness threshold
        overlap = 0.75  # overlap of 75% of the block duration
        step = 1.0 - overlap  # step size by percentage
        T = numSamples / self.rate  # length of the input in seconds
        numBlocks = int(
            np.round(((T - T_g) / (T_g * step))) + 1
        )  # total number of gated blocks (see end of eq. 3)
        z = np.empty(shape=(numChannels, numBlocks))
        # Create sliding windows of size B along the first axis
        block_size = int(T_g * self.rate)
        stride = int(T_g * step * self.rate)
        item_size = input_data.strides[-1]
        new_strides = (
            item_size * numChannels * stride,
            item_size * numChannels,
            item_size,
        )
        new_size = (numBlocks, block_size, numChannels)
        input_data **= 2
        windows = as_strided(input_data, new_size, new_strides, writeable=False)
        z = einsum("abc -> ca", windows)
        z /= T_g * self.rate

        # Sum over each block (axis 1 is the block dimension)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # loudness for each jth block (see eq. 4)
            l = -0.691 + 10 * np.log10(einsum("c,cb->b", G, z))

        # find gating block indices above absolute threshold
        J_g = l >= Gamma_a

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 5
            z_act = z[:, J_g]
            z_avg_gated = einsum("cb->c", z_act)
            z_avg_gated /= z_act.shape[-1]
        # calculate the relative threshold value (see eq. 6)
        Gamma_r = -0.691 + 10 * np.log10(einsum("c,c->", G, z_avg_gated)) - 10
        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        J_g = l > Gamma_r
        J_g &= l > Gamma_a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
            z_act = z[:, J_g]
            z_avg_gated = einsum("cb->c", z_act)
            z_avg_gated /= z_act.shape[-1]
            z_avg_gated = np.nan_to_num(z_avg_gated)
        # calculate final loudness gated loudness (see eq. 7)
        with np.errstate(divide="ignore"):
            LUFS = -0.691 + 10.0 * np.log10(einsum("c,c->", G, z_avg_gated))
        return LUFS


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
        s2 = numpy_audio_limiter.limit(
            signal=s2.astype(np.float32).reshape((1, -1)),
            attack_coeff=0.99,
            release_coeff=0.99,
            delay=527,
            threshold=0.5,
        )[0]
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
    # l_mix_check = meter.integrated_loudness(mix)  # measure loudness
    vmax = np.max(np.abs(mix))
    if vmax > 1.0:
        mix = numpy_audio_limiter.limit(
            signal=mix.astype(np.float32).reshape((1, -1)),
            attack_coeff=0.99,
            release_coeff=0.99,
            delay=527,
            threshold=0.5,
        )[0]

    if np.any(np.isnan(mix)):
        print(f"NaN with vmax={vmax}")
        # print(f"l_mix={l_mix} l_mix_check={l_mix_check} l_out={out_db}")
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
DEFAULT_MIX_FUNCTION = "L0"
