from audio_metrics.data import AudioMetricsData
from audio_metrics.metrics.fad import frechet_distance


def apa_compute_d_x_xp(reference: AudioMetricsData, anti_reference: AudioMetricsData):
    return frechet_distance(reference, anti_reference)


def apa(
    candidate: AudioMetricsData,
    reference: AudioMetricsData,
    anti_reference: AudioMetricsData,
    d_x_xp: float | None = None,
):
    d_y_x = frechet_distance(candidate, reference)
    d_y_xp = frechet_distance(candidate, anti_reference)
    if d_x_xp is None:
        d_x_xp = frechet_distance(reference, anti_reference)
    return _apa(d_y_x, d_y_xp, d_x_xp)


def _apa(d_y_x, d_y_xp, d_x_xp):
    d_y_x = max(0, d_y_x)
    d_y_xp = max(0, d_y_xp)
    d_x_xp = max(0, d_x_xp)
    numerator = d_y_xp - d_y_x
    denominator = d_x_xp
    if abs(numerator) > denominator:
        denominator = abs(numerator)
    if denominator <= 0:
        return 0.0
    return 1 / 2 + numerator / (2 * denominator)
