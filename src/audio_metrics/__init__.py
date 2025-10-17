from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("audio-metrics")
except PackageNotFoundError:
    __version__ = "unknown"

from .audio_metrics import AudioMetrics
