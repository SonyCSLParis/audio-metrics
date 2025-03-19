import enum
# from .clap import CLAP
# from .vggish import VGGish

# try:
#     import torchopenl3
# except ModuleNotFoundError:
#     have_openl3 = False
# else:
#     have_openl3 = True
#     from .openl3 import OpenL3


Embedder = enum.Enum(
    "Embedder",
    {
        k: k
        for k in [
            "vggish",
            "clap",  # legacy -> clap_music_speech
            "clap_music",
            "clap_music_speech",
        ]
        # + (["openl3"] if have_openl3 else [])
    },
)

from .dataset import async_audio_loader, multi_audio_slicer
from .audio_metrics import AudioMetrics, MetricInputData, save_embeddings
from .apa import AccompanimentPromptAdherence
from .apa import AccompanimentPromptAdherence as APA
from .embed_pipeline import EmbedderPipeline
