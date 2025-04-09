from .vggish import VGGish
from .clap import (
    LaionCLAP,
    LAION_CLAP_MUSIC_SPEECH_CHECKPOINT_URL,
    LAION_CLAP_MUSIC_CHECKPOINT_URL,
    LAION_CLAP_LAYERS,
)

EMBEDDERS = {
    "laion_clap_music": (
        LaionCLAP,
        {
            "ckpt": LAION_CLAP_MUSIC_CHECKPOINT_URL,
        },
    ),
    "laion_clap_music_l-2": (
        LaionCLAP,
        {
            "ckpt": LAION_CLAP_MUSIC_CHECKPOINT_URL,
            "layer": LAION_CLAP_LAYERS[0],
        },
    ),
    "laion_clap_music_l-1": (
        LaionCLAP,
        {
            "ckpt": LAION_CLAP_MUSIC_CHECKPOINT_URL,
            "layer": LAION_CLAP_LAYERS[1],
        },
    ),
    "laion_clap_music_speech": (
        LaionCLAP,
        {
            "ckpt": LAION_CLAP_MUSIC_SPEECH_CHECKPOINT_URL,
        },
    ),
    "laion_clap_music_speech_l-2": (
        LaionCLAP,
        {
            "ckpt": LAION_CLAP_MUSIC_SPEECH_CHECKPOINT_URL,
            "layer": LAION_CLAP_LAYERS[0],
        },
    ),
    "laion_clap_music_speech_l-1": (
        LaionCLAP,
        {
            "ckpt": LAION_CLAP_MUSIC_SPEECH_CHECKPOINT_URL,
            "layer": LAION_CLAP_LAYERS[1],
        },
    ),
    "vggish": (
        VGGish,
        {},
    ),
}

DEFAULT_EMBEDDER = "laion_clap_music"
