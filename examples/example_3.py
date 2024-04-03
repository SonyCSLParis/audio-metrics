import numpy as np
import torch
from tqdm import tqdm
import json
from audio_metrics import AudioMetrics
from audio_metrics.vggish import VGGish
from audio_metrics.clap import CLAP
from audio_metrics.openl3 import OpenL3

from audio_metrics.embed_pipeline import EmbedderPipeline


def random_audio_generator(n_items, sr, audio_len):
    for i in range(n_items):
        audio = np.random.random(audio_len).astype(np.float32) - 0.5
        yield (audio, sr)


def main():
    n_pca = 10
    dev = torch.device("cuda")
    embedders = {
        "vggish": VGGish(dev),
        "clap": CLAP(dev),
        "openl3": OpenL3(dev),
    }
    embed_kwargs = {
        "combine_mode": "average",  # 'stack'
        # "ordered": True,
        "batch_size": 10,
        "max_workers": 10,
    }
    pipeline = EmbedderPipeline(embedders)
    n_items = 100
    sr = 48000
    audio_len = 5 * sr

    real_data_iter = random_audio_generator(n_items, sr, audio_len)
    fake_data_iter = random_audio_generator(n_items, sr, audio_len)
    progress_real = tqdm(total=n_items, desc="real_embeddings")
    progress_fake = tqdm(total=n_items, desc="fake_embeddings")

    real_embeddings = pipeline.embed_join(
        real_data_iter, **embed_kwargs, progress=progress_real
    )
    fake_embeddings = pipeline.embed_join(
        fake_data_iter, **embed_kwargs, progress=progress_fake
    )

    metrics = AudioMetrics(metrics=["fad", "kd"])
    metrics.set_background_data(real_embeddings)
    metrics.set_pca_projection(n_pca)
    result = metrics.compare_to_background(fake_embeddings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
