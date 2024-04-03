from pathlib import Path
import json
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    EmbedderPipeline,
    AudioMetrics,
    CLAP,
)
from audio_metrics.example_utils import generate_audio_samples


audio_dir = Path("audio_samples")
win_dur = 5.0
n_pca = 64
dev = torch.device("cuda")

print("generating 'real' and 'fake' audio samples")
generate_audio_samples(audio_dir)

# load audio samples from files in `audio_dir`
real_items = async_audio_loader(audio_dir, audio_dir / "real")
fake_items = async_audio_loader(audio_dir, audio_dir / "fake")

# iterate over windows
real_items = multi_audio_slicer(real_items, win_dur)
fake_items = multi_audio_slicer(fake_items, win_dur)

print("creating embedder")
embedder = EmbedderPipeline({"clap": CLAP(dev)})
print("computing 'real' embeddings")
real_embeddings = embedder.embed_join(real_items)
print("computing 'fake' embeddings")
fake_embeddings = embedder.embed_join(fake_items)

# set the background data for the metrics
# use PCA projection of embeddings without whitening
metrics = AudioMetrics()
metrics.set_background_data(real_embeddings)
metrics.set_pca_projection(n_pca, whiten=True)

print("comparing 'real' to 'fake' data")
result = metrics.compare_to_background(fake_embeddings)
print(json.dumps(result, indent=2))
