from pathlib import Path
import json
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    AudioPromptAdherence,
)
from audio_metrics.example_utils import generate_audio_samples


audio_dir1 = Path("audio_samples1")
audio_dir2 = Path("audio_samples2")
win_dur = 5.0
dev = torch.device("cuda")

print("generating 'real' and 'fake' audio samples")
generate_audio_samples(audio_dir1)
generate_audio_samples(audio_dir2)

# load audio samples from files in `audio_dir`
real1_items = async_audio_loader(audio_dir1 / "real", mono=False)
fake1_items = async_audio_loader(audio_dir1 / "fake", mono=False)
real2_items = async_audio_loader(audio_dir2 / "real", mono=False)

# iterate over windows
real1_items = multi_audio_slicer(real1_items, win_dur)
fake1_items = multi_audio_slicer(fake1_items, win_dur)
real2_items = multi_audio_slicer(real2_items, win_dur)

metrics = AudioPromptAdherence(
    dev, win_dur, n_pca=100, embedder="clap", metric="fad", layer="audio_projection.2"
)
metrics.set_background(real1_items)
print("non-matching:")
result = metrics.compare_to_background(fake1_items)
print(json.dumps(result, indent=2))
print("matching:")
result = metrics.compare_to_background(real2_items)
print(json.dumps(result, indent=2))
