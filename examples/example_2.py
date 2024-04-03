from pathlib import Path
import json
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    AudioPromptAdherence,
)
from audio_metrics.example_utils import generate_audio_samples


audio_dir = Path("audio_samples")
win_dur = 5.0
dev = torch.device("cuda")

print("generating 'real' and 'fake' audio samples")
generate_audio_samples(audio_dir)

# load audio samples from files in `audio_dir`
real_items = async_audio_loader(audio_dir, audio_dir / "real", mono=False)
fake_items = async_audio_loader(audio_dir, audio_dir / "fake", mono=False)

# iterate over windows
real_items = multi_audio_slicer(real_items, win_dur)
fake_items = multi_audio_slicer(fake_items, win_dur)

metrics = AudioPromptAdherence(
    dev, win_dur, n_pca=100, embedder="clap", metric="mmd2"
)
metrics.set_background(real_items)
result = metrics.compare_to_background(fake_items)
print(json.dumps(result, indent=2))
