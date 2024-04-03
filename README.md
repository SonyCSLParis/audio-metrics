# Audio Metrics

This repository contains a python package to compute distribution-based metrics
for audio data using embeddings. 

* Fréchet Distance (see https://arxiv.org/abs/1812.08466)
* Kernel Distance/Maximum Mean Discrepancy (see https://arxiv.org/abs/1812.08466)
* Density and Coverage (see https://arxiv.org/abs/2002.09797)
* Audio Prompt Adherence https://arxiv.org/abs/2404.00775

The metrics use the publicly available pretrained VGGish model (trained on audio
event classification) to compute embeddings (see
https://arxiv.org/abs/1609.09430). In particular, it uses the 128-dimensional
embeddings from the last feature layer before the classification layer.


## Installation

Download this repo your computer, and in the root directory of the repo, run:

```
pip install .
```


## Usage: Computing FAD/Kernel Distance 

The following example computes FAD and Kernel Distance for a (unrealistically
small) set of audio samples:

```python
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
```

Which will print the metrics (exact values may slightly vary):

```json
{
  "coverage_clap_audio_projection.0": 1.0,
  "coverage_clap_audio_projection.2": 1.0,
  "coverage_clap_output": 1.0,
  "density_clap_audio_projection.0": 1.1675,
  "density_clap_audio_projection.2": 1.13375,
  "density_clap_output": 1.195,
  "fad_clap_audio_projection.0": 4.68105554318754e-11,
  "fad_clap_audio_projection.2": 4.666844688472338e-11,
  "fad_clap_output": 4.277467269275803e-11,
  "kernel_distance_mean_clap_audio_projection.0": 0.03798904296875001,
  "kernel_distance_mean_clap_audio_projection.2": 0.0375376103515625,
  "kernel_distance_mean_clap_output": 0.038380160156250044,
  "kernel_distance_std_clap_audio_projection.0": 0.0037973875005004065,
  "kernel_distance_std_clap_audio_projection.2": 0.0032739005375918514,
  "kernel_distance_std_clap_output": 0.003321617550645439,
  "n_fake": 400,
  "n_real": 400
}
```

## Usage: Audio Prompt Adherence

The Audio Prompt Adherence metric takes pairs of audio samples (mix, and
accompaniment, respectively), and computes how well mixes and accompaniments fit
together, given a background set of (mix,accompaniment) pairs. The following
example shows how to compute the Audio Prompt Adherence metric:

```python
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
```

which will print something like this:

```json
{
  "audio_prompt_adherence": 0.7607011406305113,
  "stem_distance": 0.03952705566406251,
  "n_real": 400,
  "n_fake": 400
}

```

## Citation

For more information on the audio prompt adherence metric, and to cite this work use:

```
@misc{grachten2024measuring,
  title={Measuring audio prompt adherence with distribution-based embedding distances}, 
  author={Maarten Grachten},
  year={2024},
  eprint={2404.00775},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```

