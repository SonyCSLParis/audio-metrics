# Audio Metrics

This repository contains a python package to compute distribution-based quality
measures for audio data using embeddings, with a focus on music.

* Fréchet Distance (see https://arxiv.org/abs/1812.08466 )

* Kernel Distance/Maximum Mean Discrepancy (see https://arxiv.org/abs/1812.08466 )

* Density and Coverage (see https://arxiv.org/abs/2002.09797 )

* Audio Prompt Adherence (see https://arxiv.org/abs/2404.00775 )

The measures have in common that they compare a **set** of candidate audio
tracks against a **set** of reference tracks, rather than evaluating individual
tracks, and they all work on **embedding** representations of audio, obtained
from models pretrained on tasks like audio classification.

The first two measures are typically used to measure audio quality (i.e. the
naturalness of the sound, and the absence of acoustic artifacts). Density and
Coverage explicitly measure how well the candidate set coincides with the
reference set by comparing the embedding manifolds.

The Audio Prompt Adherence measures operates on sets whose elements are
**pairs** of audio tracks, typically a **mix** and an **accompaniment**, and
quantifies how well the accompaniment fits to the mix.
 
The measures can be combined with embeddings from any of the following models:

* VGGish - https://arxiv.org/abs/1609.09430 Trained on audio event
  classification. 128-dimensional embeddings from the last feature layer before
  the classification layer.

* OpenL3 - https://github.com/marl/openl3 Trained on music

* Laion CLAP - https://github.com/LAION-AI/CLAP Trained on Music; Embeddings
  from the last three layers (512, 512, and 128-dimensional)


## Installation

Download this repo your computer, and in the root directory of the repo, run:

```
pip install .
```


## Usage

The following examples demonstrate the use of the package. Both examples are
also included under the `./examples` directory.

### Computing FAD/Kernel Distance 

The following code computes FAD and Kernel Distance for a (unrealistically
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
real_items = async_audio_loader(audio_dir / "real")
fake_items = async_audio_loader(audio_dir / "fake")

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
  "coverage_clap_audio_projection.0": 0.675,
  "coverage_clap_audio_projection.2": 0.44,
  "coverage_clap_output": 0.49,
  "density_clap_audio_projection.0": 1.2675,
  "density_clap_audio_projection.2": 0.605,
  "density_clap_output": 0.7525,
  "fad_clap_audio_projection.0": 25.192331663033556,
  "fad_clap_audio_projection.2": 33.02863890811378,
  "fad_clap_output": 32.31293025087572,
  "kernel_distance_mean_clap_audio_projection.0": 0.5756649634334309,
  "kernel_distance_mean_clap_audio_projection.2": 0.7890714981174441,
  "kernel_distance_mean_clap_output": 0.7467294878649637,
  "kernel_distance_std_clap_audio_projection.0": 0.03337777531962964,
  "kernel_distance_std_clap_audio_projection.2": 0.04380359012320949,
  "kernel_distance_std_clap_output": 0.04377045813837937,
  "n_fake": 200,
  "n_real": 200
}
```

### Audio Prompt Adherence

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
real_items = async_audio_loader(audio_dir / "real", mono=False)
fake_items = async_audio_loader(audio_dir / "fake", mono=False)

# iterate over windows
real_items = multi_audio_slicer(real_items, win_dur)
fake_items = multi_audio_slicer(fake_items, win_dur)

metrics = AudioPromptAdherence(
    dev, win_dur, n_pca=100, embedder="clap", metric="mmd"
)
metrics.set_background(real_items)
result = metrics.compare_to_background(fake_items)
print(json.dumps(result, indent=2))
```

which will print something like this:

```json
{
  "audio_prompt_adherence": 0.15253860533909092,
  "n_real": 200,
  "n_fake": 200
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

