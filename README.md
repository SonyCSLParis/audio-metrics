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
import json
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    EmbedderPipeline,
    AudioMetrics,
    CLAP,
    VGGish,
)


audio_dir = "audio_samples"
win_dur = 1.0
hop_dur = 0.1
n_pca = 16
dev = torch.device("cuda")

# load audio samples from files in `audio_dir`
items = list(async_audio_loader(audio_dir))

# split into "real" and "fake"
real_items = items[: len(items) // 2]
fake_items = items[len(items) // 2 :]

# iterate over windows
real_items = multi_audio_slicer(real_items, win_dur, hop_dur)
fake_items = multi_audio_slicer(fake_items, win_dur, hop_dur)

# create the embeddings
embedder = EmbedderPipeline({"clap": CLAP(dev)})
real_embeddings = embedder.embed_join(real_items)
fake_embeddings = embedder.embed_join(fake_items)

# set the background data for the metrics
# use PCA projection of embeddings without whitening
metrics = AudioMetrics()
metrics.set_background_data(real_embeddings)
metrics.set_pca_projection(n_pca, whiten=False)

# compare "real" to "fake" data
result = metrics.compare_to_background(fake_embeddings)
print(json.dumps(result, indent=2))
```

Which will print:

```json
{
  "fad_clap_audio_projection.0": 19.225023903605987,
  "fad_clap_audio_projection.2": 7.9856999641986075,
  "fad_clap_output": 0.11684167390293426,
  "kernel_distance_mean_clap_audio_projection.0": 4.33607215213849,
  "kernel_distance_mean_clap_audio_projection.2": 1.1579099677492803,
  "kernel_distance_mean_clap_output": 0.011475965446770518,
  "kernel_distance_std_clap_audio_projection.0": 0.4230402206366779,
  "kernel_distance_std_clap_audio_projection.2": 0.10131857193842961,
  "kernel_distance_std_clap_output": 0.0011241097288160584,
  "n_fake": 455,
  "n_real": 455
}
```						


## Usage: Audio Prompt Adherence

The Audio Prompt Adherence metric takes pairs of audio samples (mix, and
accompaniment, respectively), and computes how well mixes and accompaniments fit
together, given a background set of (mix,accompaniment) pairs.

```python
from audio_metrics import AudioPromptAdherence
win_dur = 5.0
dev = torch.device("cuda")
metric = AudioPromptAdherence(dev, win_dur)
metric.set_background(real_data)

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

