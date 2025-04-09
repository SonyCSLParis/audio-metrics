# Audio Metrics

This repository contains a python package to compute distribution-based quality
metrics for audio data using embeddings, with a focus on music.

It features the following metrics:

* Fréchet Distance (see https://arxiv.org/abs/1812.08466 )

* Kernel Distance/Maximum Mean Discrepancy (see https://arxiv.org/abs/1812.08466 )

* Density and Coverage (see https://arxiv.org/abs/2002.09797 )

* Accompaniment Prompt Adherence (see https://arxiv.org/abs/2503.06346 )

The measures have in common that they compare a **set** of candidate audio
tracks against a **set** of reference tracks, rather than evaluating individual
tracks, and they all work on **embedding** representations of audio, obtained
from models pretrained on tasks like audio classification.

The first two measures are typically used to measure audio quality (i.e. the
naturalness of the sound, and the absence of acoustic artifacts). Density and
Coverage explicitly measure how well the candidate set coincides with the
reference set by comparing the embedding manifolds.

The Accompaniment Prompt Adherence measures operates on sets whose elements are
**pairs** of audio tracks, typically a **mix** and an **accompaniment**, and
quantifies how well the accompaniment fits to the mix.
 
The measures can be combined with embeddings from any of the following models:

* VGGish - https://arxiv.org/abs/1609.09430 Trained on audio event
  classification. 128-dimensional embeddings from the last feature layer before
  the classification layer.

* Laion CLAP - https://github.com/LAION-AI/CLAP using either the checkpoint
  trained on music and speech, or the checkpoint trained on music only;
  Embeddings from the last three layers (512, 512, and 128-dimensional)



## Installation

1. Download this repository (`git clone https://github.com/SonyCSLParis/audio-metrics.git`)

2. Change to the repository root directory (`cd audio-metrics`)

3. You can either install into an existing python environment (activate it and
   go to step 4), or create and activate a new environment like this:

   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install the package and its dependencies:
   ```
	pip install .
	```


## Usage

The following examples demonstrate the use of the package. Both examples are
also included under the `./examples` directory.


```python
import torch
from audio_metrics import AudioMetrics

sr = 48000
n_seconds = 10

# reference pairs of "context" and "accompaniment"
reference = (torch.randn(n_seconds*sr, 2) for _ in range(1000))

# candidate pairs of "context" and "accompaniment"
candidate = (torch.randn(n_seconds*sr, 2) for _ in range(1000))


metrics = AudioMetrics(metrics=['apa', 'fad'])
metrics.add_reference(reference)

result = metrics(candidate)
```


### Accompaniment Prompt Adherence

The Accompaniment Prompt Adherence metric takes pairs of audio samples (mix, and
accompaniment, respectively), and computes how well mixes and accompaniments fit
together, given a background set of (mix,accompaniment) pairs. The following
example shows how to compute the Accompaniment Prompt Adherence metric:

## Notes


## Citation

For more information on the audio prompt adherence metric, and to cite this work, please use:


>  M. Grachten and J. Nistal. (2025). Accompaniment Prompt Adherence: A Measure for Evaluating Music Accompaniment Systems. Proceedings of the International Conference on Acoustics, Speech, and Signal Processing. IEEE. Hyderabad, India.

