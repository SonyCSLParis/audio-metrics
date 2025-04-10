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

The following examples demonstrate the use of the package. For more examples see
`./examples` directory.


```python
import numpy as np
from audio_metrics import AudioMetrics

sr = 48000
n_seconds = 5

n_windows = 100
window_len = sr * n_seconds

reference = np.random.random((n_windows, window_len))
candidate = np.random.random((n_windows, window_len))

metrics = AudioMetrics(metrics=["fad", "prdc"])
metrics.add_reference(reference)

print(metrics.evaluate(candidate))

# To compute APA, the input data must be pairs of context and stem (in the
# trailing dimension)
reference = np.random.random((n_windows, window_len, 2))
# Data can also be passed as a generator, to facilitate processing larger
# datasets
candidate = (np.random.random(window_len, 2) for _ in range(n_windows))

metrics = AudioMetrics(metrics=["fad", "prdc", "apa"])
metrics.add_reference(reference)
print(metrics.evaluate(candidate))
```

When computing APA the reference and candidate sets must be pairs of context and
stem. Note that when FAD and/or PRDC are computed as additional metrics, these
are only computed for the stems (the contexts are ignored for these metrics).


## Citation

To cite this work, please use:

>  M. Grachten and J. Nistal. (2025). Accompaniment Prompt Adherence: A Measure for Evaluating Music Accompaniment Systems. Proceedings of the International Conference on Acoustics, Speech, and Signal Processing. IEEE. Hyderabad, India.

