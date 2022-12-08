# Audio Metrics

This repository contains a python package to compute distribution-based metrics for audio data using embeddings.

* Fréchet Audio Distance (see https://arxiv.org/abs/1812.08466)
* Density and Coverage (see https://arxiv.org/abs/2002.09797)

The metrics use the publicly available pretrained VGGish model (trained on audio
event classification) to compute embeddings (see
https://arxiv.org/abs/1609.09430). In particular, it uses the 128-dimensional
embeddings from the last feature layer before the classification layer.

## Usage

```
from audio_metrics import AudioMetrics

metric = AudioMetrics()

# instantiate the metrics
metric.prepare_background('/path/to/real/audiofiles/')

fad, density, coverate = metric.compare_to_background('/path/to/fake/audiofiles')

metric.save_base_statistics('/path/to/background_stats.npz')

```


## TODO

Use alternative embeddings, e.g.:

* Penultimate 4096-dimensional VGGish feature layer
* A lower dimensional random projection of those features?
* Random VGGish embeddings (random parameter initialization)

## Credits

This repository was inspired by, and contains some code from https://github.com/spaghettiSystems/pytorch-fad
