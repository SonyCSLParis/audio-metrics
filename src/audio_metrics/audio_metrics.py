import os
import dataclasses
from pathlib import Path
import itertools
from collections import defaultdict

import scipy
import torch
import numpy as np
from prdc import prdc

from .vggish import VGGish
from .clap import CLAP
from .dataset import (
    async_audio_loader,
    GeneratorDataset,
    preprocess_items,
    audiofile_generator,
    async_preprocessor,
)
from .fad import compute_frechet_distance, mu_sigma_from_activations
from .density_coverage import compute_density_coverage


@dataclasses.dataclass()
class MetricInputData:
    activations: np.ndarray

    @property
    def mu(self):
        if "_mu" not in self.__dict__:
            mu, sigma = mu_sigma_from_activations(self.activations)
            self.__dict__["_mu"] = mu
            self.__dict__["_sigma"] = sigma
        return self.__dict__["_mu"]

    @property
    def sigma(self):
        if "_sigma" not in self.__dict__:
            mu, sigma = mu_sigma_from_activations(self.activations)
            self.__dict__["_mu"] = mu
            self.__dict__["_sigma"] = sigma
        return self.__dict__["_sigma"]

    def get_radii(self, k_neighbor):
        key = f"radii_{k_neighbor}"
        radii = self.__dict__.get(key)
        if radii is None:
            radii = prdc.compute_nearest_neighbour_distances(
                self.activations, k_neighbor
            )
            self.__dict__[key] = radii
        return radii

    def __len__(self):
        return len(self.activations)

    @property
    def num_samples(self):
        return len(self.activations)

    @classmethod
    def from_npz_file(cls, fp):
        with np.load(fp) as data:
            instance = cls(data["activations"])
            instance.__dict__.update(data)
            return instance

    def to_npz_file(self, fp):
        np.savez(fp, **self.__dict__)


class AudioMetrics:
    """A class for distribution-based quality metrics of generated audio, based on
    audio embeddings.  Currently supported metrics:

    * Fréchet Audio Distance (FAD)
    * Density and Coverage

    """

    def __init__(
        self,
        device=None,
        background_data=None,
        batch_size=1,
        num_workers=1,
        k_neighbor=2,
        random_weights=False,
    ):
        if device is None:
            device = torch.device("cpu")
        self.embedders = {
            "vggish": VGGish(device),
            "clap": CLAP(device),
        }
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_neighbor = k_neighbor
        self.bg_data = None
        self.activations = defaultdict(list)
        if background_data is not None:
            self.prepare_background(background_data)

    def prepare_background(self, background_data):
        self.bg_data = self.load_metric_input_data(background_data)

    def load_metric_input_data(self, source, recursive=True, num_workers=None):
        # source can be either:
        # 1. An .npz file containing the precomputed data (as produced by `save_background_statistics()`)
        # 2. A directory path from which to load audio files
        # 3. An iterator over pairs (signal, samplerate), where signal is a 1-d
        #    numpy array
        if isinstance(source, (str, Path)):
            assert os.path.exists(source)
            source_fp = Path(source)

            if source_fp.is_file():
                # assume source is npz file
                return MetricInputData.from_npz_file(source_fp)

            elif source_fp.is_dir():
                input_items = audiofile_generator(source_fp, recursive)
            else:
                raise NotImplementedError(f"Cannot load data from {source}")
        else:
            # source should be an iterable over tuples of either (audio_fp, sr),
            # or (audio_array, sr)
            input_items = source

        preprocessors = {
            name: embedder.preprocess
            for name, embedder in self.embedders.items()
        }
        result = {}
        batch_size = 100
        batch = list(itertools.islice(input_items, batch_size))
        while batch:
            for name, preprocessor in preprocessors.items():
                data_iter = async_preprocessor(batch, preprocessor)
                dataset = GeneratorDataset(data_iter)
                activation_dict = self.embedders[name].embed(dataset)
                for layer_name, activations in activation_dict.items():
                    result[(name, layer_name)] = MetricInputData(activations)
            batch = list(itertools.islice(input_items, batch_size))
        return result

    # begin incremental API

    def add(self, iterator, label=None):
        data_iter = preprocess_items(iterator, self.model._preprocess)
        dataset = GeneratorDataset(data_iter)
        activations = get_activations(
            dataset,
            model=self.model,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.activations[label].append(activations)

    def reset(self, label):
        self.activations[label] = []

    def evaluate(self, label):
        if not self.activations[label]:
            raise Exception("Nothing to evaluate, use .add() to add data first")
        activations = np.concatenate(self.activations[label], 0)
        fake_data = MetricInputData(activations)
        fad_value = compute_frechet_distance(
            self.bg_data.mu, self.bg_data.sigma, fake_data.mu, fake_data.sigma
        )
        density, coverage = compute_density_coverage(
            self.bg_data, fake_data, self.k_neighbor
        )
        return dict(
            fad=fad_value,
            density=density,
            coverage=coverage,
            n_real=len(self.bg_data),
            n_fake=len(fake_data),
        )

    # end incremental API

    def __call__(self, source):
        return self.compare_to_background(source)

    def compare_to_background(self, source=None):
        if self.bg_data is None:
            raise RuntimeError(
                "Background data not available. Please provide data using `prepare_background()`"
            )
        if source is None:
            if not self.activations:
                raise Exception(
                    "No source specified, and not activations accumulated for computing metrics"
                )
            else:
                activations = np.concatenate(self.activations, 0)
                fake_data = MetricInputData(activations)
        else:
            fake_data_dict = self.load_metric_input_data(source)
        result = dict()
        for key, real_data in self.bg_data.items():
            fake_data = fake_data_dict[key]
            fad_value = compute_frechet_distance(
                real_data.mu, real_data.sigma, fake_data.mu, fake_data.sigma
            )
            density, coverage = compute_density_coverage(
                real_data, fake_data, self.k_neighbor
            )
            key_str = "_".join(key)
            result[f"fad_{key_str}"] = fad_value
            result[f"density_{key_str}"] = density
            result[f"coverage_{key_str}"] = coverage
            # n_real=len(real_data),
            # n_fake=len(fake_data),
        return result

    def expected_coverage_for_samplesize(self, n_fake):
        # A rough estimate of the coverage of an arbitrary subset of size
        # `n_fake` of the background data.

        # TODO: check that we have self.bg_data
        n_real = self.bg_data.num_samples
        result = n_fake / n_real
        result *= coverage_correction_factor(self.k_neighbor)
        return result

    def save_background_statistics(self, outfile, ensure_radii=False):
        if ensure_radii:
            # make sure radii are computed before we save
            self.bg_data.get_radii(self.k_neighbor)
        self.bg_data.to_npz_file(outfile)


def coverage_correction_factor(k_neighbor):
    # this is a naive estimate of the expected number of neighborhoods that
    # cover a sample for a given neighborhood size
    return (
        sum(scipy.special.comb(k_neighbor, i + 1) for i in range(k_neighbor))
        / k_neighbor
    )
