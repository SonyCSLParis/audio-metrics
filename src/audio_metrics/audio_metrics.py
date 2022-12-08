import os
import dataclasses
from pathlib import Path

import scipy
import torch
import numpy as np
from prdc import prdc

from .vggish import get_vggish_model, get_activations
from .dataset import iter_data_from_path, GeneratorDataset, preprocess_items
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
    ):
        if device is None:
            device = torch.device("cpu")
        self.model = get_vggish_model(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_neighbor = k_neighbor
        self.bg_data = None
        if background_data is not None:
            self.prepare_background(background_data)

    def prepare_background(self, background_data):
        self.bg_data = self.load_metric_input_data(background_data)

    def load_metric_input_data(self, source, recursive=True, num_workers=None):
        # source can be either:
        # 1. An .npz file containing the precomputed data
        # 2. A directory path from which to load audio files
        # 3. An iterator over pairs (signal, samplerate), where signal is a 1-d
        #    numpy array
        if isinstance(source, (str, Path)):
            assert os.path.exists(source)
            source_fp = Path(source)

            if source_fp.is_file():
                # assume source is npz file
                return MetricInputData.from_npz_file(source_fp)

            if source_fp.is_dir():
                # recursive, num_workers, model
                data_iter = iter_data_from_path(
                    source_fp,
                    recursive,
                    num_workers,
                    self.model._preprocess,
                )
                return self._metrics_input_data_from_iter(data_iter)

            # TODO: can we reach this?
        else:
            # assume source iterable
            data_iter = preprocess_items(source, self.model._preprocess)
            return self._metrics_input_data_from_iter(data_iter)

    def _metrics_input_data_from_iter(self, iterator):
        dataset = GeneratorDataset(iterator)
        activations = get_activations(
            dataset,
            model=self.model,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return MetricInputData(activations)

    def compare_to_background(self, source):
        if self.bg_data is None:
            raise RuntimeError(
                "Background data not available. Please provide data using `prepare_background()`"
            )
        fake_data = self.load_metric_input_data(source)
        fad_value = compute_frechet_distance(
            self.bg_data.mu,
            self.bg_data.sigma,
            fake_data.mu,
            fake_data.sigma,
        )
        density, coverage = compute_density_coverage(
            self.bg_data,
            fake_data,
            self.k_neighbor,
        )
        return fad_value, density, coverage

    def expected_coverage_for_samplesize(self, n_fake):
        # A rough estimate of the coverage of an arbitrary subset of size
        # `n_fake` of the background data.

        # TODO: check that we have self.bg_data
        n_real = self.bg_data.num_samples
        result = n_fake / n_real
        result *= coverage_correction_factor(self.k_neighbor)
        return result

    def save_base_statistics(self, outfile, ensure_radii=False):
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
