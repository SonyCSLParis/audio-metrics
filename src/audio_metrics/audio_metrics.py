from pathlib import Path
from collections import defaultdict
import dataclasses
import warnings

import sklearn.decomposition
import torch
import numpy as np

from prdc import prdc
from .fad import (
    mu_sigma_from_activations,
    frechet_distance,
)
from .kid import compute_kernel_distance
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
            radii = prdc.compute_nearest_neighbour_distances(self.activations, k_neighbor)
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

    @classmethod
    def from_dict(cls, items):
        instance = cls(items["activations"])
        instance.__dict__.update(items)
        return instance

    def to_npz_file(self, fp):
        np.savez(fp, **self.__dict__)


class AudioMetrics:
    """A class for distribution-based quality metrics of generated audio, based on
    audio embeddings.  Currently supported metrics:

    * Fréchet Audio Distance (FAD)
    * Kernel Distance (KD)
    * Density and Coverage

    """

    def __init__(self, background_data=None, metrics=["fad", "kd", "dc"], k_neighbor=2):
        self.bg_data = background_data
        self.metrics = metrics
        self._pca_projectors = {}
        self._pca_n_components = None
        self._pca_whiten = None
        self._pca_bg_data = None
        self.k_neighbor = k_neighbor

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_background_data(self, source):
        self.bg_data = self.load_metric_input_data(source)

    def has_pca(self):
        return len(self._pca_projectors) > 0

    def set_pca_projection(self, n_components, whiten=True):
        self._pca_n_components = n_components
        self._pca_whiten = whiten
        if n_components is None:
            self._pca_projectors = {}
            self._pca_bg_data = None
        else:
            self._pca_bg_data = self._fit_pca()

    def _fit_pca(self):
        result = {}
        assert (
            self.bg_data is not None
        ), "Need background data to call set_pca_projection (use `prepare_background()`)"
        for key, data in self.bg_data.items():
            msg = f"The number of PCA components ({self._pca_n_components}) cannot be larger than the number of embedding vectors ({len(data.activations)})"
            assert self._pca_n_components <= len(data.activations), msg
            projector = sklearn.decomposition.PCA(
                n_components=self._pca_n_components, whiten=self._pca_whiten
            )
            result[key] = MetricInputData(projector.fit_transform(data.activations))
            # eagerly cache mu/sigma by accessing them
            result[key].mu
            result[key].sigma
            self._pca_projectors[key] = projector
        return result

    def project(self, data_dict):
        result = {}
        for key, data in data_dict.items():
            result[key] = MetricInputData(
                self._pca_projectors[key].transform(data.activations)
            )
        return result

    def load_metric_input_data(self, source):
        # source can be either:
        # 1. An .npz file containing the precomputed data
        #    (as produced by `save_background_statistics()`)
        # 2. a dictionary with numpy arrays as values,
        # 3. a dictionary with MetricsInputData instances as values
        if isinstance(source, dict):
            result = {}
            for k, v in source.items():
                if isinstance(v, MetricInputData):
                    result[k] = v
                else:
                    result[k] = MetricInputData(v)
            return result
        try:
            source_fp = Path(source)
        except TypeError as e:
            raise Exception(
                f"Source must be a file path to an npz file, or a MetricsInputData instance: {source}"
            ) from e
        if source_fp.is_file():
            # assume source is npz file
            return self.load_metric_input_data_from_file(source_fp)

    def __call__(self, source):
        return self.compare_to_background(source)

    def compare_to_background(self, source, return_data=False, device=None):
        if self.bg_data is None:
            raise RuntimeError(
                "Background data not available. Please provide data using `prepare_background()`"
            )
        fake_data_dict = self.load_metric_input_data(source)
        result = {}
        if self.has_pca():
            real_data_dict = self._pca_bg_data
            fake_data_dict = self.project(fake_data_dict)
        else:
            real_data_dict = self.bg_data

        for key, real_data in real_data_dict.items():
            fake_data = fake_data_dict[key]
            key_str = "_".join(key)
            if "fad" in self.metrics:
                try:
                    fad_value = frechet_distance(
                        fake_data.mu,
                        fake_data.sigma,
                        real_data.mu,
                        real_data.sigma,
                        device=device,
                    ).item()
                except ValueError as e:
                    warnings.warn(f"Error computing FAD: {e}, setting to NaN")
                    fad_value = np.nan

                result[f"fad_{key_str}"] = float(fad_value)

            if "kd" in self.metrics:
                kid_vals = compute_kernel_distance(
                    torch.from_numpy(fake_data.activations),
                    torch.from_numpy(real_data.activations),
                )
                for kid_name, kid_val in kid_vals.items():
                    result[f"{kid_name}_{key_str}"] = kid_val

            result["n_real"] = len(real_data)
            result["n_fake"] = len(fake_data)

            if "dc" in self.metrics:
                n_neighbors = min(result["n_real"], result["n_fake"], self.k_neighbor)
                density, coverage = compute_density_coverage(
                    real_data, fake_data, n_neighbors
                )
                result[f"density_{key_str}"] = density
                result[f"coverage_{key_str}"] = coverage

        result = dict(sorted(result.items()))

        if return_data:
            return result, fake_data_dict
        return result

    @staticmethod
    def check_name(name):
        if "/" in name:
            msg = f'Saving names containing "/" is not supported: {name}'
            raise Exception(msg)

    @staticmethod
    def load_metric_input_data_from_file(fp):
        with np.load(fp) as data_dict:
            return AudioMetrics.load_metric_input_data_from_dict(data_dict)

    @staticmethod
    def load_metric_input_data_from_dict(data_dict):
        bg_items = defaultdict(dict)
        for name, data in data_dict.items():
            try:
                model, layer, stat_name = name.split("/", maxsplit=2)
            except ValueError as e:
                raise ValueError(
                    f"Unexpected/invalid background statistics name {name}"
                ) from e
            bg_items[(model, layer)][stat_name] = data

        bg_data = {}
        for key, val in bg_items.items():
            bg_data[key] = MetricInputData.from_dict(val)
        return bg_data

    def get_serializable_background(self, ensure_radii=False):
        """obsolete?"""
        """Return background as a dict that can be passed to `np.savez`"""
        if self.bg_data is None:
            return None
        to_save = {}
        for (model, layer), mid in self.bg_data.items():
            AudioMetrics.check_name(model)
            AudioMetrics.check_name(layer)
            if ensure_radii:
                # make sure radii are computed before we save (to be reused in
                # future density/coverage computations)
                mid.get_radii(self.k_neighbor)

            for name, data in mid.__dict__.items():
                key = f"{model}/{layer}/{name}"
                to_save[key] = data
        return to_save

    def save_background_statistics(self, outfile, ensure_radii=False):
        """obsolete?"""
        to_save = self.get_serializable_background(ensure_radii)
        if to_save is None:
            return
        np.savez(outfile, **to_save)

    @staticmethod
    def save_embeddings_file(embeddings, fp):
        to_save = {}
        for (model, layer), emb in embeddings.items():
            AudioMetrics.check_name(model)
            AudioMetrics.check_name(layer)
            key = f"{model}/{layer}/activations"
            to_save[key] = emb
        np.savez(fp, **to_save)


def save_embeddings(outfile, embeddings):
    AudioMetrics.save_embeddings_file(embeddings, outfile)
