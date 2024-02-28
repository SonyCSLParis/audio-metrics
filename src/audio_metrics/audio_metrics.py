import os
import dataclasses
from pathlib import Path
from functools import partial
import itertools
from collections import defaultdict
import concurrent.futures as cf

import sklearn.decomposition
import einops
from tqdm import tqdm
import scipy
import torch
import numpy as np
from prdc import prdc

from .vggish import VGGish
from .clap import CLAP
from .openl3 import OpenL3
from .dataset import (
    GeneratorDataset,
    preprocess_items,
    audiofile_generator,
    async_processor,
    audio_slicer_old,
)
from .fad import (
    compute_frechet_distance,
    mu_sigma_from_activations,
    frechet_distance,
)
from .kid import (
    compute_kernel_distance,
    KEY_METRIC_KID_MEAN,
    KEY_METRIC_KID_STD,
)

from .density_coverage import compute_density_coverage


def save_embeddings(outfile, embeddings):
    to_save = {}
    for (model, layer), activations in embeddings.items():
        if "/" in model:
            raise Exception(
                f'Saving names containing "/" is not supported: {model}'
            )
        if "/" in layer:
            raise Exception(
                f'Saving names containing "/" is not supported: {layer}'
            )
        # for name, data in mid.__dict__.items():
        #     key = f"{model}/{layer}/{name}"
        #     to_save[key] = data
        key = f"{model}/{layer}/activations"
        to_save[key] = activations
    np.savez(outfile, **to_save)


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
    * Density and Coverage

    """

    def __init__(
        self,
        device=None,
        background_data=None,
        batch_size=1,
        num_workers=1,
        k_neighbor=2,
        use_embedders=["vggish"],
    ):
        if device is None:
            device = torch.device("cpu")
        self.embedders = {}
        self._set_embedders(use_embedders, device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_neighbor = k_neighbor
        self.bg_data = None
        self.activations = defaultdict(list)
        if background_data is not None:
            self.prepare_background(background_data)

        self._pca_projectors = {}
        self._pca_n_components = None
        self._pca_n_whiten = None
        self._pca_bg_data = None

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
            projector = sklearn.decomposition.PCA(
                n_components=self._pca_n_components, whiten=self._pca_whiten
            )
            result[key] = MetricInputData(
                projector.fit_transform(data.activations)
            )
            self._pca_projectors[key] = projector
        return result

    def project(self, data_dict):
        result = {}
        for key, data in data_dict.items():
            result[key] = MetricInputData(
                self._pca_projectors[key].transform(data.activations)
            )
        return result

    def _set_embedders(self, names, device):
        for name in names:
            if name == "vggish":
                embedder = VGGish(device)
            elif name == "clap":
                embedder = CLAP(device)
            elif name == "openl3":
                embedder = OpenL3(device)
            else:
                raise Exception(
                    f'embedder {name} is not one of {{"vggish", "clap", "openl3"}}, skipping'
                )
            self.embedders[name] = embedder
        assert self.embedders, "No valid embedders specified"

    @property
    def embedding_names(self):
        return [
            (emb_name, name)
            for emb_name, emb in self.embedders.items()
            for name in emb.names
        ]

    def prepare_background(
        self,
        background_data,
        progress=False,
        win_dur=None,
        combine_mode="average",
    ):
        self.bg_data = self.load_metric_input_data(
            background_data,
            progress=progress,
            win_dur=win_dur,
            combine_mode=combine_mode,
        )

    def load_metric_input_data(
        self,
        source,
        recursive=True,
        num_workers=None,
        progress=False,
        win_dur=None,
        combine_mode="average",  # "average" | "concatenate"
        executor=None,
    ):
        # source can be either:
        # 1. An .npz file containing the precomputed data
        #    (as produced by `save_background_statistics()`)
        # 2. A directory path from which to load audio files
        # 3. An iterator over pairs (signal, samplerate), where signal is a 1-d
        #    numpy array
        # 4. A MetricsInputData instance (which is checked for validity and returned as is
        if isinstance(source, (str, Path)):
            assert os.path.exists(source)
            source_fp = Path(source)

            if source_fp.is_file():
                # assume source is npz file
                inp_data = self.load_metric_input_data_from_file(source_fp)
                self._filter_data_(inp_data)
                return inp_data

            if source_fp.is_dir():
                input_items = audiofile_generator(source_fp, recursive)
            else:
                raise NotImplementedError(f"Cannot load data from {source}")
        elif isinstance(source, dict):
            filtered = {}
            for name in self.embedding_names:
                try:
                    filtered[name] = source[name]
                except KeyError:
                    raise f"data for embedder {name} missing in source"
            return filtered
        else:
            # source should be an iterable over tuples of either (audio_fp, sr),
            # or (audio_array, sr)
            input_items = source

        result = defaultdict(list)
        if progress:
            prog = tqdm()
        else:
            prog = None

        have_data = False
        # num_workers = 10
        same_size = win_dur is not None

        # SYNC LOOP:
        # 1. preprocess file with all embedder proprocessors
        # 2. compute embeddings with all embedders

        for item in input_items:
            sr_versions = {}
            for name, embedder in self.embedders.items():
                key = (embedder.sr, embedder.mono)
                if key not in sr_versions:
                    # with cf.ProcessPoolExecutor() as executor:
                    #     fut = executor.submit(embedder.preprocess, item)
                    #     preprocessed = [fut.result()]
                    preprocessed = [embedder.preprocess(item)]
                    if win_dur is None:
                        sr_versions[key] = preprocessed
                    else:
                        sr_versions[key] = list(
                            audio_slicer_old(preprocessed, win_dur)
                        )

            for name, embedder in self.embedders.items():
                key = (embedder.sr, embedder.mono)
                audio_sr_pairs = sr_versions[key]
                activation_dict = self.embedders[name].embed(
                    audio_sr_pairs,
                    same_size=same_size,
                )
                for layer_name, activations in activation_dict.items():
                    if win_dur is None:
                        # activations: list of 2D arrays
                        # print(
                        #     f"{name}, {layer_name}, len={len(activations)}",
                        #     [x.shape for x in activations],
                        # )
                        if combine_mode == "concatenate":
                            result[(name, layer_name)].extend(activations)
                        else:
                            acts = [
                                einops.reduce(act, "k d -> d", "mean")
                                for act in activations
                            ]
                            acts = einops.rearrange(acts, "k d -> k d")

                            result[(name, layer_name)].append(acts)
                    else:
                        # 3D array
                        # print(
                        #     f"{name: <10s} / {layer_name: <30s}, shape={activations.shape}",
                        # )
                        if combine_mode == "concatenate":
                            acts = einops.rearrange(
                                activations, "... d -> (...) d"
                            )
                        else:
                            acts = einops.reduce(
                                activations, "k l d -> k d", "mean"
                            )
                        result[(name, layer_name)].append(acts)

            if prog is not None:
                prog.update()
            # batch = list(itertools.islice(input_items, batch_size))
            have_data = True

        if not have_data:
            print(f"Source {input_items} did not yield any data")
        else:
            for key, acts in result.items():
                result[key] = MetricInputData(np.concatenate(acts))

        return result

    # begin incremental API (WARNING: not up-to-date)

    # def add(self, iterator, label=None):
    #     data_iter = preprocess_items(iterator, self.model._preprocess)
    #     dataset = GeneratorDataset(data_iter)
    #     activations = get_activations(
    #         dataset,
    #         model=self.model,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #     )
    #     self.activations[label].append(activations)

    # def reset(self, label):
    #     self.activations[label] = []

    # def evaluate(self, label):
    #     if not self.activations[label]:
    #         raise Exception("Nothing to evaluate, use .add() to add data first")
    #     activations = np.concatenate(self.activations[label], 0)
    #     fake_data = MetricInputData(activations)
    #     fad_value = compute_frechet_distance(
    #         fake_data.mu, fake_data.sigma, self.bg_data.mu, self.bg_data.sigma
    #     )
    #     density, coverage = compute_density_coverage(
    #         self.bg_data, fake_data, self.k_neighbor
    #     )
    #     return dict(
    #         fad=fad_value,
    #         density=density,
    #         coverage=coverage,
    #         n_real=len(self.bg_data),
    #         n_fake=len(fake_data),
    #     )

    # end incremental API

    def __call__(self, source):
        return self.compare_to_background(source)

    def compare_to_background(
        self, source=None, return_data=False, progress=False, win_dur=None
    ):
        if self.bg_data is None:
            raise RuntimeError(
                "Background data not available. Please provide data using `prepare_background()`"
            )
        if source is None:
            if not self.activations:
                raise Exception(
                    "No source specified, and no activations accumulated for computing metrics"
                )
            else:
                # this is supposed to compare background to itself but, is
                # probably not functional as is (not kept uptodate)
                raise NotImplementedError("Fix me")
                # activations = np.concatenate(self.activations, 0)
                # fake_data = MetricInputData(activations)
        else:
            fake_data_dict = self.load_metric_input_data(source)
        result = dict()

        if self.has_pca():
            real_data_dict = self._pca_bg_data
            fake_data_dict = self.project(fake_data_dict)
        else:
            real_data_dict = self.bg_data

        for key, real_data in real_data_dict.items():
            fake_data = fake_data_dict[key]
            try:
                fad_value_1 = frechet_distance(
                    fake_data.mu, fake_data.sigma, real_data.mu, real_data.sigma
                )
                print("fad 1", fad_value_1)
                fad_value_2 = compute_frechet_distance(
                    fake_data.mu, fake_data.sigma, real_data.mu, real_data.sigma
                )
                print("fad 2", fad_value_2)
                fad_value = fad_value_2
            except ValueError:
                fad_value = np.nan

            key_str = "_".join(key)
            result[f"fad_{key_str}"] = fad_value

            kid_vals = compute_kernel_distance(
                torch.from_numpy(fake_data.activations),
                torch.from_numpy(real_data.activations),
            )
            for kid_name, kid_val in kid_vals.items():
                result[f"{kid_name}_{key_str}"] = kid_val
            result["n_real"] = len(real_data)
            result["n_fake"] = len(fake_data)

            # n_neighbors = min(
            #     result["n_real"], result["n_fake"], self.k_neighbor
            # )
            # print(
            #     "nnei",
            #     n_neighbors,
            #     real_data.activations.shape,
            #     fake_data.activations.shape,
            # )
            # density, coverage = compute_density_coverage(
            #     real_data, fake_data, n_neighbors
            # )
            # result[f"density_{key_str}"] = density
            # result[f"coverage_{key_str}"] = coverage

        if return_data:
            return result, fake_data_dict
        else:
            return result

    def expected_coverage_for_samplesize(self, n_fake):
        # A rough estimate of the coverage of an arbitrary subset of size
        # `n_fake` of the background data.

        # TODO: check that we have self.bg_data
        n_real = self.bg_data.num_samples
        result = n_fake / n_real
        result *= coverage_correction_factor(self.k_neighbor)
        return result

    # def save_background_statistics(self, outfile, ensure_radii=False):
    #     if ensure_radii:
    #         # make sure radii are computed before we save (to be reused in
    #         # future density/coverage computations)
    #         self.bg_data.get_radii(self.k_neighbor)
    #     self.bg_data.to_npz_file(outfile)

    def save_background_statistics(self, outfile, ensure_radii=False):
        to_save = {}
        for (model, layer), mid in self.bg_data.items():
            if "/" in model:
                raise Exception(
                    f'Saving names containing "/" is not supported: {model}'
                )
            if "/" in layer:
                raise Exception(
                    f'Saving names containing "/" is not supported: {layer}'
                )
            if ensure_radii:
                # make sure radii are computed before we save (to be reused in
                # future density/coverage computations)
                mid.get_radii(self.k_neighbor)
            for name, data in mid.__dict__.items():
                key = f"{model}/{layer}/{name}"
                to_save[key] = data
        np.savez(outfile, **to_save)

    @classmethod
    def save_embeddings_file(cls, embeddings, fp):
        to_save = {}
        for (model, layer), emb in embeddings.items():
            if "/" in model:
                raise Exception(
                    f'Saving names containing "/" is not supported: {model}'
                )
            if "/" in layer:
                raise Exception(
                    f'Saving names containing "/" is not supported: {layer}'
                )
            # for name, data in mid.__dict__.items():
            key = f"{model}/{layer}/activations"
            to_save[key] = emb
        np.savez(fp, **to_save)

    @classmethod
    def load_metric_input_data_from_file(cls, fp):
        bg_items = defaultdict(dict)
        with np.load(fp) as data_dict:
            for name, data in data_dict.items():
                try:
                    model, layer, stat_name = name.split("/", maxsplit=2)
                except ValueError as e:
                    raise Exception(
                        f"Unexpected/invalid background statistics name {name}"
                    ) from e
                # if embedder not in self.embedders
                bg_items[(model, layer)][stat_name] = data
        # self.bg_data = {}
        # for key, val in bg_items.items():
        #     self.bg_data[key] = MetricInputData.from_dict(val)
        bg_data = {}
        for key, val in bg_items.items():
            bg_data[key] = MetricInputData.from_dict(val)
        return bg_data

    def _filter_data_(self, inp_data):
        """Remove items pertaining to embedders we are not currently using"""
        for item in list(inp_data.keys()):
            embedder, _ = item
            if embedder not in self.embedders:
                print(f"Ignoring unnecessary stats for {item}")
                del inp_data[item]


def coverage_correction_factor(k_neighbor):
    # this is a naive estimate of the expected number of neighborhoods that
    # cover a sample for a given neighborhood size
    return (
        sum(scipy.special.comb(k_neighbor, i + 1) for i in range(k_neighbor))
        / k_neighbor
    )


# def create_sr_versions(item, spec, win_dur, pool=None):
