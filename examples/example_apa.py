from pathlib import Path
from functools import partial
import random
from itertools import chain, tee

import tqdm
import numpy as np
import torch
from prdc import prdc

from audio_metrics.example_utils import generate_audio_samples
from audio_metrics.mix_functions import MIX_FUNCTIONS, mix_pair
from audio_metrics.cpu_parallel import iterable_process as cpu_iterable_process
from audio_metrics.gpu_parallel import (
    iterable_process as gpu_iterable_process,
    GPUWorkerHandler,
)
from audio_metrics.fad import frechet_distance
from audio_metrics.dataset import async_audio_loader, multi_audio_slicer


def get_data_iterators(
    basedir=".",
    n_items=500,
    sr=48000,
):
    audio_dir1 = Path(basedir) / "audio_samples1"
    audio_dir2 = Path(basedir) / "audio_samples2"

    # print("generating 'real' and 'fake' audio samples")
    if not audio_dir1.exists():
        generate_audio_samples(audio_dir1, sr=sr, n_items=n_items)
    if not audio_dir2.exists():
        generate_audio_samples(audio_dir2, sr=sr, n_items=n_items)

    # load audio samples from files in `audio_dir`
    real1_song_iterator = async_audio_loader(audio_dir1 / "real", mono=False)
    fake1_song_iterator = async_audio_loader(audio_dir1 / "fake", mono=False)
    real2_song_iterator = async_audio_loader(audio_dir2 / "real", mono=False)

    real1_song_iterator = (item for item, _ in real1_song_iterator)
    fake1_song_iterator = (item for item, _ in fake1_song_iterator)
    real2_song_iterator = (item for item, _ in real2_song_iterator)

    return (
        real1_song_iterator,
        fake1_song_iterator,
        real2_song_iterator,
    )


class AudioMetricsData:
    def __init__(self, store_embeddings=True):
        self.mean = None
        self.n = None
        self.cov = None
        self.store_embeddings = store_embeddings
        self.embeddings = None
        self.radii = {}
        self.dtype = torch.float64

    def save(self, fp):
        torch.save(self.__dict__, fp)

    @classmethod
    def load(cls, fp):
        self = cls()
        self.__dict__.update(torch.load(fp, weights_only=True))
        return self

    def add(self, embeddings):
        mean = torch.mean(embeddings, 0).to(dtype=self.dtype)
        cov = torch.cov(embeddings.T).to(dtype=self.dtype)
        n = len(embeddings)
        self._update_stats(mean, cov, n)
        if self.store_embeddings:
            self._update_embeddings(embeddings)

    def get_radii(self, k_neighbor):
        key = f"radii_{k_neighbor}"
        radii = self.radii.get(key)
        if radii is None and self.embeddings is not None:
            radii = prdc.compute_nearest_neighbour_distances(
                self.embeddings.numpy(), k_neighbor
            )
            self.radii[key] = radii
        return radii

    def _update_embeddings(self, embeddings):
        if self.embeddings is None:
            self.embeddings = embeddings
            return
        self.embeddings = torch.cat((self.embeddings, embeddings))

    def __len__(self):
        return self.n or 0

    def _update_stats(self, mean, cov, n):
        if self.n is None:
            self.mean = mean
            self.cov = cov
            self.n = n
            return
        n_prod = self.n * n
        n_total = self.n + n
        new_mean = (self.n * self.mean + n * mean) / n_total
        diff_mean = self.mean - mean
        diff_mean_mat = torch.einsum("i,j->ij", diff_mean, diff_mean)
        w_self = (self.n - 1) / (n_total - 1)
        w_other = (n - 1) / (n_total - 1)
        w_diff = (n_prod / n_total) / (n_total - 1)
        new_cov = w_self * self.cov + w_other * cov + w_diff * diff_mean_mat
        self.n = n_total
        self.mean = new_mean
        self.cov = new_cov


# class MeanCovTorch:
#     def __init__(self, mean=None, cov=None, n=None):
#         self.mean = mean
#         self.n = n
#         self.cov = cov
#         self.dtype = torch.float64

#     def compute(self, embeddings):
#         self.mean = torch.mean(embeddings, 0).to(dtype=self.dtype)
#         self.n = len(embeddings)
#         self.cov = torch.cov(embeddings.T).to(dtype=self.dtype)
#         return self

#     def add(self, embeddings):
#         new = self + MeanCovTorch().compute(embeddings)
#         self.mean = new.mean
#         self.cov = new.cov
#         self.n = new.n

#     def __add__(self, other):
#         assert isinstance(other, MeanCovTorch)
#         if self.n is None:
#             return MeanCovTorch(other.mean.clone(), other.cov.clone(), other.n)

#         n_prod = self.n * other.n
#         n_total = self.n + other.n
#         mean = (self.n * self.mean + other.n * other.mean) / n_total
#         diff_mean = self.mean - other.mean
#         diff_mean_mat = torch.einsum("i,j->ij", diff_mean, diff_mean)
#         w_self = (self.n - 1) / (n_total - 1)
#         w_other = (other.n - 1) / (n_total - 1)
#         w_diff = (n_prod / n_total) / (n_total - 1)
#         cov = w_self * self.cov + w_other * other.cov + w_diff * diff_mean_mat
#         return MeanCovTorch(mean, cov, n_total)


def main():
    clap_cktpt = "/home/maarten/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt"
    clap_encoder = CLAP(ckpt=clap_cktpt)
    (
        real1_song_iterator,
        fake1_song_iterator,
        real2_song_iterator,
    ) = get_data_iterators(sr=clap_encoder.sr)
    # waveforms = chain(real1_song_iterator, fake1_song_iterator, real2_song_iterator)
    n_gpus = torch.cuda.device_count()
    gpu_handler = GPUWorkerHandler(n_gpus)

    R, Rp = embedding_pipeline_dual(
        real2_song_iterator, clap_encoder, n_gpus=n_gpus, gpu_handler=gpu_handler
    )
    C = embedding_pipeline_single(
        real1_song_iterator, clap_encoder, n_gpus=n_gpus, gpu_handler=gpu_handler
    )
    Cp = embedding_pipeline_single(
        fake1_song_iterator, clap_encoder, n_gpus=n_gpus, gpu_handler=gpu_handler
    )

    import ipdb

    ipdb.set_trace()
    C


if __name__ == "__main__":
    # main_test_shuffle()
    main()
