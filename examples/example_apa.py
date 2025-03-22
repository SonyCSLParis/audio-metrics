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


def shuffle_stream(iterator, buffer_size=100, seed=None, min_age=0, desc=None):
    """
    Shuffles an iterator using a fixed-size buffer with a minimum age constraint.

    Once the buffer is filled, each new item replaces an item from an "eligible"
    region of the buffer. The eligible region is defined such that an item that was
    just inserted is kept out of that region until at least 'min_age' subsequent
    replacements have occurred.

    To ensure at least one slot is always eligible, we clamp min_age to at most
    (buffer_size - 1). The eligible window is then of size:
          n_eligible = len(buffer) - effective_min_age
    where effective_min_age = min(min_age, len(buffer)-1).

    The algorithm uses an indices list and an offset pointer. The eligible region
    is the consecutive block starting at `offset` (modulo the buffer length) of
    size n_eligible.

    Parameters:
      iterator (iterable): The input iterable.
      buffer_size (int): The number of items to store in the buffer.
      seed (int, optional): A seed for random number generation.
      min_age (int): The minimum number of new insertions that must occur before
                     a slot can be replaced again.

    Yields:
      Items from the iterator in a shuffled order.
    """
    buffer = []
    indices = []
    offset = 0  # points to the beginning of the eligible region

    # Set up the random number generator.
    rng = random if seed is None else random.Random(seed)

    tqdm_kwargs = {"desc": desc, "leave": False} if desc else {"disable": True}
    progress = tqdm.tqdm(**tqdm_kwargs)
    progress.total = 0

    # Fill the buffer.
    for i in range(buffer_size):
        try:
            buffer.append(next(iterator))
            progress.total += 1
            progress.refresh()
            indices.append(i)
        except StopIteration:
            break

    total = len(buffer)
    if total == 0:
        return

    # Clamp min_age so that effective_min_age is at most total-1.
    effective_min_age = min(min_age, total - 1)
    # The eligible window size is then:
    n_eligible = total - effective_min_age  # always at least 1

    # Process new items from the iterator.
    for item in iterator:
        progress.total += 1
        progress.refresh()
        # Pick a random index from the eligible window.
        # Eligible window positions in 'indices' are offset, offset+1, ..., offset+n_eligible-1 (mod total).
        pos = rng.randrange(n_eligible)
        j = (offset + pos) % total
        idx = indices[j]
        yield buffer[idx]
        progress.update() and progress.refresh()
        # Replace the chosen slot with the new item.
        buffer[idx] = item
        # Swap the chosen index with the one at the current offset.
        indices[j], indices[offset] = indices[offset], indices[j]
        # Advance offset cyclically.
        offset = (offset + 1) % total

    # When the iterator is exhausted, yield the remaining items in random order.
    rng.shuffle(indices)
    for i in indices:
        yield buffer[i]
        progress.update() and progress.refresh()
    progress.close()


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


class CLAP:
    def __init__(self, ckpt, model_name="clap"):
        import laion_clap

        self.model_name = model_name
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.clap.load_ckpt(ckpt, verbose=False)

    @property
    def sr(self):
        return self.clap.model.audio_cfg.sample_rate  # 48000

    def get_device(self):
        return next(self.clap.parameters()).device

    @torch.no_grad()
    def forward(self, data, sr=None):
        audio = (
            torch.from_numpy(data["audio"])
            .float()
            .to(self.get_device(), non_blocking=True)
        )
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        embedding = self.clap.get_audio_embedding_from_data(audio, use_tensor=True)
        return {"embedding": embedding}


def batch_iterator_single(items, batch_size=32):
    audio = []
    for item in items:
        audio.append(item["audio"])
        if len(audio) == batch_size:
            yield {"audio": np.stack(audio)}
            audio = []
    if audio:
        yield {"audio": np.stack(audio)}


def batch_iterator_dual(items, batch_size=32):
    audio = []
    aligned = []
    for item in items:
        audio.append(item["audio"])
        aligned.append(item["aligned"])
        if len(audio) == batch_size:
            yield {"audio": np.stack(audio), "aligned": np.array(aligned)}
            audio = []
            aligned = []
    if audio:
        yield {"audio": np.stack(audio), "aligned": np.array(aligned)}


def audio_slicer(item, win_dur, sr, hop_dur=None):
    audio = item
    N = len(audio)
    win_len = int(sr * win_dur)
    hop_len = win_len if hop_dur is None else int(sr * hop_dur)
    for i in range(0, N - win_len + 1, hop_len):
        yield audio[i : i + win_len]


def multi_audio_slicer(items, win_dur, sr, hop_dur=None, drop_last=True):
    if not drop_last:
        raise NotImplementedError
    for item in items:
        yield from audio_slicer(item, win_dur, sr, hop_dur)


def serialize_pairs(pairs1, pairs2):
    for pair1, pair2 in zip(pairs1, pairs2):
        # aligned pair
        yield {"audio": pair1, "aligned": True}
        # misaligned pair
        misaliged = np.column_stack((pair1[:, 0], pair2[:, 1]))
        yield {"audio": misaliged, "aligned": False}


def embedding_pipeline_single(waveforms, clap_encoder, n_gpus, gpu_handler=None):
    win_dur = 5.0
    sr = 48000
    _mix_pair = partial(mix_pair, mix_func=MIX_FUNCTIONS["L0"], sr=clap_encoder.sr)
    # slice songs into windows
    items = multi_audio_slicer(waveforms, win_dur, sr=sr)
    # convert to dictionaries
    items = ({"audio": item} for item in items)
    # create mix the context stem pairs
    items = cpu_iterable_process(
        items, _mix_pair, n_workers=16, desc="mixing pairs", use_threads=True
    )
    # accumulate into batches
    items = batch_iterator_single(items, batch_size=128)
    # compute the clap embeddings
    items = gpu_iterable_process(
        items,
        clap_encoder,
        desc="computing clap",
        n_gpus=n_gpus,
        discard_input=False,
        gpu_worker_handler=gpu_handler,
    )
    # aggreate the statistics
    mean_cov = AudioMetricsData()
    for item in items:
        mean_cov.add(item["embedding"].cpu())
    return mean_cov


def embedding_pipeline_dual(waveforms, clap_encoder, n_gpus, gpu_handler=None):
    win_dur = 5.0
    sr = 48000
    song_buffer_size = 50
    win_buffer_size = 200
    win_min_age = 100
    seed = 1243
    _mix_pair = partial(mix_pair, mix_func=MIX_FUNCTIONS["L0"], sr=clap_encoder.sr)
    # shuffle songs
    items = shuffle_stream(
        waveforms, buffer_size=song_buffer_size, seed=seed, desc="shuffling songs"
    )
    # slice songs into windows
    items = multi_audio_slicer(items, win_dur, sr=sr)
    # duplicate the iterator in order to create misaligned pairs
    pairs1, pairs2 = tee(items)
    # shuffle pairs2
    pairs2 = shuffle_stream(
        pairs2,
        buffer_size=win_buffer_size,
        min_age=win_min_age,
        seed=seed,
        desc="shuffling windows",
    )
    # create a stream of aligned/misaligned items
    items = serialize_pairs(pairs1, pairs2)
    # mix the context stem pairs
    items = cpu_iterable_process(
        items,
        _mix_pair,
        n_workers=64,
        desc="mixing pairs",
        discard_input=False,
        in_buffer_size=256,
        out_buffer_size=256,
    )
    # accumulate into batches
    items = batch_iterator_dual(items, batch_size=32)
    # compute the clap embeddings
    items = gpu_iterable_process(
        items,
        clap_encoder,
        desc="computing clap",
        n_gpus=n_gpus,
        discard_input=False,
        gpu_worker_handler=gpu_handler,
        in_buffer_size=256,
        out_buffer_size=256,
    )
    # aggregate the statistics
    mean_cov_aligned = AudioMetricsData()
    mean_cov_misaligned = AudioMetricsData()
    for item in items:
        aligned = item["aligned"]
        embedding = item["embedding"].cpu()
        if np.any(aligned):
            mean_cov_aligned.add(embedding[aligned])
        if not np.all(aligned):
            mean_cov_misaligned.add(embedding[~aligned])
    return mean_cov_aligned, mean_cov_misaligned


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
