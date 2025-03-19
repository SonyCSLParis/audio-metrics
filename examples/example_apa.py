from pathlib import Path
from functools import partial
import random
from itertools import chain, tee
from time import perf_counter

import numpy as np
import torch
import laion_clap

from audio_metrics.example_utils import generate_audio_samples
from audio_metrics.mix_functions import MIX_FUNCTIONS, mix_pair
from audio_metrics.cpu_parallel import iterable_process as cpu_iterable_process
from audio_metrics.gpu_parallel import iterable_process as gpu_iterable_process


from audio_metrics import async_audio_loader, multi_audio_slicer, APA


def shuffle_stream(iterator, buffer_size=100, seed=None):
    """
    Partially shuffle elements from an iterator using a reservoir sampling approach.

    This function maintains a buffer of elements from the iterator. As new elements
    are pulled from the iterator, they replace random elements in the buffer, which
    are then yielded. This provides a partial shuffle of the input stream while
    maintaining memory efficiency.

    :param iterator: Input iterator to be shuffled
    :param buffer_size: Size of the reservoir buffer. Larger buffers provide better
        shuffling at the cost of increased memory usage.

    :yield: Elements from the iterator in a partially shuffled order
    """
    buffer = []

    if seed is None:
        rng = random
    else:
        rng = random.Random(seed)

    # Fill the buffer
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break

    # Yield items from the buffer while replacing them with new ones
    for item in iterator:
        idx = rng.randint(0, len(buffer) - 1)
        yield buffer[idx]
        buffer[idx] = item

    # Yield the remaining items in the buffer
    rng.shuffle(buffer)
    yield from buffer


def get_data_iterators(win_dur, basedir=".", sr=48000):
    audio_dir1 = Path(basedir) / "audio_samples1"
    audio_dir2 = Path(basedir) / "audio_samples2"

    # print("generating 'real' and 'fake' audio samples")
    if not audio_dir1.exists():
        generate_audio_samples(audio_dir1, sr=sr)
    if not audio_dir2.exists():
        generate_audio_samples(audio_dir2, sr=sr)

    # load audio samples from files in `audio_dir`
    real1_song_iterator = async_audio_loader(audio_dir1 / "real", mono=False)
    fake1_song_iterator = async_audio_loader(audio_dir1 / "fake", mono=False)
    real2_song_iterator = async_audio_loader(audio_dir2 / "real", mono=False)

    # iterate over windows
    real1_win_iterator = multi_audio_slicer(real1_song_iterator, win_dur)
    fake1_win_iterator = multi_audio_slicer(fake1_song_iterator, win_dur)
    real2_win_iterator = multi_audio_slicer(real2_song_iterator, win_dur)

    return (
        real1_win_iterator,
        fake1_win_iterator,
        real2_win_iterator,
    )


class MeanCov:
    def __init__(self, mean=None, cov=None, n=None):
        self.mean = mean
        self.n = n
        self.cov = cov
        self.dtype = np.float64

    def compute(self, embeddings):
        self.mean = np.mean(embeddings, 0, dtype=self.dtype)
        self.n = len(embeddings)
        self.cov = np.cov(embeddings, rowvar=False, dtype=self.dtype)
        return self

    def __add__(self, other):
        assert isinstance(other, MeanCov)
        if self.n is None:
            return MeanCov(other.mean.copy(), other.cov.copy(), other.n)

        n_prod = self.n * other.n
        n_total = self.n + other.n
        mean = (self.n * self.mean + other.n * other.mean) / n_total
        diff_mean = self.mean - other.mean
        diff_mean_mat = np.einsum("i,j->ij", diff_mean, diff_mean)
        w_self = (self.n - 1) / (n_total - 1)
        w_other = (other.n - 1) / (n_total - 1)
        w_diff = (n_prod / n_total) / (n_total - 1)
        cov = w_self * self.cov + w_other * other.cov + w_diff * diff_mean_mat
        return MeanCov(mean, cov, n_total)


class CLAP:
    def __init__(self, ckpt, model_name="clap"):
        self.model_name = model_name
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.ckpt_path = Path(ckpt)
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
        return {"embedding": embedding.cpu().numpy()}


def batch_iterator(items, batch_size=32):
    batch = []
    for item in items:
        batch.append(item["audio"])
        if len(batch) == batch_size:
            yield {"audio": np.stack(batch)}
            batch = []
    if batch:
        yield {"audio": np.stack(batch)}


def embedding_pipeline(waveforms, clap_encoder):
    n_gpus = torch.cuda.device_count()
    mix_l0 = partial(mix_pair, mix_func=MIX_FUNCTIONS["L0"], sr=clap_encoder.sr)

    items = ({"ctx": ctx, "stem": stem} for ctx, stem, _ in waveforms)

    items = cpu_iterable_process(
        items,
        mix_l0,
        n_workers=16,
        desc="mixing pairs",
        in_buffer_size=16,
        out_buffer_size=16,
    )
    items = batch_iterator(items, batch_size=64)
    items = gpu_iterable_process(
        items,
        clap_encoder,
        desc="computing clap",
        n_gpus=n_gpus,
        in_buffer_size=n_gpus,
        out_buffer_size=n_gpus,
    )
    mean_cov = MeanCov()
    # t0 = perf_counter()
    for item in items:
        mean_cov = mean_cov + MeanCov().compute(item["embedding"])
    # t1 = perf_counter()
    # print(f"dur: {t1 - t0:.3f}s")
    return mean_cov


def main_test_shuffle():
    items = ((-i, i) for i in range(100))
    buffer_size = 1000
    seed = 1243
    _shuffle_stream = partial(shuffle_stream, buffer_size=buffer_size, seed=seed)

    pairs1, pairs2 = tee(items)
    stems2 = (s for _, s in pairs2)
    shuffled_stems = _shuffle_stream(stems2)
    for (ctx, stem), stem_shuf in zip(pairs1, shuffled_stems):
        print(ctx, stem, stem_shuf)


def main():
    win_dur = 5.0
    clap_cktpt = "/home/maarten/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt"
    clap_encoder = CLAP(ckpt=clap_cktpt)
    (
        real1_win_iterator,
        fake1_win_iterator,
        real2_win_iterator,
    ) = get_data_iterators(win_dur, sr=clap_encoder.sr)
    waveforms = chain(real1_win_iterator, fake1_win_iterator, real2_win_iterator)
    mean_cov = embedding_pipeline(waveforms, clap_encoder)

    import ipdb

    ipdb.set_trace()
    mean_cov
    # apa = APA()
    # reference_set = load_ctx_stem_pairs()
    # # candidate_set = ...
    # apa.set_reference(reference_set)
    # apa.compute(candidate_set)


if __name__ == "__main__":
    # main_test_shuffle()
    main()
