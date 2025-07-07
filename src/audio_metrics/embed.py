from collections.abc import Iterator
from typing import Literal
from functools import partial
from enum import IntEnum
from itertools import tee, chain

import torch
import numpy as np
import resampy

from audio_metrics.util.audio import multi_audio_slicer
from audio_metrics.util.shuffle import shuffle_stream
from audio_metrics.util.cpu_parallel import cpu_parallel
from audio_metrics.util.gpu_parallel import gpu_parallel
from audio_metrics.data import AudioMetricsData, ensure_ndarray


class ItemCategory(IntEnum):
    aligned = 1
    misaligned = 2
    stem = 3


def batch_accumulator(items, batch_size=32):
    audio = []
    category = []
    for item in items:
        audio.append(item["audio"])
        category.append(item["category"])
        if len(audio) == batch_size:
            yield {
                "audio": np.stack(audio),
                "category": np.array(category),
            }
            audio = []
            category = []
    if audio:
        yield {
            "audio": np.stack(audio),
            "category": np.array(category),
        }


def serialize_items(items1, items2=None, apa_mode=False, stems_mode=False):
    if items2 is None:
        item_pairs = ((item, None) for item in items1)
    else:
        item_pairs = zip(items1, items2)

    for item1, item2 in item_pairs:
        item1 = ensure_ndarray(item1)
        if apa_mode:
            # aligned pair
            if not len(item1.shape) == 2:
                msg = "When computing APA items should be tensors/arrays of shape [n_samples, 2] (pairing context and stem)"
                raise ValueError(msg)
            yield {"audio": item1, "category": ItemCategory.aligned}
            if item2 is not None:
                item2 = ensure_ndarray(item2)
                # misaligned pair
                assert len(item2.shape) == 2, msg
                misaliged = np.column_stack((item1[:, 0], item2[:, 1]))
                yield {"audio": misaliged, "category": ItemCategory.misaligned}
        if stems_mode:
            stem = item1[:, -1] if len(item1.shape) == 2 else item1
            yield {"audio": stem, "category": ItemCategory.stem}


def resample(item, **kwargs):
    return resampy.resample(ensure_ndarray(item).T, **kwargs).T


def mix_pair(data, mix_func, sr):
    if data["category"] == ItemCategory.stem:
        # TODO: loudness normalize
        return {"audio": data["audio"]}
    return {"audio": mix_func(data["audio"], sr=sr)}


def embedding_pipeline(
    waveforms,
    embedder,
    mix_function,
    gpu_handler=None,
    apa_mode: Literal["reference", "candidate"] | None = None,
    stems_mode: bool = False,
    store_mix_embeddings: bool = False,
    store_stem_embeddings: bool = False,
    batch_size: int = 32,
    win_dur: float = 5.0,
    song_buffer_size: int = 100,
    win_buffer_size: int = 1000,
    win_min_age: int = 100,
    seed: int | None = None,
    input_sr: int | None = None,
):
    """
    # Input Data

    Below are the valid input formats for `waveforms`:

    For APA:

        - iterable where items may have any of the following type:

            - context_stem_audio

            # - dictionary: {'context': context_audio, 'stem': stem_audio}

        - tensor/array of shape (first dimension contains context and stem):
          (batch, 2, n_samples)

    For FAD/KD/Precision/Recall:

        - iterable where items may have any of the following type:

            - stem_audio

            # - dictionary: {'stem': stem_audio}

        - tensor/array of shape: (batch, n_samples)

    In the above context_stem_audio/stem_audio are tensors/arrays of shape:

        - context_stem_audio: (n_samples, 2)

        - stem_audio: (n_samples,)

    The audio data is expected to be mono, and have a single samplerate.  The
    length of the audio may vary from one item to the next.  Note that, since
    the audio data will be windowed, in function of the window duration,
    trailing parts of each sample will be discarded.  To avoid this, ensure that
    all audio lengths are integer multiples of the window duration.
    """

    _mix_pair = partial(mix_pair, mix_func=mix_function, sr=embedder.sr)
    _resample = partial(
        resample, sr_orig=input_sr, sr_new=embedder.sr, filter="kaiser_fast"
    )

    items = iter(waveforms)

    if apa_mode == "reference":
        # 1. shuffle songs
        items = shuffle_stream(
            items, buffer_size=song_buffer_size, seed=seed, desc="shuffling songs"
        )

    # resample if needed
    if input_sr is not None and input_sr != embedder.sr:
        items = cpu_parallel(
            items,
            _resample,
            n_workers=64,
            in_buffer_size=32,
            out_buffer_size=32,
        )

    # 2. slice songs into windows
    items = multi_audio_slicer(items, win_dur, sr=embedder.sr)

    if apa_mode == "reference":
        # duplicate the iterator in order to create misaligned pairs
        items, shuffled_items = tee(items)
        # shuffle items
        shuffled_items = shuffle_stream(
            shuffled_items,
            buffer_size=win_buffer_size,
            min_age=win_min_age,
            seed=seed,
            desc="shuffling windows",
        )
    else:
        shuffled_items = None

    # create a stream of aligned/misaligned items
    items = serialize_items(items, shuffled_items, apa_mode, stems_mode)
    if apa_mode is not None:
        # mix the context stem pairs
        items = cpu_parallel(
            items,
            _mix_pair,
            n_workers=64,
            desc="mixing pairs",
            discard_input=False,
            in_buffer_size=32,
            out_buffer_size=32,
        )

    # accumulate into batches
    items = batch_accumulator(items, batch_size=batch_size)

    # compute the clap embeddings
    items = gpu_parallel(
        items,
        embedder,
        desc="computing embeddings",
        discard_input=False,
        gpu_worker_handler=gpu_handler,
        in_buffer_size=32,
        out_buffer_size=32,
    )

    # aggregate the statistics
    metrics_data = {}
    if apa_mode is not None:
        metrics_data[ItemCategory.aligned] = AudioMetricsData(store_mix_embeddings)
    if apa_mode == "reference":
        metrics_data[ItemCategory.misaligned] = AudioMetricsData(store_mix_embeddings)
    if stems_mode:
        metrics_data[ItemCategory.stem] = AudioMetricsData(store_stem_embeddings)

    for item in items:
        embedding = item["embedding"].cpu()
        aligned = item["category"] == ItemCategory.aligned
        misaligned = item["category"] == ItemCategory.misaligned
        stem = item["category"] == ItemCategory.stem
        if np.any(aligned):
            metrics_data[ItemCategory.aligned].add(embedding[aligned])
        if np.any(misaligned):
            metrics_data[ItemCategory.misaligned].add(embedding[misaligned])
        if np.any(stem):
            metrics_data[ItemCategory.stem].add(embedding[stem])
    return metrics_data
