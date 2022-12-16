import os
import concurrent.futures as cf
from pathlib import Path

import numpy as np
import torch
import soundfile
import resampy

SAMPLE_RATE = 16000
AUDIO_FILE_PATTERNS = [f"*.{fmt}" for fmt in soundfile.available_formats()] + [
    f"*.{fmt.lower()}" for fmt in soundfile.available_formats()
]


def file_generator(path, recursive, patterns):
    path = Path(path)
    glob = path.rglob if recursive else path.glob
    for pattern in patterns:
        for item in glob(pattern):
            yield item


def vggish_audiofile_generator(path, recursive):
    for item in file_generator(path, recursive, AUDIO_FILE_PATTERNS):
        # vggish preprocessor needs samplerate as second arg (even if it isn't
        # used when the first arg is a file path)
        yield (os.fspath(item), None)


def load_audio_task(args):
    fname = args[0]  # args is an item as returned by get_items
    data, sr = soundfile.read(fname, always_2d=True)
    data = np.mean(data, axis=1)  # convert to mono
    if sr != SAMPLE_RATE:
        data = resampy.resample(data, sr, SAMPLE_RATE)
    return data, SAMPLE_RATE


def async_audio_loader(audio_dir, recursive=True, num_workers=None):
    items = vggish_audiofile_generator(audio_dir, recursive)
    with cf.ThreadPoolExecutor(num_workers) as pool:
        futures = {pool.submit(load_audio_task, item) for item in items}
        for fut in cf.as_completed(futures):
            yield fut.result()


class GeneratorDataset(torch.utils.data.IterableDataset):
    """Turn a generator that yields model preprocess output into an IterableDataset"""

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        for track in self.generator:
            for frame in track:
                yield frame


def preprocess_items(items, preprocessor):
    for item in items:
        result = preprocessor(*item)
        # result = result.new_empty(result.shape).normal_()
        yield result


def iter_data_from_path(folder_fp, recursive, num_workers, preprocessor):
    loader = async_audio_loader(folder_fp, recursive, num_workers)
    yield from preprocess_items(loader, preprocessor)
