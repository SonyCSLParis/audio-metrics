import os
import concurrent.futures as cf
from pathlib import Path
from functools import partial
from typing import Optional
import subprocess

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


def audiofile_generator_with_sr(path, recursive, file_patterns=None, sr=None):
    """Yield each audio file found in `path`, together with `sr`"""
    if not file_patterns:
        file_patterns = AUDIO_FILE_PATTERNS
    for item in file_generator(path, recursive, file_patterns):
        # vggish preprocessor needs samplerate as second arg (even if it isn't
        # used when the first arg is a file path)
        yield (os.fspath(item), sr)


def load_audio(fp, sr=None, mono=True, dtype=None):
    data, sr_orig = soundfile.read(fp, always_2d=True)
    if mono:
        data = np.mean(data, axis=1)  # convert to mono
    if sr is not None and sr_orig != sr:
        data = resampy.resample(data, sr_orig, sr)
    else:
        sr = sr_orig
    if dtype:
        data = data.astype(dtype)
    return data, sr


# def load_audio_ffmpeg(
#     fp: Path, sr: Optional[int] = None, mono: bool = False, timeout: Optional[float] = None
# ) -> np.ndarray:
#     cmd = ["ffmpeg", "-i", Path(fp).as_posix()]
#     if sr is not None:
#         cmd.extend(["-ar", f"{sr}"])
#     if mono:
#         cmd.extend(["-ac", "1"])
#     cmd.extend(["-f", "f32le", "-acodec", "pcm_f32le", "-"])

#     with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
#         try:
#             (stdout, stderr) = proc.communicate(timeout=timeout)
#         except subprocess.TimeoutExpired:
#             raise Exception("audio timeout error")
#         if proc.returncode != 0:
#             raise Exception(f"ffmpeg audio loading error {stderr.decode('utf-8')}")
#         return np.frombuffer(stdout, dtype=np.float32)


def async_audio_loader(
    audio_dir, recursive=True, num_workers=None, file_patterns=None, sr=None
):
    items = audiofile_generator_with_sr(
        audio_dir, recursive, file_patterns, sr=sr
    )

    # def load_audio_func(args):
    #     return load_audio_ffmpeg(*args, mono=True), SAMPLE_RATE

    with cf.ThreadPoolExecutor(num_workers) as pool:
        futures = {pool.submit(load_audio, *item) for item in items}
        # futures = {pool.submit(load_audio_func, item) for item in items}
        for fut in cf.as_completed(futures):
            yield fut.result()


class GeneratorDataset(torch.utils.data.IterableDataset):
    """Turn a generator that yields model preprocess output into an IterableDataset"""

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        for track in self.generator:
            print('track shape', track.shape)
            for frame in track:
                yield frame


def preprocess_items(items, preprocessor):
    for item in items:
        result = preprocessor(*item)
        # result = result.new_empty(result.shape).normal_()
        yield result


def audiofile_generator(path, recursive, file_patterns=None):
    """Yield each audio file found in `path`"""
    if not file_patterns:
        file_patterns = AUDIO_FILE_PATTERNS
    for item in file_generator(path, recursive, file_patterns):
        # vggish preprocessor needs samplerate as second arg (even if it isn't
        # used when the first arg is a file path)
        yield item.as_posix()


def async_preprocessor(
    # audio_dir,
    items,
    preprocessor,
    # recursive=True,
    num_workers=None,
    # file_patterns=None,
):
    # items = audiofile_generator(audio_dir, recursive, file_patterns)

    # def load_audio_func(args):
    #     return load_audio_ffmpeg(*args, mono=True), SAMPLE_RATE

    with cf.ThreadPoolExecutor(num_workers) as pool:
        futures = {pool.submit(preprocessor, item) for item in items}
        # futures = {pool.submit(load_audio_func, item) for item in items}
        for fut in cf.as_completed(futures):
            yield fut.result()


# def iter_data_from_path(folder_fp, recursive, num_workers, preprocessor):
#     loader = async_audio_loader(folder_fp, recursive, num_workers)
#     yield from preprocess_items(loader, preprocessor)


class Embedder:
    def __init__(self):
        self.sr = None

    def preprocess_path(self, audio_fp):
        # given audio (array), return a torch tensor of the data that should go
        # into the embedder, with a leading batch dimension
        raise NotImplementedError()

    def preprocess_array(self, audio, sr=None):
        # given audio (array), return a torch tensor of the data that should go
        # into the embedder, with a leading batch dimension
        raise NotImplementedError()

    def embed(self, iterable):
        raise NotImplementedError()
