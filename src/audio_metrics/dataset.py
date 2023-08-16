import os
import concurrent.futures as cf
from pathlib import Path
from functools import partial
from typing import Optional
import subprocess
from tqdm import tqdm
from queue import Queue
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
        for track in tqdm(self.generator):
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


def _push_tasks(items, executor, preprocessor, queue):
    for item in items:
        fut = executor.submit(preprocessor, item)
        # crucially, this blocks until there is space on the queue
        queue.put(fut)


def async_preprocessor(
    items,
    preprocessor,
    num_workers=None,
    buffer_size=10,
):
    """Apply `preprocessor` to `items` in parallel threads and yield the
    results. `buffer_size` controls how many items are preprocessed in
    advance. This limit is necessary to avoid the preprocessing getting ahead
    too far of the consumer of this generator, which can lead to excessive
    memory use.
    """

    queue = Queue(maxsize=buffer_size)
    with (
        cf.ThreadPoolExecutor(1) as pusher_thread,
        cf.ThreadPoolExecutor(num_workers) as workers,
    ):
        pusher = pusher_thread.submit(
            _push_tasks, items, workers, preprocessor, queue
        )
        while not pusher.done():
            yield queue.get().result()
        while not queue.empty():
            yield queue.get().result()


# def sync_preprocessor(
#     items,
#     preprocessor,
#     num_workers=None,
# ):
#     for item in items:
#         yield preprocessor(item)


class Embedder:
    def __init__(self):
        self.sr = None

    def preprocess(self, item):
        print(item)
        # given audio (array), return a torch tensor of the data that should go
        # into the embedder, with a leading batch dimension
        if isinstance(item, (Path, str)):
            # load audio as numpy array from file
            audio, sr = load_audio(item, self.sr, mono=True, dtype=np.float32)
        elif isinstance(item, (tuple, list)):
            audio, sr = item
            if sr is not None and sr != self.sr:
                audio = resampy.resample(audio, sr, self.sr)
        else:
            raise NotImplementedError(
                "argument must be tuple of (audio_waveform, samplerate)"
            )
        return audio, sr

    def embed(self, iterable):
        raise NotImplementedError()
