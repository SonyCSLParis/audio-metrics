import os
import concurrent.futures as cf
from pathlib import Path
from collections import defaultdict
from functools import partial
from typing import Optional
import subprocess
from tqdm import tqdm
import einops
from queue import Queue, Empty
import threading
import numpy as np
import torch
import soundfile
import resampy

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
        data = resampy.resample(data, sr_orig, sr, filter="kaiser_fast")
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


def _async_load_audio(item):
    fp, sr = item
    return load_audio(fp, sr=sr)


def async_audio_loader(
    audio_dir, recursive=True, num_workers=None, file_patterns=None, sr=None
):
    items = audiofile_generator_with_sr(
        audio_dir, recursive, file_patterns, sr=sr
    )
    yield from async_processor(
        items, _async_load_audio, num_workers=num_workers
    )


def audio_slicer_old(items, win_dur, hop_dur=None, drop_last=True):
    if not drop_last:
        raise NotImplementedError
    for audio, sr in items:
        win_len = int(sr * win_dur)
        N = len(audio)
        if hop_dur is None:
            hop_len = win_len
        else:
            hop_len = int(sr * hop_dur)
        for i in range(0, N - win_len + 1, hop_len):
            start = i
            end = start + win_len
            yield audio[start:end], sr


def audio_slicer(item, win_dur, hop_dur=None):
    audio, sr = item
    N = len(audio)
    win_len = int(sr * win_dur)
    hop_len = win_len if hop_dur is None else int(sr * hop_dur)
    for i in range(0, N - win_len + 1, hop_len):
        yield audio[i : i + win_len], sr


def multi_audio_slicer(items, win_dur, hop_dur=None, drop_last=True):
    if not drop_last:
        raise NotImplementedError
    for item in items:
        yield from audio_slicer(item, win_dur, hop_dur)


class GeneratorDataset(torch.utils.data.IterableDataset):
    """Turn a generator that yields model preprocess output into an IterableDataset"""

    def __init__(self, generator, multi=True):
        self.generator = generator
        self.multi = multi

    def __iter__(self):
        for item in self.generator:
            if self.multi:
                for frame in item:
                    yield frame
            else:
                yield item


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


def async_processor(
    items,
    func,
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
    with cf.ProcessPoolExecutor(num_workers) as pool:
        pusher = threading.Thread(
            target=_push_tasks,
            args=(items, pool, func, queue)
            # daemon=True,
        )
        pusher.start()
        while pusher.is_alive() or not queue.empty():
            try:
                got = queue.get(timeout=0.1)
            except Empty:
                pass
            else:
                result = got.result()
                queue.task_done()
                yield result


def _prep(item, target_sr, mono):
    """
    Given a pair (audio_waveform, samplerate), return a torch tensor of
    the data that should go into the embedder, with a leading batch
    dimension

    """
    if isinstance(item, (Path, str)):
        # load audio as numpy array from file
        audio, sr = load_audio(item, target_sr, mono=mono, dtype=np.float32)
    elif isinstance(item, (tuple, list)):
        if len(item) != 2:
            raise ValueError(
                "Input to `preprocess` must be a pair of (audio_waveform, samplerate)"
            )
        audio, sr = item
        if sr is not None and sr != target_sr:
            # print(f"resampling from {sr} to {target_sr}")
            if len(audio.shape) > 1:
                audio = audio.T
            audio = resampy.resample(
                audio, sr, target_sr, filter="kaiser_fast"
            ).T
        sr = target_sr
        if mono and len(audio.shape) > 1:
            audio = audio.mean(1)
    else:
        raise ValueError(
            "argument must be either a filepath (str, or Path), or a tuple of (audio_waveform, samplerate)"
        )
    # print('embedder preprocess out', audio.shape, sr)
    return audio.astype(np.float32), sr


class Embedder:
    def __init__(self, sr, mono):
        self.sr = sr
        self.mono = mono
        self.names = []
        self._preprocess = partial(_prep, target_sr=sr, mono=mono)

    def preprocess(self, item):
        """
        Given a pair (audio_waveform, samplerate), return a torch tensor of the data that should go into the embedder, with a leading batch dimension

        """
        if isinstance(item, (Path, str)):
            # load audio as numpy array from file
            audio, sr = load_audio(
                item, self.sr, mono=self.mono, dtype=np.float32
            )
        elif isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise ValueError(
                    "Input to `preprocess` must be a pair of (audio_waveform, samplerate)"
                )
            audio, sr = item
            if sr is not None and sr != self.sr:
                print(f"resampling from {sr} to {self.sr}")
                if len(audio.shape) > 1:
                    audio = audio.T
                audio = resampy.resample(
                    audio, sr, self.sr, filter="kaiser_fast"
                ).T
            sr = self.sr
            if self.mono and len(audio.shape) > 1:
                audio = audio.mean(1)
        else:
            raise ValueError(
                "argument must be either a filepath (str, or Path), or a tuple of (audio_waveform, samplerate)"
            )
        # print('embedder preprocess out', audio.shape, sr)
        return audio.astype(np.float32), sr

    @property
    def preprocess_key(self):
        return (self.sr, self.mono)

    def embed(self, iterable, same_size=False):
        raise NotImplementedError()

    def postprocess(self, activation_dict, combine_mode="concatenate"):
        result = defaultdict(list)
        for layer_name, activations in activation_dict.items():
            if combine_mode == "concatenate":
                # combine batch and time dims
                acts = einops.rearrange(activations, "... d -> (...) d")
                result[layer_name] = acts
            elif combine_mode == "stack":
                # keep batch and time dims
                acts = einops.rearrange(activations, "b ... d -> b (...) d")
                result[layer_name] = acts
            elif combine_mode == "average":
                # collapse time dim by mean
                acts = einops.reduce(activations, "k l d -> k d", "mean")
                result[layer_name] = acts
            else:
                msg = 'combine_mode must be one of ("concatenate", "stack", "average")'
                raise NotImplementedError(msg)
        return result

    # def postprocess(self, activation_dict, win_dur, combine_mode="concatenate"):
    #     result = defaultdict(list)
    #     for layer_name, activations in activation_dict.items():
    #         if win_dur is None:
    #             # activations: list of 2D arrays
    #             # print(
    #             #     f"{name}, {layer_name}, len={len(activations)}",
    #             #     [x.shape for x in activations],
    #             # )
    #             if combine_mode == "concatenate":
    #                 result[layer_name].extend(activations)
    #             else:
    #                 acts = [
    #                     einops.reduce(act, "k d -> d", "mean")
    #                     for act in activations
    #                 ]
    #                 acts = einops.rearrange(acts, "k d -> k d")

    #                 result[layer_name].append(acts)
    #         else:
    #             # 3D array
    #             # print(
    #             #     f"{name: <10s} / {layer_name: <30s}, shape={activations.shape}",
    #             # )
    #             if combine_mode == "concatenate":
    #                 acts = einops.rearrange(activations, "... d -> (...) d")
    #             else:
    #                 acts = einops.reduce(activations, "k l d -> k d", "mean")
    #             result[layer_name].append(acts)
    #     return result
