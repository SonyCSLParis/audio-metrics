#!/usr/bin/env python
import argparse
from pathlib import Path
from collections import defaultdict
import concurrent.futures as cf
from functools import partial
from itertools import tee
import numpy as np

import queue
import torch

from audio_metrics.dataset import async_audio_loader, audio_slicer, _prep
from audio_metrics.vggish import VGGish
from audio_metrics.clap import CLAP
from audio_metrics.openl3 import OpenL3
from audio_metrics import AudioMetrics


class ActivationStorage:
    """
    Gather activations from the pipeline, and store them as npz files (one
    file per song) in a folder. Since windows from songs can be spread across
    batches, we need to split the batches by song.
    """

    def __init__(self, names, outdir=None):
        self.outdir = outdir
        self.names = names
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
        self.song_acts = defaultdict(lambda: defaultdict(list))
        self.song_status = defaultdict(set)

    def _split_act_dict(self, act_dict, song_idx, split_points, name):
        song_acts = defaultdict(dict)
        song_idx_parts = np.split(song_idx, split_points)
        for k, v in act_dict.items():
            for idx, item in zip(song_idx_parts, np.split(v, split_points)):
                if len(idx) > 0:
                    song_acts[idx[0]][(name, k)] = item
        for i, act in song_acts.items():
            for k, v in act.items():
                self.song_acts[i][k].append(v)

    def add_item(self, act_dict, idx, name):
        idx = idx.numpy()
        song_ends = np.where(idx[:, -1])[0]
        self._split_act_dict(act_dict, idx[:, 0], song_ends + 1, name)
        ended_songs = idx[song_ends, 0]
        for ended_song in ended_songs:
            # print("ended songs", ended_song, name)
            self.song_status[ended_song].add(name)
            if len(self.song_status[ended_song]) == len(self.names):
                self._store_song(ended_song)
                del self.song_status[ended_song]

    def _store_song(self, i):
        act = self.song_acts.pop(i)
        fp = self.outdir / f"{i}.npz"
        stacked = dict((k, np.vstack(v)) for k, v in act.items())
        AudioMetrics.save_embeddings_file(stacked, fp)


class EmbedderPipeline:
    def __init__(self, embedders):
        self.embedders = embedders
        self.emb_by_key = defaultdict(list)
        for name, emb in self.embedders.items():
            self.emb_by_key[emb.preprocess_key].append((name, emb))
        self.executor = cf.ProcessPoolExecutor(max_workers=2)

    def _embed(self, embedder, dataset, batch_size, name, storage):
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=False
        )
        for act_dict, idx in embedder.embed_from_loader(dl):
            act_dict_avg = embedder.postprocess(act_dict, "average")
            storage.add_item(act_dict_avg, idx, name)

    def _feed(self, items, q):
        for i, item in enumerate(items):
            q.put((i, item))

    # def embed(self, audio_sr_iter, win_dur, outdir, max_workers=None):
    #     batch_size = 3
    #     # todo set queue size relative to max_workers
    #     pp_q = queue.Queue(maxsize=10)
    #     datasets = {
    #         (name, emb): QueueDataset(win_dur, name=name)
    #         for name, emb in self.embedders.items()
    #     }
    #     storage = ActivationStorage(list(self.embedders.keys()), outdir)
    #     with (
    #         cf.ProcessPoolExecutor(max_workers=max_workers) as executor,
    #         cf.ThreadPoolExecutor(max_workers=1) as feeder_exec,
    #         cf.ThreadPoolExecutor(
    #             max_workers=len(self.embedders)
    #         ) as embedder_exec,
    #     ):
    #         # start a future for a thread which sends work in through the queue
    #         futures = {
    #             feeder_exec.submit(
    #                 self._feed, audio_sr_iter, pp_q
    #             ): "FEEDER DONE"
    #         }
    #         embed_futures = {}
    #         for (name, embedder), dataset in datasets.items():
    #             fut = embedder_exec.submit(
    #                 # self._embed(embedder, dataset, emb_q)
    #                 self._embed,
    #                 embedder,
    #                 dataset,
    #                 batch_size,
    #                 name,
    #                 storage,
    #             )
    #             embed_futures[fut] = "EMBEDDER DONE"
    #         while futures:
    #             # check for status of the futures which are currently working
    #             done, _ = cf.wait(
    #                 futures, timeout=0.25, return_when=cf.FIRST_COMPLETED
    #             )
    #             # if there is incoming work, start a new future
    #             while not pp_q.empty():
    #                 # fetch an audio_sr pair from the queue
    #                 idx, audio_sr = pp_q.get()
    #                 # start each preprocess operation and mark result with idx and key
    #                 for key, embs in self.emb_by_key.items():
    #                     # preprocess funcs are equiv per key: only need one
    #                     _, emb = embs[0]
    #                     futures[executor.submit(emb._preprocess, audio_sr)] = (
    #                         idx,
    #                         key,
    #                     )

    #             # process any completed futures
    #             for future in done:
    #                 item = futures.pop(future)
    #                 if item == "FEEDER DONE":
    #                     continue
    #                 idx, key = item
    #                 try:
    #                     audio_sr = future.result()
    #                 except Exception as exc:
    #                     print(f"{(idx, key)} generated an exception: {exc}")
    #                     continue

    #                 for name, embedder in self.emb_by_key[key]:
    #                     datasets[(name, embedder)].add_item(audio_sr, idx)

    #         print("Finalizing datasets")
    #         for dataset in datasets.values():
    #             dataset.finalize()

    def embed(self, audio_sr_iter, win_dur, outdir, max_workers=None):
        batch_size = 3
        # todo set queue size relative to max_workers
        pp_q = queue.Queue(maxsize=10)
        datasets = {
            (name, emb): QueueDataset(win_dur, name=name)
            for name, emb in self.embedders.items()
        }
        storage = ActivationStorage(list(self.embedders.keys()), outdir)
        with (
            cf.ProcessPoolExecutor(max_workers=max_workers) as executor,
            cf.ThreadPoolExecutor(max_workers=1) as feeder_exec,
            cf.ThreadPoolExecutor(
                max_workers=len(self.embedders)
            ) as embedder_exec,
        ):
            # start a future for a thread which sends work in through the queue
            futures = {
                feeder_exec.submit(
                    self._feed, audio_sr_iter, pp_q
                ): "FEEDER DONE"
            }
            embed_futures = {}
            for (name, embedder), dataset in datasets.items():
                fut = embedder_exec.submit(
                    # self._embed(embedder, dataset, emb_q)
                    self._embed,
                    embedder,
                    dataset,
                    batch_size,
                    name,
                    storage,
                )
                embed_futures[fut] = "EMBEDDER DONE"
            while futures:
                # check for status of the futures which are currently working
                done, _ = cf.wait(
                    futures, timeout=0.25, return_when=cf.FIRST_COMPLETED
                )
                # if there is incoming work, start a new future
                while not pp_q.empty():
                    # fetch an audio_sr pair from the queue
                    idx, audio_sr = pp_q.get()
                    # start each preprocess operation and mark result with idx and key
                    for key, embs in self.emb_by_key.items():
                        # preprocess funcs are equiv per key: only need one
                        _, emb = embs[0]
                        futures[executor.submit(emb._preprocess, audio_sr)] = (
                            idx,
                            key,
                        )

                # process any completed futures
                for future in done:
                    item = futures.pop(future)
                    if item == "FEEDER DONE":
                        continue
                    idx, key = item
                    try:
                        audio_sr = future.result()
                    except Exception as exc:
                        print(f"{(idx, key)} generated an exception: {exc}")
                        continue

                    for name, embedder in self.emb_by_key[key]:
                        datasets[(name, embedder)].add_item(audio_sr, idx)

            print("Finalizing datasets")
            for dataset in datasets.values():
                dataset.finalize()


class QueueDataset(torch.utils.data.IterableDataset):
    def __init__(self, win_dur, queue_size=100, name=None):
        self.win_dur = win_dur
        self.queue = queue.Queue(maxsize=queue_size)
        self.name = name

    # def add_item(self, item, song_idx):
    #     for win_idx, audio_sr in enumerate(audio_slicer(item, self.win_dur)):
    #         self.queue.put((audio_sr, song_idx, win_idx))

    def add_item(self, item, song_idx):
        to_be_put = None
        for win_idx, audio_sr in enumerate(audio_slicer(item, self.win_dur)):
            if to_be_put is not None:
                self.queue.put(to_be_put)
            to_be_put = [audio_sr, song_idx, win_idx, 0]
        if to_be_put is not None:
            to_be_put[-1] = 1
            self.queue.put(to_be_put)

    def finalize(self):
        self.queue.put((None, None, None, None))

    def __next__(self):
        item, song_idx, win_idx, last = self.queue.get()
        if item is None:
            raise StopIteration()
        audio, _ = item
        return audio, torch.tensor([song_idx, last])

    def __iter__(self):
        return self


def main():
    parser = argparse.ArgumentParser(description="Do something")
    parser.add_argument("indir", type=Path, help=("audio directory"))
    parser.add_argument("outdir", type=Path, help=("output directory"))
    args = parser.parse_args()

    bass_file_pats = [
        "*Bass*.flac",
        "bass.wav",
        "*Bass*.wav",
        "808*.wav",
        "*Ukulele*.flac",
    ]
    # bass_file_pats = ["Bass.flac", "bass.wav", "*Bass*.wav", "808*.wav"]
    file_pats = bass_file_pats
    bg_loader = async_audio_loader(
        args.indir, file_patterns=file_pats, num_workers=16
    )
    dev = torch.device("cuda")
    embedders = {
        "openl3": OpenL3(dev),
        "clap": CLAP(dev),
        "vggish": VGGish(dev),
    }
    win_dur = 5.0
    p = EmbedderPipeline(embedders)
    p.embed(bg_loader, win_dur, args.outdir)


if __name__ == "__main__":
    main()
