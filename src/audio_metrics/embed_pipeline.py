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


# class ActivationStorage:
#     """
#     Gather activations from the pipeline, and store them as npz files (one
#     file per song) in a folder. Since windows from songs can be spread across
#     batches, we need to split the batches by song.
#     """

#     def __init__(self, names, outdir=None, out_queue=None):
#         self.outdir = outdir
#         self.out_queue = out_queue
#         self.names = names
#         if outdir:
#             outdir.mkdir(parents=True, exist_ok=True)
#         self.song_acts = defaultdict(lambda: defaultdict(list))
#         self.song_status = defaultdict(set)

#     def _split_act_dict(self, act_dict, song_idx, split_points, name):
#         song_acts = defaultdict(dict)
#         song_idx_parts = np.split(song_idx, split_points)
#         for k, v in act_dict.items():
#             for idx, item in zip(song_idx_parts, np.split(v, split_points)):
#                 if len(idx) > 0:
#                     song_acts[idx[0]][(name, k)] = item
#         for i, act in song_acts.items():
#             for k, v in act.items():
#                 self.song_acts[i][k].append(v)

#     def add_item(self, act_dict, idx, name):
#         idx = idx.numpy()
#         song_ends = np.where(idx[:, -1])[0]
#         self._split_act_dict(act_dict, idx[:, 0], song_ends + 1, name)
#         ended_songs = idx[song_ends, 0]
#         for ended_song in ended_songs:
#             # print("ended songs", ended_song, name)
#             self.song_status[ended_song].add(name)
#             if len(self.song_status[ended_song]) == len(self.names):
#                 self._store_song(ended_song)
#                 del self.song_status[ended_song]

#     def _store_song(self, i):
#         act = self.song_acts.pop(i)
#         stacked = dict((k, np.vstack(v)) for k, v in act.items())
#         if self.outdir:
#             fp = self.outdir / f"{i}.npz"
#             AudioMetrics.save_embeddings_file(stacked, fp)
#         if self.out_queue:
#             self.out_queue.put(stacked)


class ActivationStorage:
    """
    Gather activations from the pipeline, and aggregate them by item_id (an
    integer), when all activations for an item_id are in, pass it on to the
    queue.

    :param names: The names of the embedder models
    :param out_queue: The queue on which to put finished items
    :param ordered: Flag indicating whether to return the items according to
        their item_id order.
    """

    def __init__(
        self, names, in_queue, out_queue, ordered=True, return_index=False
    ):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.n_names = len(names)
        # keep unfinished item activations here:
        self.item_acts = defaultdict(dict)
        # keep track of the names we have for each ongoing item
        self.item_status = defaultdict(set)
        self.ordered = ordered
        self.return_index = return_index
        self.i = 0

    def run(self):
        while True:
            inp = self.in_queue.get()
            if inp is None:
                break
            self.add_item(*inp)

    def finalize(self):
        self.in_queue.put(None)

    def add_item(self, act_dict, item_ids, name):
        item_ids = item_ids.tolist()
        print("storing", item_ids, name)
        for k, v in act_dict.items():
            for item_id, item in zip(item_ids, v):
                self.item_acts[item_id][(name, k)] = item
                self.item_status[item_id].add(name)
        print("stored", item_ids, name)
        self._dispatch_finished_items()
        print("dispatcher ran")

    def _dispatch_finished_items(self):
        for item_id in sorted(self.item_status.keys()):
            is_complete = len(self.item_status[item_id]) == self.n_names
            is_due = not self.ordered or (self.ordered and item_id <= self.i)
            if is_complete and is_due:
                acts = self.item_acts.pop(item_id)
                self.item_status.pop(item_id)
                # if self.return_index:
                #     item = (item_id, acts)
                # else:
                #     item = acts
                self.out_queue.put((item_id, acts))
                self.i += 1


class EmbedderPipeline:
    def __init__(self, embedders):
        self.embedders = embedders
        self.emb_by_key = defaultdict(list)
        for name, emb in self.embedders.items():
            self.emb_by_key[emb.preprocess_key].append((name, emb))
        self.executor = cf.ProcessPoolExecutor(max_workers=2)

    # def _embed(self, embedder, dataset, batch_size, name, storage):
    #     dl = torch.utils.data.DataLoader(
    #         dataset, batch_size=batch_size, drop_last=False
    #     )
    #     for act_dict, idx in embedder.embed_from_loader(dl):
    #         print("embed", idx)
    #         act_dict_avg = embedder.postprocess(act_dict, "average")
    #         print("embed postprocessed", idx)
    #         storage.add_item(act_dict_avg, idx, name)
    #         print("post-processed item stored", idx)
    #     print("embedder done", name)

    def _embed(self, embedder, dataset, batch_size, name, storage_queue):
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=False
        )
        for act_dict, idx in embedder.embed_from_loader(dl):
            print("embed", idx)
            act_dict_avg = embedder.postprocess(act_dict, "average")
            print("embed postprocessed", idx)
            # storage.add_item(act_dict_avg, idx, name)
            storage_queue.put((act_dict_avg, idx, name))
            print("post-processed item stored", idx)
        print("embedder done", name)

    def _store(self, storage):
        storage.run()

    def _feed(self, items, q):
        for i, item in enumerate(items):
            print("feeding item", i, "qsize", q.qsize())
            q.put((i, item))

    # def embed(self, data_iter, win_dur, ordered=True, max_workers=None):
    def embed(self, data_iter, ordered=True, max_workers=None):
        """
        data_iter should be an iterator over pairs ((waveform, sr), identifier).
        waveform should be 1d, and identifier can be anything that is collatable
        by pytorch's default collate_fn.  It is returned as is along with the
        embeddings.
        """
        batch_size = 3
        # todo set queue size relative to max_workers
        preprocess_q = queue.Queue(maxsize=100)
        storage_q = queue.Queue()
        out_q = queue.Queue()
        datasets = {
            (name, emb): QueueDataset(name=name)
            for name, emb in self.embedders.items()
        }
        storage = ActivationStorage(
            list(self.embedders.keys()), storage_q, out_q, ordered=ordered
        )
        with (
            cf.ProcessPoolExecutor(max_workers=max_workers) as executor,
            cf.ThreadPoolExecutor(max_workers=1) as feeder_exec,
            cf.ThreadPoolExecutor(
                max_workers=len(self.embedders)
            ) as embedder_exec,
            cf.ThreadPoolExecutor(max_workers=1) as storage_exec,
        ):
            # start a future for a thread that sends work in through the queue
            futures = {
                feeder_exec.submit(
                    self._feed, data_iter, preprocess_q
                ): "FEEDER DONE"
            }
            embed_futures = []
            storage_future = storage_exec.submit(self._store, storage)
            for (name, embedder), dataset in datasets.items():
                fut = embedder_exec.submit(
                    # self._embed(embedder, dataset, emb_q)
                    self._embed,
                    embedder,
                    dataset,
                    batch_size,
                    name,
                    storage_q,
                )
                embed_futures.append(fut)
            while futures:
                print(
                    "working  pp out; futures",
                    preprocess_q.qsize(),
                    out_q.qsize(),
                    len(futures)
                    # futures.values(),
                )
                if len(futures) == 1:
                    print("last remaining future", futures.values())
                # check for status of the futures that are currently working
                done, _ = cf.wait(
                    futures, timeout=0.25, return_when=cf.FIRST_COMPLETED
                )
                # if there are any items to be preprocessed, pass them on to the
                # preprocessors
                while not preprocess_q.empty():
                    # fetch an audio_sr pair from the queue
                    item_ids, audio_sr = preprocess_q.get()
                    # start each preprocess operation and mark result with item_ids and key
                    for key, embs in self.emb_by_key.items():
                        # preprocess funcs are equiv per key: only need one
                        _, emb = embs[0]
                        futures[executor.submit(emb._preprocess, audio_sr)] = (
                            item_ids,
                            key,
                        )

                # process any completed futures (either feeder or preprocessed items)
                for future in done:
                    item = futures.pop(future)
                    if item == "FEEDER DONE":
                        print("feeder finished")
                        continue
                    item_ids, preprocess_key = item
                    try:
                        audio_sr = future.result()
                    except Exception as exc:
                        print(
                            f"{(item_ids, preprocess_key)} generated an exception: {exc}"
                        )
                        continue
                    print("  preprocessing returned for ", item)
                    # pass preprocessed items on to the corresponding embedders
                    for name, embedder in self.emb_by_key[preprocess_key]:
                        print("adding preprocessed item", item, "to dataset")
                        datasets[(name, embedder)].add_item(audio_sr, item_ids)
                        preprocess_q.task_done()

                while True:
                    try:
                        yield out_q.get_nowait()
                    except queue.Empty:
                        break
            preprocess_q.join()
            print("Finalizing datasets; futures:", len(futures))
            for dataset in datasets.values():
                dataset.finalize()
            print(embed_futures)
            cf.wait(embed_futures)
            storage.finalize()
            cf.wait((storage_future,))
            while not out_q.empty():
                yield out_q.get()


class QueueDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset wrapper around a queue.  This class expects ((audio, sr),
    id) items from the queue and yields (audio, id).  (TODO: drop sr somewhere
    else)
    """

    def __init__(self, queue_size=100, name=None):
        self.queue = queue.Queue(maxsize=queue_size)
        self.name = name

    def add_item(self, item, item_id):
        self.queue.put((item, item_id))

    def finalize(self):
        self.queue.put((None, None))

    def __next__(self):
        item, item_id = self.queue.get()
        # print("qds got", item)
        if item is None:
            raise StopIteration()
        audio, _ = item
        return audio, item_id

    def __iter__(self):
        return self


# NOTE: not a drop in replacement for QueueDataset (need to check)
class QueueSlicingDataset(torch.utils.data.IterableDataset):
    def __init__(self, win_dur, queue_size=100, name=None):
        self.win_dur = win_dur
        self.queue = queue.Queue(maxsize=queue_size)
        self.name = name

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
