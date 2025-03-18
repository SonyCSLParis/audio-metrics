#!/usr/bin/env python
from collections import defaultdict
from pathlib import Path
import tempfile
import concurrent.futures as cf
import queue
import traceback
import numpy as np
import torch


class ActivationStorage:
    """
    Gather activations from the pipeline, and aggregate them by item_id (an
    integer), when all activations for an item_id are in, pass it on to the
    queue.

    :param names: The names of the embedder models
    :param out_queue: The queue on which to put finished items
    :param ordered: Flag indicating whether to return the items according to
        their item_id order
    :param return_index: Flag indicating whether to return just the activations
        or a tuple (i, activations), where i is the index of the item.  This can
        be useful when ordered=False, and you need to keep track of the order
        externally
    """

    def __init__(self, names, out_queue, ordered=True, return_index=False):
        self.out_queue = out_queue
        self.n_names = len(names)
        # keep unfinished item activations here:
        self.item_acts = defaultdict(dict)
        # keep track of the names we have for each ongoing item
        self.item_status = defaultdict(set)
        self.ordered = ordered
        self.return_index = return_index
        self.i = 0

    def add_item(self, act_dict, item_ids, name):
        item_ids = item_ids.tolist()
        for k, v in act_dict.items():
            for item_id, item in zip(item_ids, v):
                self.item_acts[item_id][(name, k)] = item
                self.item_status[item_id].add(name)
        self._dispatch_finished_items()

    def _dispatch_finished_items(self):
        for item_id in sorted(self.item_status.keys()):
            is_complete = len(self.item_status[item_id]) == self.n_names
            is_due = not self.ordered or (self.ordered and item_id <= self.i)
            if is_complete and is_due:
                del self.item_status[item_id]
                acts = self.item_acts.pop(item_id)
                if self.return_index:
                    self.out_queue.put((item_id, acts))
                else:
                    self.out_queue.put(acts)
                self.i += 1


class FeedQueue(queue.Queue):
    def put(self, v):
        (i, (data, sr)) = v
        # This doesn't work, need name temporary file
        f = tempfile.TemporaryFile(delete=False)
        np.save(f.name, data)
        super().put((i, (f.name, sr)))

    def get(self):
        (i, (fn, sr)) = super().get()
        data = np.load(fn)
        Path(fn).unlink()
        return (i, (data, sr))


class EmbedderPipeline:
    def __init__(self, embedders):
        self.embedders = embedders
        self.emb_by_key = defaultdict(list)
        for name, emb in self.embedders.items():
            self.emb_by_key[emb.preprocess_key].append((name, emb))

    def _embed(self, embedder, dataset, batch_size, name, storage, combine_mode):
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)
        for act_dict, idx in embedder.embed_from_loader(dl):
            act_dict_joint = embedder.postprocess(act_dict, combine_mode)
            storage.add_item(act_dict_joint, idx, name)

    def _feed(self, items, q):
        for i, item in enumerate(items):
            tensor_item = (torch.as_tensor(item[0]), item[1])
            q.put((i, tensor_item))

    def embed_join(self, data_iter, **kwargs):
        return join_embeddings(self.embed(data_iter, **kwargs))

    def embed(
        self,
        data_iter,
        batch_size=3,
        ordered=True,
        max_workers=None,
        progress=None,
        combine_mode="average",
    ):
        """
        Embed waveforms provided by `data_iter`, and yield the embeddings.

        :param data_iter: An iterator over pairs (waveform, sr).  waveform
            should be 1d.

        :param batch_size: Batch size for computing embeddings (default 3)

        :param ordered: Flag indicating whether the embeddings should be yielded
            in the same order as the input, or in the order they become
            available.  Default: True.

        :param max_workers: Number of workers for the preprocessing pool

        :param progress: A tqdm instance, which will be updated at each yielded
            embedding.  Default: None

        :param combine_mode: Whether multiple embeddings per window are combined
            by averaging or by concatenation.  Possible values: {"average",
            "concatenate")
        """
        # todo set queue size relative to max_workers
        # preprocess_q = queue.Queue(maxsize=10)
        preprocess_q = FeedQueue(maxsize=100)
        out_q = queue.Queue()
        datasets = {
            (name, emb): QueueDataset(name=name) for name, emb in self.embedders.items()
        }
        storage = ActivationStorage(list(self.embedders.keys()), out_q, ordered=ordered)
        with (
            cf.ThreadPoolExecutor(max_workers=len(self.embedders) + 2) as pool1,
            cf.ThreadPoolExecutor(max_workers=max_workers) as pool2,
        ):
            # pool1 runs the feeder thread, and an embedding thread for each embedder
            # pool2 runs the preprocessor thread
            futures = {pool1.submit(self._feed, data_iter, preprocess_q): "FEEDER DONE"}
            embed_futures = []
            for (name, embedder), dataset in datasets.items():
                fut = pool1.submit(
                    self._embed,
                    embedder,
                    dataset,
                    batch_size,
                    name,
                    storage,
                    combine_mode,
                )
                embed_futures.append(fut)
            while futures:
                # check for status of the futures that are currently working
                done, _ = cf.wait(futures, timeout=0.25, return_when=cf.FIRST_COMPLETED)
                # if there are any items to be preprocessed, pass them on to the
                # preprocessors
                while not preprocess_q.empty():
                    # fetch an audio_sr pair from the queue
                    item_ids, audio_sr = preprocess_q.get()
                    # start each preprocess operation and mark result with item_ids and key
                    for key, embs in self.emb_by_key.items():
                        # preprocess funcs are equiv per key: only need one
                        _, emb = embs[0]
                        futures[pool2.submit(emb._preprocess, audio_sr)] = (
                            item_ids,
                            key,
                        )

                # process any completed futures (either feeder or preprocessed items)
                for future in done:
                    item = futures.pop(future)
                    if item == "FEEDER DONE":
                        try:
                            future.result()
                        except Exception as exc:
                            print("Exception in feeder:", exc)
                            traceback.print_exc()

                        continue
                    item_ids, preprocess_key = item
                    try:
                        audio_sr = future.result()
                    except Exception as exc:
                        print("Exception in preprocessor:", exc)
                        # TODO: don't raise but exit gracefully
                        raise exc

                    # pass preprocessed items on to the corresponding embedders
                    for name, embedder in self.emb_by_key[preprocess_key]:
                        datasets[(name, embedder)].add_item((audio_sr[0], item_ids))

                while True:
                    try:
                        yield out_q.get_nowait()
                        if progress is not None:
                            progress.update()
                    except queue.Empty:
                        break
            for dataset in datasets.values():
                dataset.finalize()
            cf.wait(embed_futures)

            # get results to raise any exceptions that occurred in the embedder
            # futures
            for fut in embed_futures:
                x = fut.result()

            while not out_q.empty():
                yield out_q.get()
                if progress is not None:
                    progress.update()


class QueueDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset wrapper around a queue.  This class expects (audio, id)
    items from the queue and yields them.
    """

    def __init__(self, queue_size=100, name=None):
        self.queue = queue.Queue(maxsize=queue_size)
        self.name = name
        self.end_token = object()

    def add_item(self, item):
        self.queue.put(item)

    def finalize(self):
        self.queue.put(self.end_token)

    def __next__(self):
        item = self.queue.get()
        if item is self.end_token:
            raise StopIteration()
        return item

    def __iter__(self):
        return self


def join_embeddings(embeddings):
    result = defaultdict(list)
    for emb in embeddings:
        for k, v in emb.items():
            result[k].append(v)
    for k, v in result.items():
        vv = np.stack(v)
        result[k] = vv
    return result


# # NOTE: not a drop in replacement for QueueDataset (need to check)
# class QueueSlicingDataset(torch.utils.data.IterableDataset):
#     def __init__(self, win_dur, queue_size=100, name=None):
#         self.win_dur = win_dur
#         self.queue = queue.Queue(maxsize=queue_size)
#         self.name = name

#     def add_item(self, item, song_idx):
#         to_be_put = None
#         for win_idx, audio_sr in enumerate(audio_slicer(item, self.win_dur)):
#             if to_be_put is not None:
#                 self.queue.put(to_be_put)
#             to_be_put = [audio_sr, song_idx, win_idx, 0]
#         if to_be_put is not None:
#             to_be_put[-1] = 1
#             self.queue.put(to_be_put)

#     def finalize(self):
#         self.queue.put((None, None, None, None))

#     def __next__(self):
#         item, song_idx, win_idx, last = self.queue.get()
#         if item is None:
#             raise StopIteration()
#         audio, _ = item
#         return audio, torch.tensor([song_idx, last])

#     def __iter__(self):
#         return self
