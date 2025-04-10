import io
import time
import queue
import concurrent.futures as cf

import numpy as np
import tqdm
import torch
from .cpu_parallel import handle_futures


def clone_model(model, device):
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    clone = torch.load(buf, map_location=device, weights_only=False)
    return clone


class GPUWorkerHandler:
    def __init__(self, device_indices=None, thread_pool=None):
        self.pool = thread_pool
        self.available_gpus = queue.Queue()
        if device_indices is None:
            self.device_indices = tuple(range(torch.cuda.device_count()))
        if not self.device_indices:
            raise "No GPUs found, cannot use `gpu_parallel()`"
        for i in self.device_indices:
            self.available_gpus.put(i)
        self.models = {}

    def __len__(self):
        return len(self.device_indices)

    def set_thread_pool(self, pool):
        self.pool = pool

    def clear_thread_pool(self):
        self.pool = None

    def get_current_gpu_i(self, model):
        dev = model.get_device()
        if dev.type != "cuda":
            return None
        if dev.index is None:
            return 0
        return dev.index

    def get_model_on_gpu(self, model, target_gpu_i):
        current_gpu_i = self.get_current_gpu_i(model)
        if current_gpu_i == target_gpu_i:
            return model
        key = (hash(model), target_gpu_i)
        if key not in self.models:
            self.models[key] = clone_model(model, torch.device(f"cuda:{target_gpu_i}"))
        return self.models[key]

    def submit(self, model, args=None, kwargs=None, target=None):
        """`model` should have a `forward()` and `get_device()` method"""
        assert self.pool, "Need to call .set_thread_pool() before submitting jobs"
        # get a gpu (index) from the free pool
        gpu_i = self.available_gpus.get()
        # get a copy of the model on the free gpu
        gpu_i_model = self.get_model_on_gpu(model, gpu_i)
        if target is None:
            target = "forward"
        method = getattr(gpu_i_model, target)
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        future = self.pool.submit(method, *args, **kwargs)
        # put the used gpu (index) back to the free pool
        future.add_done_callback(lambda fut: self.available_gpus.put(gpu_i))
        return future


def gpu_parallel(
    iterator,
    model,
    target=None,
    desc=None,
    discard_input=True,
    device_indices=None,
    gpu_worker_handler=None,
    in_buffer_size=None,
    out_buffer_size=None,
):
    if gpu_worker_handler is None:
        gpu_worker_handler = GPUWorkerHandler(device_indices)
    if in_buffer_size is None:
        in_buffer_size = 2 * len(gpu_worker_handler)
    if out_buffer_size is None:
        out_buffer_size = 2 * len(gpu_worker_handler)

    tqdm_kwargs = {"desc": desc, "leave": False} if desc else {"disable": True}
    progress = tqdm.tqdm(**tqdm_kwargs)
    progress.total = 0

    with cf.ThreadPoolExecutor(max_workers=len(gpu_worker_handler)) as thread_pool:
        gpu_worker_handler.set_thread_pool(thread_pool)
        futures = {}
        ready = {}
        for item in iterator:
            fut = gpu_worker_handler.submit(model, target=target, args=(item,))
            futures[fut] = None if discard_input else item
            progress.total += 1
            progress.refresh()
            if len(futures) >= in_buffer_size:
                to_yield, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for fut in to_yield:
                    ready[fut] = futures.pop(fut)
            yield from handle_futures(ready, discard_input, progress, out_buffer_size)
        yield from handle_futures(ready, discard_input, progress)
        yield from handle_futures(futures, discard_input, progress)
        gpu_worker_handler.clear_thread_pool()
    progress.close()
