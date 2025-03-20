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
    def __init__(self, n_devs, thread_pool=None):
        self.n_devs = n_devs
        self.pool = thread_pool
        self.available_gpus = queue.Queue()
        for i in range(self.n_devs):
            self.available_gpus.put(i)
        self.models = {}

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
        key = (hash(model), target_gpu_i)
        if key not in self.models:
            if current_gpu_i is None:
                self.models[key] = clone_model(
                    model, torch.device(f"cuda:{target_gpu_i}")
                )
            else:
                self.models[key] = model
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


def iterable_process(
    iterator,
    model,
    n_gpus,
    target=None,
    desc=None,
    discard_input=True,
    gpu_worker_handler=None,
    in_buffer_size=None,
    out_buffer_size=None,
):
    if in_buffer_size is None:
        in_buffer_size = 2 * n_gpus
    if out_buffer_size is None:
        out_buffer_size = 2 * n_gpus

    with cf.ThreadPoolExecutor(max_workers=n_gpus) as thread_pool:
        if gpu_worker_handler is None:
            gpu_worker_handler = GPUWorkerHandler(n_gpus)
        gpu_worker_handler.set_thread_pool(thread_pool)
        tqdm_kwargs = {"desc": desc, "leave": False} if desc else {"disable": True}
        progress = tqdm.tqdm(**tqdm_kwargs)
        progress.total = 0
        futures = {}
        ready = {}
        t_0 = time.perf_counter()
        get_times = []
        for item in iterator:
            t_1 = time.perf_counter()
            get_times.append(t_1 - t_0)
            fut = gpu_worker_handler.submit(model, target=target, args=(item,))
            futures[fut] = None if discard_input else item
            progress.total += 1
            progress.refresh()
            if len(futures) >= in_buffer_size:
                to_yield, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for fut in to_yield:
                    ready[fut] = futures.pop(fut)
            yield from handle_futures(ready, discard_input, progress, out_buffer_size)
            t_0 = time.perf_counter()

        yield from handle_futures(ready, discard_input, progress)
        yield from handle_futures(futures, discard_input, progress)
        progress.close()
        gpu_worker_handler.clear_thread_pool()
    get_times = np.array(get_times)
    print(
        f"{desc} get times: {get_times.min():.4f}--{get_times.max():.4f} (median: {np.median(get_times):.4f}"
    )
