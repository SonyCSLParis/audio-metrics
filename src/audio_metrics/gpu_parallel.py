import io
import queue
import concurrent.futures as cf

import tqdm
import torch


def clone_model(model, device):
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    clone = torch.load(buf, map_location=device, weights_only=False)
    return clone


class GPUWorkerHandler:
    def __init__(self, n_devs, thread_pool):
        self.n_devs = n_devs
        self.pool = thread_pool
        self.available_gpus = queue.Queue()
        for i in range(self.n_devs):
            self.available_gpus.put(i)
        self.models = {}

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
    return_inputs=False,
    prefetch_factor=2,
):
    n_prefetch = prefetch_factor * n_gpus
    with cf.ThreadPoolExecutor(max_workers=n_gpus) as thread_pool:
        gpu_worker_handler = GPUWorkerHandler(n_gpus, thread_pool)

        if desc is None:
            tqdm_kwargs = {"disable": True}
        else:
            tqdm_kwargs = {"desc": desc, "leave": False}
        progress = tqdm.tqdm(**tqdm_kwargs)
        progress.total = 0
        futures = {}
        for item in iterator:
            fut = gpu_worker_handler.submit(model, target=target, args=item)
            futures[fut] = item
            progress.total += 1
            progress.refresh()

            # until we are done prefetching, don't wait for futures to complete,
            # but yield any completed futures
            timeout = 0 if len(futures) < n_prefetch else None
            to_yield, _ = cf.wait(
                futures, timeout=timeout, return_when=cf.FIRST_COMPLETED
            )

            for fut in to_yield:
                progress.update()
                progress.refresh()
                ready_result = fut.result()
                if return_inputs:
                    ready_item = futures[fut]
                    yield ready_result, ready_item
                else:
                    yield ready_result
                del futures[fut]

        for fut in cf.as_completed(futures):
            progress.update()
            progress.refresh()
            ready_result = fut.result()
            if return_inputs:
                ready_item = futures[fut]
                yield ready_result, ready_item
            else:
                yield ready_result
            del futures[fut]
