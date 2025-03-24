import time
import concurrent.futures as cf
import numpy as np
import tqdm


def handle_futures(futures, discard_inputs, progress=None, size=None):
    """yield from futures as they are completed, updating progress, and removing
    the completed futures from the list. When size is not None, return as soon
    as len(futures) drops below size"""
    for fut in cf.as_completed(futures):
        if size is not None and len(futures) < size:
            break
        if progress:
            progress.update()
            progress.refresh()
        ready_result = fut.result()
        ready_item = futures.pop(fut)
        if discard_inputs:
            yield ready_result
        else:
            ready_item.update(ready_result)
            yield ready_item


def cpu_parallel(
    iterator,
    target,
    n_workers=0,
    desc=None,
    use_threads=False,
    discard_input=True,
    in_buffer_size=None,
    out_buffer_size=None,
):
    if use_threads:
        Executor = cf.ThreadPoolExecutor
    else:
        Executor = cf.ProcessPoolExecutor
    if in_buffer_size is None:
        in_buffer_size = 2 * n_workers
    if out_buffer_size is None:
        out_buffer_size = 2 * n_workers
    with Executor(max_workers=n_workers) as pool:
        tqdm_kwargs = {"desc": desc, "leave": False} if desc else {"disable": True}
        progress = tqdm.tqdm(**tqdm_kwargs)
        progress.total = 0
        futures = {}
        ready = {}
        for item in iterator:
            fut = pool.submit(target, item)
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
        progress.close()
