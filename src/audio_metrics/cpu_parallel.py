import concurrent.futures as cf

import tqdm


def iterable_process(
    iterator,
    target,
    n_workers=0,
    desc=None,
    use_threads=False,
    return_inputs=False,
    prefetch_factor=2,
):
    if use_threads:
        Executor = cf.ThreadPoolExecutor
    else:
        Executor = cf.ProcessPoolExecutor
    n_prefetch = prefetch_factor * n_workers
    with Executor(max_workers=n_workers) as pool:
        if desc is None:
            tqdm_kwargs = {"disable": True}
        else:
            tqdm_kwargs = {"desc": desc, "leave": False}
        progress = tqdm.tqdm(**tqdm_kwargs)
        progress.total = 0
        futures = {}
        for item in iterator:
            fut = pool.submit(target, *item)
            if return_inputs:
                futures[fut] = item
            else:
                futures[fut] = None
            progress.total += 1
            progress.refresh()
            if len(futures) >= n_prefetch:
                to_yield, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)

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
