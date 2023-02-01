import concurrent.futures as cf
import functools
import gc
import multiprocessing
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

__worker_num = None
__queue = None


def _run_fn(fn: Callable, args) -> Tuple[Optional[Any], bool]:
    kwargs, seedseq = args
    try:
        seed = seedseq.generate_state(1).item()
        # First, set all the seeds.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        result = fn(**kwargs)
        completed = True
    except Exception as e:
        print(f"encountered an error: {e}", flush=True)
        completed = False
        result = None

    global __worker_num, __queue
    if __queue:
        __queue.put(__worker_num)

    torch.cuda.empty_cache()
    gc.collect()

    return result, completed


def _init_proc(q, proc_start, n_workers, n_proc_per_worker):
    worker_num = q.get(timeout=5)
    if sys.platform == "linux":
        s = proc_start
        n = worker_num
        N = n_workers * n_proc_per_worker
        procs = [s + (n * n_proc_per_worker + i) % N for i in range(n_proc_per_worker)]
        # print(
        #     f"acquired worker number {worker_num}, setting processors: {procs}",
        #     flush=True,
        # )

        os.sched_setaffinity(os.getpid(), procs)
    global __worker_num, __queue
    __worker_num = worker_num
    __queue = q


def distributed_eval(
    fn: Callable,
    kwargs_list: List[Dict],
    n_workers: int = 30,
    n_proc_per_worker: int = 2,
    proc_start: int = 0,
    seed: int = 123456,
) -> Tuple[List[Any], List[bool]]:

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(kwargs_list))

    results = []
    completeds = []

    with tqdm.tqdm(total=len(kwargs_list)) as pbar:
        # Case where we aren't doing multiprocessing.
        if n_workers == 0:
            for kwargs, child_seed in zip(kwargs_list, child_seeds):
                result, completed = _run_fn(fn, (kwargs, child_seed))
                results.append(result)
                completeds.append(completed)
                pbar.update(1)
        else:
            # Construct a multiprocessing queue to pass around, containing the worker ids.
            queue = multiprocessing.Queue()
            for i in range(n_workers):
                queue.put(i)

            # Gotta spawn.
            # mp_context = multiprocessing.get_context("spawn")

            with cf.ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_proc,
                initargs=(queue, proc_start, n_workers, n_proc_per_worker),
                # mp_context=mp_context,
            ) as executor:
                futures = {
                    executor.submit(
                        functools.partial(_run_fn, fn),
                        (kwargs, child_seed),
                    ): i
                    for i, (kwargs, child_seed) in enumerate(
                        zip(kwargs_list, child_seeds)
                    )
                }
                result_dict = {}
                completed_dict = {}
                for future in cf.as_completed(futures):
                    result, completed = future.result()
                    result_dict[futures[future]] = result
                    completed_dict[futures[future]] = completed
                    pbar.update(1)

                results = [result_dict[i] for i in range(len(kwargs_list))]
                completeds = [completed_dict[i] for i in range(len(kwargs_list))]
    return results, completeds
