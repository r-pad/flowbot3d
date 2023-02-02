import multiprocessing

import numpy as np

from flowbot3d.eval.utils import distributed_eval


def __return_seeds(x):
    return x**2


def test_distributed_eval():
    multiprocessing.set_start_method("spawn")
    args = [{"x": i} for i in range(10)]

    results, completeds = distributed_eval(
        __return_seeds,
        args,
        n_workers=2,
        n_proc_per_worker=1,
    )

    assert np.array_equal(results, [i**2 for i in range(10)])