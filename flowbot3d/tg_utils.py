"""This file is a helper for processing the PM dataset in parallel. """
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, runtime_checkable

import torch
import torch_geometric.data as tgd
import tqdm
from torch import multiprocessing


class SingleObjDataset(tgd.InMemoryDataset):
    """A simple class to contain a dataset sampled for a single object."""

    def __init__(self, processed_path):
        super().__init__()
        self.data, self.slices = torch.load(processed_path)


@runtime_checkable
class CanSample(Protocol):
    """This is just a dataset which has the 'get_data' function. This function doesn't
    need to be deterministic, for instance if we want to randomly sample viewpoints."""

    def get_data(self, obj_id: str, *args) -> Any:
        """The"""

    @property
    def processed_dir(self) -> str:
        pass


__dataset: Optional[CanSample] = None


def __can_sample_init(dset_cls, dset_kwargs):
    global __dataset
    __dataset = dset_cls(**dset_kwargs)
    assert isinstance(dset_cls, CanSample)


def __paralel_sample(args: Tuple[int, Tuple, int]):
    task_id = args[0]
    if sys.platform == "linux":
        os.sched_setaffinity(os.getpid(), [task_id % os.cpu_count()])  # type: ignore
    get_data_args = args[1]
    assert len(get_data_args) >= 1 and isinstance(get_data_args[0], str)
    obj_id = get_data_args[0]
    n_repeat = args[2]

    global __dataset
    assert __dataset is not None

    # The output should go in a file that indexes exactly what params
    # were used to generate it.
    prefix = "_".join(get_data_args)
    data_file = f"{prefix}_{n_repeat}.pt"
    data_path = os.path.join(__dataset.processed_dir, data_file)

    if os.path.exists(data_path):
        logging.log(f"{data_path} already exists, skipping...")
        return True

    try:
        # Sample the data.
        data_list = []
        for _ in range(n_repeat):
            data_list.append(__dataset.get_data(*get_data_args))

        # Save it in an in-memory dataset.
        data, slices = tgd.InMemoryDataset.collate(data_list)
        torch.save((data, slices), data_path)

        return True
    except Exception as e:
        logging.error(f"unable to sample get_data({get_data_args}): {e}")
        return False


def parallel_sample(
    dset_cls: Type[CanSample],
    dset_kwargs: Dict[str, Any],
    get_data_args: List[Tuple],
    n_repeat: int = 100,
    n_proc: int = -1,
):
    """Run parallel sampling in a dataset. Assumes that we're operating on a
    partnet-style dataset, where we have a unique object which may be sampled
    many different times (i.e. different point clouds, different camera positions)

    This works across dataset types; the only restriction on the dataset is that
    it implements CanSample.

    Each object will be sampled n_repeat times, and collated into a single file
    for each object, placed in the processed_dir (defined by the dataset). This
    filename is <OBJ_ID>_<N_REPEAT>.pt

    For downstream datasets, you can load each file into a SingleObjDataset,
    and then use torch.utils.data.ConcatDataset to concatentate them.

    Args:
        dset_cls (Type[CanSample]): The constructor for the dataset.
        dset_kwargs (Dict): The kwargs to pass to the dataset constructor.
        get_data_args (List[Tuple]): The arguments to pass to each call to get_data. If there
            are N distinct objects in the dataset, this parameter should have the form:

            [(OBJ_ID, arg_1, ...), ...]

            where the length of the list is N, and the first entry in each tuple is the
            object ID (a string). These arguments will be passed to get_data via *get_data_args.
        n_repeat (int, optional): Number of times to call get_data on the same args
            (i.e. when get_data is nondeterministic, to sample viewpoints, etc.). Defaults to 100.
        n_proc (int, optional): Number of processes for multiprocessing.
            If 0, multiprocessing won't be used at all (good for debugging serially).
            If -1, uses as many CPUs as are available.
            Defaults to -1.

    Raises:
        ValueError: If any sampling fails, this method will raise an exception.
    """
    if n_proc == -1:
        n_proc = os.cpu_count()  # type: ignore

    # task_number, args, n_repeat
    ps_args = list(
        zip(range(len(get_data_args)), get_data_args, [n_repeat] * len(get_data_args))
    )

    # Debugging, means in-memory.
    if n_proc == 0:
        print("sampling with no workers, debug")
        # Initialize the dataset.
        __can_sample_init(dset_cls, dset_kwargs)

        res = [__paralel_sample(args) for args in tqdm.tqdm(ps_args)]

    else:
        multiprocessing.set_start_method("spawn")
        # Create a multiprocessing pool, and initialize it with a global dataset object
        # that can be used to sample things.
        pool = multiprocessing.Pool(n_proc, __can_sample_init, (dset_cls, dset_kwargs))

        # Perform the parallel mapping.
        res = list(
            tqdm.tqdm(pool.imap(__paralel_sample, ps_args), total=len(get_data_args))
        )

    if not all(res):
        raise ValueError("Sampling failed, please debug.")
