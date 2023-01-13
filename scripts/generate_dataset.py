import os

import typer
from rpad.pyg.dataset import CachedByKeyDataset

from flowbot3d.dataset import Flowbot3DDataset
from flowbot3d.tg_dataset import Flowbot3DTGDataset


def main(
    root: str = os.path.expanduser("~/datasets/partnet-mobility"),
    split: str = "umpnet-train-train",
    randomize_joints: bool = True,
    randomize_camera: bool = True,
    n_points: int = 1000,
    n_repeat: int = 100,
    n_proc: int = os.cpu_count() - 10,
    seed: int = 12345,
):
    kwargs = dict(
        root=root + "/raw",
        split=split,
        randomize_joints=randomize_joints,
        randomize_camera=randomize_camera,
        n_points=n_points,
    )
    raw_dset = Flowbot3DDataset(**kwargs)

    # We could get the ids some other way but we choose not to.
    # Make it a list of tuples.
    data_keys = list(zip(raw_dset._dataset._ids))

    dset = CachedByKeyDataset(
        dset_cls=Flowbot3DTGDataset,
        dset_kwargs=kwargs,
        data_keys=data_keys,
        root=root,
        processed_dirname=Flowbot3DTGDataset.get_processed_dir(
            randomize_joints, randomize_camera
        ),
        n_repeat=n_repeat,
        n_proc=n_proc,
        seed=seed,
    )

    dset[0]


if __name__ == "__main__":
    typer.run(main)
