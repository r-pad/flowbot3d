import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from rpad.pyg.dataset import CachedByKeyDataset

from flowbot3d.dataset import Flowbot3DDataset
from flowbot3d.tg_dataset import Flowbot3DTGDataset

TESTDATA_DIR = Path(__file__).parent / "testdata"


def test_simple_dataset():
    dset = Flowbot3DDataset(
        root=TESTDATA_DIR,
        split=["7179"],
        randomize_joints=True,
        randomize_camera=True,
        n_points=1000,
    )

    data_dict1 = dset.get_data("7179", seed=12345)
    data_dict2 = dset.get_data("7179", seed=12345)
    data_dict3 = dset.get_data("7179", seed=54321)

    assert np.array_equal(data_dict1["pos"], data_dict2["pos"])
    assert not np.array_equal(data_dict1["pos"], data_dict3["pos"])


def test_simple_tg_dataset():
    dset = Flowbot3DTGDataset(
        root=TESTDATA_DIR,
        split=["7179"],
        randomize_joints=True,
        randomize_camera=True,
        n_points=1000,
    )

    data1 = dset.get_data("7179", seed=12345)
    data2 = dset.get_data("7179", seed=12345)
    data3 = dset.get_data("7179", seed=54321)

    assert torch.equal(data1.pos, data2.pos)
    assert not torch.equal(data1.pos, data3.pos)


def test_parallel_sampling():
    with tempfile.TemporaryDirectory() as tmpdir:
        dset = CachedByKeyDataset(
            dset_cls=Flowbot3DTGDataset,
            dset_kwargs=dict(
                root=TESTDATA_DIR,
                split=["7179", "100809"],
                randomize_joints=True,
                randomize_camera=True,
                n_points=1000,
            ),
            data_keys=[("7179",), ("100809",)],
            root=tmpdir,
            processed_dirname="processed_test",
            n_repeat=5,
            n_proc=2,
            seed=12345,
        )

        # Try to get the thing.
        data0 = dset[0]

        assert len(os.listdir(os.path.join(tmpdir, "processed_test"))) == 4
        for id in ["7179", "100809"]:
            assert os.path.exists(os.path.join(tmpdir, "processed_test", f"{id}_5.pt"))
