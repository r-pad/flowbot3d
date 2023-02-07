import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import pytorch_lightning as pl
import pytorch_lightning.loggers as plog
import torch
import torch_geometric.loader as tgl
from rpad.pyg.dataset import CachedByKeyDataset

from flowbot3d.models.artflownet import ArtFlowNet
from flowbot3d.tg_dataset import Flowbot3DTGDataset

TESTDATA_DIR = Path(__file__).parent / "testdata"


class MetricsLogger(plog.Logger):
    """Logger to collect the loss during training, to compare."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_sequence: List[float] = []

    @property
    def name(self):
        return "MetricsLogger"

    @property
    def version(self):
        return "0.1"

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step):
        self.loss_sequence.append(metrics["train/loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU available")
def test_deterministic_training():
    # Generate a new dataset on the fly. Very nice!
    with tempfile.TemporaryDirectory() as tmpdir:
        dset = CachedByKeyDataset(
            dset_cls=Flowbot3DTGDataset,
            dset_kwargs=dict(
                root=TESTDATA_DIR,
                split=["7179", "100809"],
                randomize_joints=True,
                randomize_camera=True,
                n_points=100,
            ),
            data_keys=[("7179",), ("100809",)],
            root=tmpdir,
            processed_dirname="processed_test",
            n_repeat=5,
            n_proc=0,
            seed=12345,
        )

    seed = 42

    # Run 10 steps of training, twice. Make sure that the loss sequence is identical!
    # This 'roughly' ensures reproducibility, even though there's a scatter in there
    # somewhere which might not be fully reproducible (aka why we put 'warn' instead of True).
    seqs = []
    for _ in range(2):
        pl.seed_everything(seed, workers=True)

        # Set up training.
        train_loader = tgl.DataLoader(dset, batch_size=1, shuffle=True, num_workers=0)
        model = ArtFlowNet(lr=1e-3)
        logger = MetricsLogger()
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=logger,
            log_every_n_steps=1,
            max_epochs=1,
            deterministic="warn",
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )

        # Run training.
        trainer.fit(model, train_loader)

        seqs.append(logger.loss_sequence)

    assert np.array_equal(np.array(seqs[0]), np.array(seqs[1]))
