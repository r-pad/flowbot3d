from typing import List, Optional, Tuple

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import typer

from flowbot3d.models.flowbot3d import ArtFlowNet
from flowbot3d.tg_dataset import Flowbot3DTGDataset


def create_model(model: str) -> pl.LightningModule:
    if model == "flowbot":
        return ArtFlowNet()
    else:
        raise ValueError(f"bad model: {model}")


def create_datasets(
    root: str, dataset: str, n_proc=-1
) -> Tuple[tgd.Dataset, tgd.Dataset, tgd.Dataset]:
    if dataset == "umpnet":
        train_dset = Flowbot3DTGDataset(
            root, "umpnet-train-train", n_repeat=50, n_proc=n_proc
        )
        test_dset = Flowbot3DTGDataset(
            root, "umpnet-train-test", n_repeat=1, n_proc=n_proc
        )
        unseen_dset = Flowbot3DTGDataset(root, "umpnet-test", n_repeat=1, n_proc=n_proc)
    else:
        raise ValueError(f"bad dataset: {dataset}")

    return train_dset, test_dset, unseen_dset


def train(
    pm_root: str,
    dataset: str = "umpnet",
    model_type: str = "flowbot",
    batch_size: int = 64,
    epochs: int = 100,
    n_proc: int = 100,
    wandb: bool = False,
):
    # Create the datasets.
    train_dset, test_dset, unseen_dset = create_datasets(pm_root, dataset, n_proc)
    train_loader = tgl.DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    test_loader = tgl.DataLoader(test_dset, batch_size, shuffle=False, num_workers=0)
    unseen_loader = tgl.DataLoader(
        unseen_dset, batch_size, shuffle=False, num_workers=0
    )

    # Create the model.
    model = create_model(model_type)

    # Create some logging.
    logger: Optional[plog.WandbLogger]
    cbs: Optional[List[plc.Callback]]
    if wandb:
        raise NotImplementedError
    else:
        logger = None
        cbs = None

    # Create the trainer, which we'll train on only 1 gpu.
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=cbs,
        log_every_n_steps=5,
        max_epochs=epochs,
    )

    # Run training.
    trainer.fit(model, train_loader, [test_loader, unseen_loader])

    if wandb and logger is not None:
        logger.experiment.finish()


if __name__ == "__main__":
    typer.run(train)
