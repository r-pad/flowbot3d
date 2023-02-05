import abc
import time
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import rpad.partnet_mobility_utils.dataset as rpd
import torch
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import typer
from rpad.pyg.dataset import CachedByKeyDataset

from flowbot3d.models.flowbot3d import ArtFlowNet, ArtFlowNetParams
from flowbot3d.models.umpnet_di import UMPNet, UMPNetParams
from flowbot3d.tg_dataset import Flowbot3DTGDataset


def create_model(
    model: str, lr: float, mask_input_channel: bool = True
) -> pl.LightningModule:
    if model == "flowbot":
        return ArtFlowNet(
            p=ArtFlowNetParams(mask_input_channel=mask_input_channel),
            lr=lr,
        )
    elif model == "umpnet":
        return UMPNet(params=UMPNetParams(lr=lr))
    else:
        raise ValueError(f"bad model: {model}")


def create_datasets(
    root: Path,
    dataset: str,
    n_proc=-1,
    randomize_camera: bool = True,
) -> Tuple[tgd.Dataset, tgd.Dataset, tgd.Dataset]:
    if dataset == "umpnet":
        train_dset = CachedByKeyDataset(
            dset_cls=Flowbot3DTGDataset,
            dset_kwargs=dict(
                root=root / "raw",
                split="umpnet-train-train",
                randomize_camera=randomize_camera,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS,
            root=root,
            processed_dirname=Flowbot3DTGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=100,
            n_proc=n_proc,
        )

        test_dset = CachedByKeyDataset(
            dset_cls=Flowbot3DTGDataset,
            dset_kwargs=dict(
                root=root / "raw",
                split="umpnet-train-test",
                randomize_camera=randomize_camera,
            ),
            data_keys=rpd.UMPNET_TRAIN_TEST_OBJ_IDS,
            root=root,
            processed_dirname=Flowbot3DTGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=1,
            n_proc=n_proc,
        )

        unseen_dset = CachedByKeyDataset(
            dset_cls=Flowbot3DTGDataset,
            dset_kwargs=dict(
                root=root / "raw",
                split="umpnet-test",
                randomize_camera=randomize_camera,
            ),
            data_keys=rpd.UMPNET_TEST_OBJ_IDS,
            root=root,
            processed_dirname=Flowbot3DTGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=1,
            n_proc=n_proc,
        )
    elif dataset == "single":
        dset = CachedByKeyDataset(
            dset_cls=Flowbot3DTGDataset,
            dset_kwargs=dict(
                root=root / "raw",
                split=["7179"],
                randomize_camera=randomize_camera,
            ),
            data_keys=["7179"],
            root=root,
            processed_dirname=Flowbot3DTGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=1,
            n_proc=n_proc,
        )
        train_dset = dset
        test_dset = dset
        unseen_dset = dset

    else:

        raise ValueError(f"bad dataset: {dataset}")

    return train_dset, test_dset, unseen_dset


class CanMakePlots(Protocol):
    @staticmethod
    @abc.abstractmethod
    def make_plots(preds, batch: tgd.Batch):
        pass


class LightningModuleWithPlots(pl.LightningModule, CanMakePlots):
    pass


class WandBPlotlyCallback(plc.Callback):
    def __init__(
        self, train_dset, val_dset, unseen_dset=None, eval_per_n_epoch: int = 1
    ):
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.unseen_dset = unseen_dset
        self.eval_per_n_epoch = eval_per_n_epoch

    @staticmethod
    def eval_log_random_sample(
        trainer: pl.Trainer,
        pl_module: LightningModuleWithPlots,
        dset,
        prefix: Literal["train", "val", "unseen"],
    ):
        data = dset[np.random.randint(0, len(dset))]
        data = tgd.Batch.from_data_list([data]).to(pl_module.device)

        with torch.no_grad():
            pl_module.eval()
            preds = pl_module(data)

            if isinstance(preds, tuple):
                preds = (pred.cpu() for pred in preds)
            else:
                preds = preds.cpu()

        plots = pl_module.make_plots(preds, data.cpu())

        assert trainer.logger is not None and isinstance(
            trainer.logger, plog.WandbLogger
        )
        trainer.logger.experiment.log(
            {
                **{f"{prefix}/{plot_name}": plot for plot_name, plot in plots.items()},
                "global_step": trainer.global_step,
            },
            step=trainer.global_step,
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):  # type: ignore
        if pl_module.current_epoch % self.eval_per_n_epoch == 0:
            self.eval_log_random_sample(trainer, pl_module, self.train_dset, "train")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):  # type: ignore
        if pl_module.current_epoch % self.eval_per_n_epoch == 0:
            self.eval_log_random_sample(trainer, pl_module, self.val_dset, "val")
            if self.unseen_dset is not None:
                self.eval_log_random_sample(
                    trainer, pl_module, self.unseen_dset, "unseen"
                )


def train(
    pm_root: Path = typer.Option(None, exists=True, file_okay=False, dir_okay=True),
    dataset: str = "umpnet",
    model_type: str = "flowbot",
    batch_size: int = 64,
    lr: float = 1e-3,
    mask_input_channel: bool = True,
    randomize_camera: bool = True,
    epochs: int = 100,
    n_proc: int = 100,
    log_every_n_steps: int = 5,
    check_val_every_n_epoch: int = 1,
    wandb: bool = False,
    seed: int = 42,
):
    # Seed!
    pl.seed_everything(seed, workers=True)

    # Create the datasets.
    train_dset, test_dset, unseen_dset = create_datasets(
        pm_root,
        dataset,
        n_proc,
        randomize_camera=randomize_camera,
    )

    train_loader = tgl.DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    test_loader = tgl.DataLoader(test_dset, batch_size, shuffle=False, num_workers=0)
    unseen_loader = tgl.DataLoader(
        unseen_dset, batch_size, shuffle=False, num_workers=0
    )

    # Create the model.
    model = create_model(model_type, lr=lr, mask_input_channel=mask_input_channel)

    # Create some logging.
    logger: Union[plog.WandbLogger, Literal[False]]
    cbs: Optional[List[plc.Callback]]

    if wandb:
        project = "flowbot3d"
        logger = plog.WandbLogger(
            project=project,
            config={
                "dataset": dataset,
                "model": model_type,
                "batch_size": batch_size,
                "lr": lr,
                "mask_input_channel": mask_input_channel,
                "randomize_camera": randomize_camera,
                "seed": seed,
            },
        )

        run_name = logger.experiment.name
        checkpoint_dir = f"checkpoints/wandb/{run_name}"
        cbs = [
            plc.ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=1),
            WandBPlotlyCallback(
                train_dset=train_dset,
                val_dset=test_dset,
                unseen_dset=unseen_dset,
                eval_per_n_epoch=check_val_every_n_epoch,
            ),
        ]
    else:
        logger = False
        run_name = time.strftime("%Y_%m_%d-%H_%M_%S")
        checkpoint_dir = f"checkpoints/no-wandb/{run_name}"
        cbs = [plc.ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=1)]

    # Create the trainer, which we'll train on only 1 gpu.
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=cbs,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        max_epochs=epochs,
        deterministic="warn",
    )

    # Run training.
    trainer.fit(model, train_loader, [test_loader, unseen_loader])

    # Clean up training.
    if wandb and logger:
        logger.experiment.finish()


if __name__ == "__main__":
    typer.run(train)
