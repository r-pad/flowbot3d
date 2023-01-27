import math
import random
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rpad.partnet_mobility_utils.dataset as rpd
import torch
import torch_geometric.loader as tgl
import tqdm
import typer
from rpad.pyg.dataset import CachedByKeyDataset

from flowbot3d.models.flowbot3d import ArtFlowNet, flow_metrics
from flowbot3d.tg_dataset import Flowbot3DTGDataset


def make_eval_dataset(
    pm_root: Path, dataset: str, randomize_camera: bool, n_proc: int
) -> CachedByKeyDataset:
    keys = {
        "umpnet-train-train": rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS,
        "umpnet-train-test": rpd.UMPNET_TRAIN_TEST_OBJ_IDS,
        "umpnet-test": rpd.UMPNET_TEST_OBJ_IDS,
    }[dataset]

    return CachedByKeyDataset(
        dset_cls=Flowbot3DTGDataset,
        dset_kwargs=dict(
            root=pm_root / "raw",
            split=dataset,
            randomize_camera=randomize_camera,
        ),
        data_keys=keys,
        root=pm_root,
        processed_dirname=Flowbot3DTGDataset.get_processed_dir(
            True,
            randomize_camera,
        ),
        n_repeat=1,
        n_proc=n_proc,
        seed=42,
    )


def make_model(model: str, ckpt_path: Union[str, Path]) -> pl.LightningModule:
    if model == "flowbot":
        return ArtFlowNet.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"invalid model {model}")


@torch.no_grad()
def run_eval(dset, model, batch_size=64) -> pd.DataFrame:

    all_objs = (
        rpd.UMPNET_TRAIN_TRAIN_OBJS + rpd.UMPNET_TRAIN_TEST_OBJS + rpd.UMPNET_TEST_OBJS
    )
    id_to_obj_class = {obj_id: obj_class for obj_id, obj_class in all_objs}

    # Batch predictions.
    loader = tgl.DataLoader(dset, batch_size, shuffle=False, num_workers=0)

    metrics = []

    for batch in tqdm.tqdm(loader, total=int(math.ceil(len(dset) / batch_size))):
        preds = model(batch.to(model.device))

        st = 0
        for data in batch.to_data_list():
            f_pred = preds[st : st + data.num_nodes]
            f_ix = data.mask.bool()
            f_target = data.flow

            rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

            metrics.append(
                {
                    "id": data.id,
                    "obj_class": id_to_obj_class[data.id],
                    "metrics": {
                        "rmse": rmse.cpu().item(),
                        "cos_dist": cos_dist.cpu().item(),
                        "mag_error": mag_error.cpu().item(),
                    },
                }
            )

            st += data.num_nodes

    rows = [
        (
            m["id"],
            m["obj_class"],
            m["metrics"]["rmse"],
            m["metrics"]["cos_dist"],
            m["metrics"]["mag_error"],
        )
        for m in metrics
    ]
    raw_df = pd.DataFrame(
        rows, columns=["id", "category", "rmse", "cos_dist", "mag_error"]
    )
    df = raw_df.groupby("category").mean(numeric_only=True)
    df.loc["unweighted_mean"] = raw_df.mean(numeric_only=True)
    df.loc["class_mean"] = df.mean()

    return df.T


def main(
    pm_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    model_type: str = "flowbot",
    model_name: str = typer.Option(...),
    ckpt_path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    batch_size: int = 64,
    randomize_camera: bool = False,
    n_proc: int = 50,
    device: str = "cuda:0",
    out_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True),
):
    out_dir = out_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ["umpnet-train-train", "umpnet-train-test", "umpnet-test"]:
        out_file = out_dir / f"{dataset}.csv"
        if out_file.exists():
            raise ValueError(f"{out_file} already exists...")

        # Make it deterministic.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        dset = make_eval_dataset(pm_root, dataset, randomize_camera, n_proc)

        model = make_model(model_type, ckpt_path).to(device)
        model.eval()

        df = run_eval(dset, model, batch_size)

        print(dataset.upper())
        print(df)

        df.to_csv(out_file, float_format="%.3f")


if __name__ == "__main__":
    typer.run(main)
