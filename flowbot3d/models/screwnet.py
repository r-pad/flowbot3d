from dataclasses import dataclass
from typing import Dict, Tuple

import plotly.graph_objects as go
import pytorch_lightning as pl
import rpad.pyg.nets.pointnet2 as pnp
import rpad.visualize_3d.plots as rvpl
import rpad.visualize_3d.primitives as rvpr
import torch
import torch.nn.functional as F
import torch.optim as opt
import torch_geometric.data as tgd


def _articulation_trace(
    origin, direction, color="red", joint_type=0, scene="scene", name="joint"
):
    x, y, z = origin.tolist()
    u, v, w = direction.tolist()

    return rvpr.vector(x, y, z, u, v, w, color, scene=scene, name=name)


@dataclass
class ScrewNetParams:
    encoder_params: pnp.PN2EncoderParams = pnp.PN2EncoderParams()
    embedding_dim: int = 1024
    lin1_dim: int = 128
    lin2_dim: int = 128
    ignore_prismatic_origin: bool = True


class ScrewNet(pl.LightningModule):
    def __init__(self, p: ScrewNetParams = ScrewNetParams(), lr=0.0001) -> None:
        super().__init__()
        self.params = p
        self.pn_encoder = pnp.PN2Encoder(
            in_dim=1, out_dim=p.embedding_dim, p=p.encoder_params
        )
        self.lin1 = torch.nn.Linear(p.embedding_dim, p.lin1_dim)
        self.lin2 = torch.nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = torch.nn.Linear(p.lin2_dim, 6)
        self.lr = lr

    def forward(self, data: tgd.Batch) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        data.x = data.mask
        x = self.pn_encoder(data)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x[:, :3], x[:, 3:]

    def _step(self, batch: tgd.Batch, mode):
        pred_axis, pred_origin = self(batch)

        if self.params.ignore_prismatic_origin:
            # Only apply an origin loss to revolute joints.
            is_rev = torch.as_tensor([jt == "revolute" for jt in batch.joint_type], dtype=torch.float32).to(self.device).unsqueeze(-1)  # type: ignore
            loss = F.mse_loss(batch.axis, pred_axis) + F.mse_loss(
                batch.origin * is_rev, pred_origin * is_rev
            )
        else:
            loss = F.mse_loss(batch.axis, pred_axis) + F.mse_loss(
                batch.origin, pred_origin
            )
        self.log_dict({f"{mode}/loss": loss}, add_dataloader_idx=False)

        return loss

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        return self._step(batch, "train")

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()
        name = "val" if dataloader_idx == 0 else "unseen"
        return self._step(batch, name)

    def configure_optimizers(self):
        return opt.Adam(params=self.parameters(), lr=self.lr)

    @staticmethod
    def make_plots(preds, batch: tgd.Batch) -> Dict[str, go.Figure]:
        pos = batch.pos
        mask = batch.mask
        pred_axis, pred_origin = preds
        gt_origin = batch.origin.squeeze()
        gt_axis = batch.axis.squeeze()

        fig = go.Figure()

        traces = rvpl._segmentation_traces(pos, mask.int().squeeze(), scene="scene1")
        fig.add_traces(traces)

        traces = _articulation_trace(
            torch.as_tensor(pred_origin.squeeze()),
            torch.as_tensor(pred_axis.squeeze()),
            scene="scene1",
            name="pred",
            color="red",
        )
        fig.add_traces(traces)
        traces = _articulation_trace(
            torch.as_tensor(gt_origin),
            torch.as_tensor(gt_axis),
            scene="scene1",
            name="gt",
            color="blue",
        )
        fig.add_traces(traces)

        fig.update_layout(
            scene1=rvpl._3d_scene(
                torch.cat(
                    [pos, pred_origin.reshape(-1, 3), gt_origin.reshape(-1, 3)], dim=0
                )
            )
        )
        return {"screwnet_plot": fig}
