import typing
from dataclasses import dataclass
from typing import Dict

import plotly.graph_objects as go
import pytorch_lightning as pl
import rpad.pyg.nets.pointnet2 as pnp
import rpad.visualize_3d.plots as v3p
import torch
import torch.optim as opt
import torch_geometric.data as tgd
from plotly.subplots import make_subplots
from torch_geometric.data import Batch, Data


def flow_metrics(pred_flow, gt_flow):
    with torch.no_grad():
        # RMSE
        rmse = (pred_flow - gt_flow).norm(p=2, dim=1).mean()

        # Cosine similarity, normalized.
        nonzero_gt_flowixs = torch.where(gt_flow.norm(dim=1) != 0.0)
        gt_flow_nz = gt_flow[nonzero_gt_flowixs]
        pred_flow_nz = pred_flow[nonzero_gt_flowixs]
        cos_dist = torch.cosine_similarity(pred_flow_nz, gt_flow_nz, dim=1).mean()

        # Magnitude
        mag_error = (pred_flow.norm(p=2, dim=1) - gt_flow.norm(p=2, dim=1)).abs().mean()
    return rmse, cos_dist, mag_error


def artflownet_loss(
    f_pred: torch.Tensor,
    f_target: torch.Tensor,
    n_nodes: torch.Tensor,
) -> torch.Tensor:
    """Maniskill loss.

    Args:
        f_pred (torch.Tensor): Predicted flow.
        f_target (torch.Tensor): Target flow.
        n_nodes (torch.Tensor): A list describing the number of nodes

    Returns:
        Loss
    """

    # Flow loss, per-point.
    raw_se = ((f_pred - f_target) ** 2).sum(dim=1)

    # Compute a per-point weighting, so that each point clouQd in a batch
    # is weighted the same. i.e. so that if one point cloud has 1000 points,
    # and another point cloud has 15 points, the loss from each point cloud
    # is equal (by weighting each point's contribution to the loss).
    weights = (1 / n_nodes).repeat_interleave(n_nodes)
    l_se = (raw_se * weights).sum()

    # Full loss, aberaged across the batch.
    loss: torch.Tensor = l_se / len(n_nodes)

    return loss


@dataclass
class ArtFlowNetParams:
    net: pnp.PN2DenseParams = pnp.PN2DenseParams()
    mask_input_channel: bool = False


class ArtFlowNet(pl.LightningModule):
    def __init__(
        self,
        p: ArtFlowNetParams = ArtFlowNetParams(),
        lr=0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.params = p
        self.lr = lr

        # The network is just this dense backbone.
        mask_channel = 1 if p.mask_input_channel else 0
        self.flownet = pnp.PN2Dense(in_channels=mask_channel, out_channels=3, p=p.net)

    def predict(self, xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict the flow for a single object. The point cloud should
        come straight from the maniskill processed observation function.

        Args:
            xyz (torch.Tensor): Nx3 pointcloud
            mask (torch.Tensor): Nx1 mask of the part that will move.

        Returns:
            torch.Tensor: Nx3 dense flow prediction
        """
        assert len(xyz) == len(mask)
        assert len(xyz.shape) == 2
        assert len(mask.shape) == 1

        data = Data(pos=xyz, mask=mask)
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        self.eval()
        with torch.no_grad():
            flow = self.forward(batch)
        return flow

    def forward(self, data) -> torch.Tensor:  # type: ignore
        # Maybe add the mask as an input to the network.
        if self.params.mask_input_channel:
            data.x = data.mask.reshape(len(data.mask), 1)

        # Run the model.
        flow = typing.cast(torch.Tensor, self.flownet(data))

        return flow

    def _step(self, batch: tgd.Batch, mode):
        # Make a prediction.
        f_pred = self(batch)

        # Compute the loss.
        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        f_target = batch.flow
        loss = artflownet_loss(f_pred, f_target, n_nodes)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

        self.log_dict(
            {
                f"{mode}/loss": loss,
                f"{mode}/rmse": rmse,
                f"{mode}/cosine_similarity": cos_dist,
                f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )

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
        obj_id = batch.id
        pos = batch.pos.numpy()
        mask = batch.mask.numpy()
        f_target = batch.flow
        f_pred = preds

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene"}, {"type": "table"}],
                [{"type": "scene"}, {"type": "scene"}],
            ],
            subplot_titles=(
                "input data",
                "N/A",
                "target flow",
                "pred flow",
            ),
        )

        # Parent/child plot.
        labelmap = {0: "unselected", 1: "part"}
        labels = torch.zeros(len(pos)).int()
        labels[mask == 1.0] = 1
        fig.add_traces(v3p._segmentation_traces(pos, labels, labelmap, "scene1"))

        fig.update_layout(
            scene1=v3p._3d_scene(pos),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=1.0, y=0.75),
        )

        # Connectedness table.
        fig.append_trace(
            go.Table(
                header=dict(values=["IGNORE", "IGNORE"]),
                cells=dict(values=[[1.0], [1.0]]),
            ),
            row=1,
            col=2,
        )

        # normalize the flow for visualization.
        n_f_gt = (f_target / f_target.norm(dim=1).max()).numpy()
        n_f_pred = (f_pred / f_target.norm(dim=1).max()).numpy()

        # GT flow.
        fig.add_trace(v3p.pointcloud(pos, downsample=1, scene="scene2"), row=2, col=1)
        fig.add_traces(v3p._flow_traces(pos, n_f_gt, scene="scene2"), rows=2, cols=1)
        fig.update_layout(scene2=v3p._3d_scene(pos))

        # Predicted flow.
        fig.add_trace(v3p.pointcloud(pos, downsample=1, scene="scene3"), row=2, col=2)
        fig.add_traces(v3p._flow_traces(pos, n_f_pred, scene="scene3"), rows=2, cols=2)
        fig.update_layout(scene3=v3p._3d_scene(pos))

        fig.update_layout(title=f"Object {obj_id}")

        return {"artflownet_plot": fig}
