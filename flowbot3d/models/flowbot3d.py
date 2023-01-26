import typing
from dataclasses import dataclass
from typing import Dict, Union

import plotly.graph_objects as go
import pytorch_lightning as pl
import rpad.pyg.nets.dgcnn as pnd
import rpad.pyg.nets.pointnet2 as pnp
import rpad.visualize_3d.plots as v3p
import torch
import torch.optim as opt
import torch_geometric.data as tgd
from plotly.subplots import make_subplots
from torch_geometric.data import Batch, Data

DenseParams = Union[pnp.PN2DenseParams, pnd.DGCNNDenseParams]


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
    mask: torch.Tensor,
    n_nodes: torch.Tensor,
    use_mask=False,
) -> torch.Tensor:
    """Maniskill loss.

    Args:
        f_pred (torch.Tensor): Predicted flow.
        f_target (torch.Tensor): Target flow.
        mask (torch.Tensor): only mask
        n_nodes (torch.Tensor): A list describing the number of nodes
        use_mask: Whether or not to compute loss over all points, or just some.

    Returns:
        Loss
    """
    weights = (1 / n_nodes).repeat_interleave(n_nodes)

    if use_mask:
        f_pred = f_pred[mask]
        f_target = f_target[mask]
        weights = weights[mask]

    # Flow loss, per-point.
    raw_se = ((f_pred - f_target) ** 2).sum(dim=1)
    # weight each PC equally in the sum.
    l_se = (raw_se * weights).sum()

    # Full loss.
    loss: torch.Tensor = l_se / len(n_nodes)

    return loss


def create_flownet(
    in_channels=0, out_channels=3, p: DenseParams = pnp.PN2DenseParams()
) -> Union[pnp.PN2Dense, pnd.DGCNNDense]:
    if isinstance(p, pnp.PN2DenseParams):
        return pnp.PN2Dense(in_channels, out_channels, p)
    elif isinstance(p, pnd.DGCNNDenseParams):
        return pnd.DGCNNDense(in_channels, out_channels, p)
    else:
        raise ValueError(f"invalid model type: {type(p)}")


@dataclass
class ArtFlowNetParams:
    net: DenseParams = pnp.PN2DenseParams()
    mask_output_flow: bool = False


class ArtFlowNet(pl.LightningModule):
    def __init__(self, p: ArtFlowNetParams = ArtFlowNetParams(), lr=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.params = p
        self.lr = lr

        # The network is just this dense backbone.
        self.flownet = create_flownet(0, 3, p.net)

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
        # data = Data(pos=xyz, x=mask)
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        self.eval()
        with torch.no_grad():
            flow = self.forward(batch)
        return flow

    def forward(self, data) -> torch.Tensor:  # type: ignore
        flow = typing.cast(torch.Tensor, self.flownet(data))
        # Since we're given the mask at input, we can zero it out.
        if self.params.mask_output_flow:
            flow[~data.mask] = 0.0

        return flow

    def _step(self, batch: tgd.Batch, mode):

        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore

        f_ix = batch.mask.bool()
        # batch.x = batch.mask
        f_pred = self(batch)
        f_target = batch.flow
        loss = artflownet_loss(f_pred, f_target, batch.mask, n_nodes)
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
