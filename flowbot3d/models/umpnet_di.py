from dataclasses import dataclass
from typing import Dict

import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch_geometric.data as tgd
from rpad.pyg.nets.pointnet2 import PN2Encoder, PN2EncoderParams
from rpad.visualize_3d.plots import _3d_scene, _flow_traces
from rpad.visualize_3d.primitives import pointcloud


@dataclass
class DirectionNetParams:
    encoder: PN2EncoderParams = PN2EncoderParams()
    encoder_outdim: int = 1024

    # Dimensions of the final 2 linear layers.
    lin1_dim: int = 128
    lin2_dim: int = 128


class DirectionNet(nn.Module):
    def __init__(self, p: DirectionNetParams = DirectionNetParams()):
        super().__init__()
        self.p = p

        self.pc_encoder = PN2Encoder(out_dim=p.encoder_outdim, p=p.encoder)

        # Final linear layers at the output.
        self.lin1 = nn.Linear(p.encoder_outdim + 3, p.lin1_dim)
        self.lin2 = nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = nn.Linear(p.lin2_dim, 1)

    def score(self, x: torch.Tensor, actions: torch.Tensor):
        # BATCH_SIZE X N_SAMPLES X 3
        assert len(actions.shape) == 3 and actions.shape[-1] == 3
        bs = actions.shape[0]
        nsamples = actions.shape[1]

        # Normalize the actions.
        actions = actions / actions.norm(dim=2).unsqueeze(-1)

        # BATCH_SIZE X N_SAMPLES X 1024
        x = x.view(bs, 1, self.p.encoder_outdim)
        x = x.repeat_interleave(nsamples, dim=1)

        # BATCH_SIZE X N_SAMPLES X 1027
        x = torch.cat([x, actions], dim=2)

        # Flatten to put into the network.
        # (BATCH_SIZE*N_SAMPLES X 1027)
        x = x.view((-1, self.p.encoder_outdim + 3))

        # Run the MLP.
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        scores = self.lin3(x)

        # BATCH_SIZE X N_SAMPLES X 1
        scores = scores.view(bs, nsamples, 1)

        return scores

    def forward(self, data: tgd.Data, actions: torch.Tensor):
        # BATCH_SIZE X 1024
        x = self.pc_encoder(data)
        scores = self.score(x, actions)
        return scores


@torch.no_grad()
def score(actions, flows):
    assert len(flows.shape) == 2 and flows.shape[-1] == 3
    assert len(actions.shape) == 3 and actions.shape[-1] == 3

    # Normalize the flow and the actions.
    n_flow_ixs = (flows != 0.0).all(dim=-1)  # Sometimes it's zero.
    flows[n_flow_ixs] = flows[n_flow_ixs] / flows[n_flow_ixs].norm(dim=1).unsqueeze(-1)
    actions = actions / actions.norm(dim=2).unsqueeze(-1)

    scores = (flows.unsqueeze(1) * actions).sum(dim=-1)

    return scores


def umpnet_loss(pred_scores, actions, flows):
    gt_scores = score(actions, flows)
    loss = F.mse_loss(pred_scores.squeeze(), gt_scores.squeeze())
    return loss


@dataclass
class UMPNetParams:
    direction_net: DirectionNetParams = DirectionNetParams()
    lr: float = 0.0001
    n_train_samples: int = 100
    n_cem_samples: int = 64
    T_cem: float = 20
    cem_noise: float = 0.1


class UMPNet(pl.LightningModule):
    def __init__(self, params: UMPNetParams = UMPNetParams()) -> None:
        super().__init__()
        self.params = params
        self.net = DirectionNet(params.direction_net)

    @staticmethod
    def sample_from_unit_sphere(shape):
        # From: https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
        m = torch.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
        vecs = m.sample(shape)
        vecs = vecs / vecs.norm(dim=-1).unsqueeze(-1)
        return vecs

    def _step(self, batch: tgd.Batch, mode):
        batch = batch.to(self.device)
        actions = self.sample_from_unit_sphere(
            [batch.num_graphs, self.params.n_train_samples]
        ).to(self.device)

        data_list = batch.to_data_list()
        flows = torch.stack(
            [data.flow[data.flow.norm(dim=-1).argmax()] for data in data_list], dim=0
        )
        pred_scores = self.net(batch, actions)
        loss = umpnet_loss(pred_scores, actions, flows)

        self.log_dict({f"{mode}/loss": loss}, add_dataloader_idx=False)
        return loss

    @torch.no_grad()
    def forward(self, batch: tgd.Batch):  # type: ignore
        batch = batch.to(self.device)

        # First, sample candidates uniformly.
        r1_candidates = self.sample_from_unit_sphere(
            [batch.num_graphs, self.params.n_cem_samples]
        ).to(self.device)

        # Score the candidates.
        # BATCH_SIZE X N_SAMPLES
        r1_scores = self.net(batch, r1_candidates).squeeze(dim=-1)

        # Resample with replacement (a. la. boostrap)
        r1_probs = torch.exp(self.params.T_cem * r1_scores)
        r1_probs = r1_probs / r1_probs.sum(dim=-1).unsqueeze(-1)
        ixs = torch.multinomial(r1_probs, self.params.n_cem_samples, replacement=True)
        r2_candidates = torch.stack(
            [r1_candidates[i][ix] for i, ix in enumerate(ixs)], dim=0
        )

        # Add some noise, and renormalize.
        r2_noise = self.sample_from_unit_sphere(
            [batch.num_graphs, self.params.n_cem_samples]
        ).to(self.device)
        r2_noise = r2_noise * self.params.cem_noise
        r2_candidates = r2_candidates + r2_noise
        r2_candidates = r2_candidates / r2_candidates.norm(dim=2).unsqueeze(-1)

        # Rescore the candidates. Ugly indexing. Probably a better way...
        r2_scores = self.net(batch, r2_candidates).squeeze(dim=-1)
        r2_winners_ixs = r2_scores.argmax(dim=-1)
        r2_winners = torch.stack(
            [r2_candidates[i][ix] for i, ix in enumerate(r2_winners_ixs)]
        )
        r2_winners_scores = torch.stack(
            [r2_scores[i][ix] for i, ix in enumerate(r2_winners_ixs)]
        )

        return r2_winners, r2_winners_scores

    @staticmethod
    def val_metrics(batch):
        pass

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        return self._step(batch, "train")

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()
        name = "val" if dataloader_idx == 0 else "unseen"

        loss = self._step(batch, name)

        # Perform the optimization
        winners, scores = self(batch)
        data_list = batch.to_data_list()
        flows = torch.stack(
            [data.flow[data.flow.norm(dim=-1).argmax()] for data in data_list], dim=0
        )
        winner_quality = score(winners.unsqueeze(1), flows).squeeze()
        self.log_dict(
            {
                f"{name}/selected_scores": scores.mean().cpu().item(),
                f"{name}/selected_quality": winner_quality.mean().cpu().item(),
            },
            add_dataloader_idx=False,
        )

        return loss

    def configure_optimizers(self):
        return opt.Adam(params=self.parameters(), lr=self.params.lr)

    @staticmethod
    def make_plots(preds, batch: tgd.Batch) -> Dict[str, go.Figure]:

        data_list = batch.to_data_list()
        maxs = torch.stack([data.flow.norm(dim=-1).argmax() for data in data_list])
        flows = batch.flow[maxs]
        flows = flows / flows.norm(dim=-1).unsqueeze(-1)
        flow_pts = batch.pos[maxs]
        winners, _ = preds

        pos = batch.pos.detach().cpu()
        flow_pos = torch.cat([flow_pts, flow_pts], dim=0).cpu()
        flows = torch.cat([flows, winners], dim=0).cpu()
        flowcolor = ["darkblue", "red"]

        f = go.Figure()
        f.add_trace(pointcloud(pos, downsample=1, scene="scene1", name="pointcloud"))
        ts = _flow_traces(
            flow_pos[:1],
            flows[:1],
            scene="scene1",
            flowcolor=flowcolor[0],
            name="gt_flow",
        )
        for t in ts:
            f.add_trace(t)
        ts = _flow_traces(
            flow_pos[1:],
            flows[1:],
            scene="scene1",
            flowcolor=flowcolor[1],
            name="pred_flow",
        )
        for t in ts:
            f.add_trace(t)
        f.update_layout(scene1=_3d_scene(pos))

        return {"umpnet_plot": f}
