from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import rpad.visualize_3d.plots as rvpl
import rpad.visualize_3d.primitives as rvpr
import torch
import torch_geometric.data as tgd

from flowbot3d.models.umpnet_di import UMPNet


class UMPNetPullDirectionDetector:
    def __init__(
        self, ckpt_path, device, animation: Optional["UMPAnimation"] = None
    ) -> None:
        self.device = device
        self.model = UMPNet.load_from_checkpoint(ckpt_path).to(device)
        self.model.eval()
        self.animation = animation

    def choose_pull_direction(self, obs) -> npt.NDArray[np.float32]:
        return self.max_flow_pt_calc(obs, self.model, self.animation)

    def max_flow_pt_calc(
        self, obs, model, animation: Optional["UMPAnimation"]
    ) -> npt.NDArray[np.float32]:
        ee_coords = obs["ee_coords"]
        filter = np.where(obs["pointcloud"]["xyz"][:, 2] > 1e-3)
        pcd_all = obs["pointcloud"]["xyz"][filter]
        # this one gives the door only
        mask_1 = obs["pointcloud"]["seg"][:, :-1].any(axis=1)[filter]
        # This one should give everything but the door
        mask_2 = np.logical_not(obs["pointcloud"]["seg"][:, :].any(axis=1))[filter]
        filter_2 = np.random.permutation(np.arange(pcd_all.shape[0]))[:1200]
        pcd_all = pcd_all[filter_2]
        mask_1 = mask_1[filter_2]
        mask_2 = mask_2[filter_2]
        # This one should give us only the cabinet
        mask_meta = np.logical_or(mask_1, mask_2)
        pcd = pcd_all[mask_meta]
        xyz = pcd
        pcd = tgd.Data(pos=torch.from_numpy(pcd))
        pcd = tgd.Batch.from_data_list([pcd]).to(self.device)
        model.eval()
        pred_flow, score = model.forward(pcd)
        pred_flow = pred_flow.cpu().numpy()
        if animation:
            animation.add_trace(
                torch.as_tensor(xyz),
                torch.as_tensor([[0.5 * (ee_coords[0] + ee_coords[1])]]),
                torch.as_tensor([pred_flow]),
                "red",
            )

        return pred_flow / np.linalg.norm(pred_flow)  # type: ignore


class UMPAnimation:
    def __init__(self):
        self.num_frames = 0
        self.traces = {}

    def add_trace(self, pos, flow_pos, flows, flowcolor):
        pcd = rvpr.pointcloud(pos.T, downsample=1)
        ts = rvpl._flow_traces(
            flow_pos[0],
            flows[0],
            0.1,
            scene="scene1",
            flowcolor=flowcolor,
            name="pred_flow",
        )
        self.traces[self.num_frames] = [pcd]
        for t in ts:
            self.traces[self.num_frames].append(t)

        self.num_frames += 1
        if self.num_frames == 1:
            self.pos = pos
            self.ts = ts

    def show_animation(self):
        self.fig = go.Figure(
            frames=[
                go.Frame(data=self.traces[k], name=str(k))
                for k in range(self.num_frames)
            ]
        )

    def frame_args(self, duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    def set_sliders(self):
        self.sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], self.frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(self.fig.frames)
                ],
            }
        ]
        return self.sliders

    def animate(self):
        self.show_animation()
        self.fig.add_trace(rvpr.pointcloud(self.pos, downsample=1))
        for t in self.ts:
            self.fig.add_trace(t)
        # Layout
        self.fig.update_layout(
            title="UMPNet Flow Prediction",
            width=600,
            height=600,
            scene1=rvpl._3d_scene(self.pos),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, self.frame_args(50)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], self.frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=self.set_sliders(),
        )
        # self.fig.show()
        return self.fig
