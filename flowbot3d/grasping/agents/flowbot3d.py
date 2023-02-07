from typing import Optional, Tuple

import gif
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import rpad.visualize_3d.plots as rvpl
import rpad.visualize_3d.primitives as rvpr
import torch

from flowbot3d.models.flowbot3d import ArtFlowNet


class FlowBot3DDetector:
    def __init__(
        self,
        ckpt_path,
        device,
        cam_frame: bool,
        animation: Optional["FlowNetAnimation"] = None,
    ):
        self.model = ArtFlowNet.load_from_checkpoint(ckpt_path).to(device)  # type: ignore
        self.model.eval()
        self.device = device
        self.animation = animation
        self.cam_frame = cam_frame

    def detect_contact_point(
        self, obs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        contact_point, flow_vector = self.max_flow_point(
            obs,
            top_k=1,
            animation=self.animation,
            cam_frame=self.cam_frame,
        )
        return contact_point, flow_vector

    def choose_pull_direction(self, obs) -> npt.NDArray[np.float32]:
        _, flow_vector = self.max_flow_point(
            obs,
            top_k=1,
            animation=self.animation,
            cam_frame=self.cam_frame,
        )
        return flow_vector

    def max_flow_point(
        self,
        obs,
        top_k,
        animation: Optional["FlowNetAnimation"],
        cam_frame=False,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """For the initial grasping point selection"""
        ee_coords = obs["ee_coords"]
        ee_center = 0.5 * (ee_coords[0] + ee_coords[1])

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
        gripper_pts = pcd_all[np.logical_not(mask_meta)]
        ee_to_pt_dist = np.linalg.norm(pcd - ee_center, axis=1)
        if not cam_frame:
            xyz = pcd - pcd.mean(axis=-2)
            scale = (1 / np.abs(xyz).max()) * 0.999999
            xyz = xyz * scale

            pred_flow = self.model.predict(
                torch.from_numpy(xyz).to(self.device),
                torch.from_numpy(mask_1[mask_meta]).float(),
            )
            pred_flow = pred_flow.cpu().numpy()

        else:
            cam_mat = obs["cam_mat"]
            pred_flow = self.model.predict(
                torch.from_numpy(pcd @ cam_mat[:3, :3] + cam_mat[:3, -1]).to(
                    self.device
                ),
                torch.from_numpy(mask_1[mask_meta]).float(),
            )
            pred_flow = pred_flow.cpu().numpy()
            pred_flow = pred_flow @ np.linalg.inv(cam_mat)[:3, :3]

        flow_norm_allpts = np.linalg.norm(pred_flow, axis=1)
        flow_norm_allpts = np.divide(flow_norm_allpts, ee_to_pt_dist)
        max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]
        max_flow_pt = pcd[max_flow_idx]
        max_flow_vector = pred_flow[max_flow_idx]  # type: ignore
        if animation:
            temp = animation.add_trace(
                torch.as_tensor(pcd),
                torch.as_tensor([pcd[mask_1[mask_meta]]]),
                torch.as_tensor(
                    [pred_flow[mask_1[mask_meta]] / np.linalg.norm(max_flow_vector)]
                ),
                "red",
            )
            animation.append_gif_frame(temp)

        max_flow_dir = max_flow_vector / np.linalg.norm(max_flow_vector)
        return (
            max_flow_pt.reshape((3,)),
            max_flow_dir,
        )  # type: ignore


class FlowNetAnimation:
    def __init__(self):
        self.num_frames = 0
        self.traces = {}
        self.gif_frames = []

    def add_trace(self, pos, flow_pos, flows, flowcolor):
        pcd = rvpl.pointcloud(pos, downsample=1)
        try:
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
        except IndexError:
            return

        # self.traces[self.num_frames] = pcd
        self.num_frames += 1
        if self.num_frames == 1 or self.num_frames == 0:
            self.pos = pos
            self.ts = ts

    @gif.frame
    def add_trace_gif(self, pos, flow_pos, flows, flowcolor):
        f = go.Figure()
        f.add_trace(rvpr.pointcloud(pos, downsample=1, scene="scene1"))
        ts = rvpl._flow_traces(
            flow_pos[0],
            flows[0],
            scene="scene1",
            flowcolor=flowcolor,
            name="pred_flow",
        )
        for t in ts:
            f.add_trace(t)
        f.update_layout(scene1=rvpl._3d_scene(pos))
        return f

    def append_gif_frame(self, f):
        self.gif_frames.append(f)

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

    @gif.frame
    def save_gif(self, dir):
        gif.save(self.gif_frames, dir, duration=100)

    def animate(self):
        self.show_animation()
        if self.num_frames == 0:
            return None
        k = np.random.permutation(np.arange(self.pos.shape[0]))[:500]
        self.fig.add_trace(rvpr.pointcloud(self.pos[k], downsample=1))
        for t in self.ts:
            self.fig.add_trace(t)
        # Layout
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.5, y=0, z=1.5),
        )
        self.fig.update_layout(
            title="FlowNet Flow Prediction",
            scene_camera=camera,
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
