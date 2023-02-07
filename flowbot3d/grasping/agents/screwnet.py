from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from rpad.partnet_mobility_utils.articulate import articulate_points

from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation
from flowbot3d.models.screwnet import ScrewNet


class ScrewNetDetector:
    def __init__(
        self,
        ckpt_path,
        device,
        animation: Optional[FlowNetAnimation] = None,
        cam_frame: bool = False,
    ):
        self.device = device
        self.model = ScrewNet.load_from_checkpoint(ckpt_path).to(device)
        self.model.eval()
        self.animation = animation
        self.cam_frame = cam_frame

    def detect_contact_point(
        self, obs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return self.max_flow_pt_calc(
            obs, self.model, top_k=1, animation=self.animation, cam_frame=self.cam_frame
        )

    def choose_pull_direction(self, obs) -> npt.NDArray[np.float32]:
        _, pull_vector = self.max_flow_pt_calc(
            obs, self.model, top_k=1, animation=self.animation, cam_frame=self.cam_frame
        )
        return pull_vector

    def max_flow_pt_calc(
        self, obs, model, top_k, animation, cam_frame=False
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
            pred_axis, pred_origin = model.predict(
                torch.from_numpy(pcd).to(self.device),
                torch.from_numpy(mask_1[mask_meta]).float(),
            )
            pred_axis = pred_axis.cpu().numpy()
            pred_axis = pred_axis / np.linalg.norm(pred_axis.squeeze())
            pred_origin = pred_origin.cpu().numpy()

            final_joint = obs["final_joint"]
            final_joint.origin = pred_origin.squeeze(), np.zeros((3,))
            final_joint.axis = pred_axis.squeeze()

            newpts = articulate_points(
                pcd, np.eye(4), [final_joint], current_ja=[0.0], target_ja=[0.01]
            )
            pred_flow = newpts - pcd
            nonzero = (pred_flow != 0.0).all(axis=-1)
            pred_flow[nonzero] = (
                pred_flow[nonzero] / np.linalg.norm(pred_flow[nonzero], axis=-1).max()
            )

            assert len(nonzero) > 0

            pred_flow[~mask_1[mask_meta]] = [0, 0, 0]

        else:
            cam_mat = obs["cam_mat"]
            pred_flow = model.predict(
                torch.from_numpy(pcd @ cam_mat[:3, :3] + cam_mat[:3, -1]).to(
                    self.device
                ),
                torch.from_numpy(mask_1[mask_meta]).float(),
            )
            pred_flow = pred_flow.cpu().numpy()
            pred_flow = pred_flow @ np.linalg.inv(cam_mat)[:3, :3]

        flow_norm_allpts = np.linalg.norm(pred_flow, axis=1)
        # flow_norm_allpts = np.divide(flow_norm_allpts, ee_to_pt_dist)
        max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]
        max_flow_pt = pcd[max_flow_idx]
        max_flow_vector = pred_flow[max_flow_idx]
        # OUTPUT A DEFAULT Y FLOW.
        if (max_flow_vector == 0.0).all():
            max_flow_vector[:, -2] = 1.0
        if animation:
            temp = animation.add_trace(
                torch.as_tensor(pcd),
                torch.as_tensor([pcd[mask_1[mask_meta]]]),
                torch.as_tensor(
                    [pred_flow[mask_1[mask_meta]] / np.linalg.norm(max_flow_vector)]
                ),
                "red",
            )

        max_flow_dir = (max_flow_vector / np.linalg.norm(max_flow_vector),)

        return max_flow_pt.reshape((3,)), max_flow_dir  # type: ignore
