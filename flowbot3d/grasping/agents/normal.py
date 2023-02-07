from typing import Optional

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
from scipy.spatial.distance import cdist

from flowbot3d.grasping.agents.umpnet_di import UMPAnimation


class NormalPullDirectionDetector:
    def __init__(
        self, animation: Optional[UMPAnimation] = None, cam_frame: bool = False
    ):
        self.cam_frame = cam_frame
        self.animation = animation

    def choose_pull_direction(self, obs) -> npt.NDArray[np.float32]:
        pull_vector = NormalPullDirectionDetector.normal_calc(
            obs, self.animation, self.cam_frame
        )
        return pull_vector

    @staticmethod
    def normal_calc(obs, animation, cam_frame=False) -> npt.NDArray[np.float32]:
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

        mask_meta = np.logical_or(mask_1, mask_2)
        pcd = pcd_all[mask_meta]
        if not cam_frame:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd_all[mask_1])
            pcd_o3d.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            cam_pos = obs["cam_pos"]
            pcd_o3d.orient_normals_towards_camera_location(camera_location=cam_pos)

            normals = np.array(pcd_o3d.normals)
            # pred_flow = normals.mean(axis=0)
            dist = cdist(
                0.5 * (ee_coords[0] + ee_coords[1]).reshape(1, 3),
                pcd_all[mask_1],
            )[:, 1]
            min_dist_ind = dist.argmin()
            pred_flow = normals[min_dist_ind]
        else:
            raise NotImplementedError

        if animation:
            animation.add_trace(
                torch.as_tensor(pcd),
                torch.as_tensor([0.5 * (ee_coords[0] + ee_coords[1])]),
                torch.as_tensor([pred_flow]),
                "red",
            )

        # print("max_flow: ", pred_flow)
        return pred_flow / np.linalg.norm(pred_flow)  # type: ignore
