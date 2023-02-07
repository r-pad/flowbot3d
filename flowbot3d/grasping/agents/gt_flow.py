from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from rpad.partnet_mobility_utils.articulate import articulate_points


class GTFlowDetector:
    def __init__(self, bad_door):
        self.bad_door = bad_door

    def detect_contact_point(
        self, obs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        _, contact_point, _, pull_vector = GTFlowDetector.max_flow_pt_calc_no_ransac(
            obs, self.bad_door, 1
        )
        return contact_point, pull_vector

    def choose_pull_direction(self, obs) -> npt.NDArray[np.float32]:
        _, _, _, pull_vector = GTFlowDetector.max_flow_pt_calc_no_ransac(
            obs, self.bad_door, top_k=1
        )
        return pull_vector

    @staticmethod
    def transform_pcd(obs, chain, magnitude):
        """Transform the PCD observed in ManiSkill Env"""
        pcd = obs["pointcloud"]["xyz"]
        clr = obs["pointcloud"]["rgb"]
        seg = obs["pointcloud"]["seg"][:, :-1]

        # Get the seg pcd.
        seg_pcd = pcd[np.where(seg)[0], :]
        seg_clr = pcd[np.where(clr)[0], :]
        org_config = np.zeros(len(chain))
        target_config = np.ones(len(chain)) * magnitude * 5

        p_world_flowedpts = articulate_points(
            seg_pcd, np.eye(4), chain, org_config, target_config
        )

        flow = p_world_flowedpts - seg_pcd

        return p_world_flowedpts, flow, seg_pcd, seg_clr

    @staticmethod
    def transform_pcd_door(obs, chain, magnitude):
        """
        Special case to deal with Door objects' convex hull issues
        """

        pcd = obs["pointcloud"]["xyz"]
        clr = obs["pointcloud"]["rgb"]
        seg = obs["pointcloud"]["seg"][:, :-1]
        rot_mat = obs["T_org_object"]
        rot_mat = np.linalg.inv(rot_mat)
        pcd = pcd @ rot_mat[:3, :3] + rot_mat[:3, -1]

        # Get the seg pcd.
        seg_pcd = pcd[np.where(seg)[0], :]
        seg_clr = pcd[np.where(clr)[0], :]
        org_config = np.zeros(len(chain))
        target_config = np.ones(len(chain)) * magnitude * 5

        p_world_flowedpts = articulate_points(
            seg_pcd, np.eye(4), chain, org_config, target_config
        )

        flow = p_world_flowedpts - seg_pcd
        p_world_flowedpts = (
            p_world_flowedpts @ np.linalg.inv(rot_mat)[:3, :3]
            + np.linalg.inv(rot_mat)[:3, -1]
        )
        flow = flow @ np.linalg.inv(rot_mat)[:3, :3]
        seg_pcd = (
            seg_pcd @ np.linalg.inv(rot_mat)[:3, :3] + np.linalg.inv(rot_mat)[:3, -1]
        )

        return p_world_flowedpts, flow, seg_pcd, seg_clr

    @staticmethod
    def max_flow_pt_calc_no_ransac(
        obs, door, top_k=1
    ) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], Any, npt.NDArray[np.float32]
    ]:
        chain = obs["chain"]
        ee_coords = obs["ee_coords"]
        ee_center = 0.5 * (ee_coords[0] + ee_coords[1])
        if door:
            _, flow_test, seg_pcd, _ = GTFlowDetector.transform_pcd_door(
                obs, chain, 0.1
            )
        else:
            _, flow_test, seg_pcd, _ = GTFlowDetector.transform_pcd(obs, chain, 0.1)
        ee_to_pt_dist = np.linalg.norm(seg_pcd - ee_center, axis=1)
        flow_norm_allpts = np.linalg.norm(flow_test, axis=1)
        flow_norm_allpts = np.divide(flow_norm_allpts, ee_to_pt_dist)
        max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]

        # Map the argmax back to the original seg pcd
        max_flow_knob_pt = seg_pcd[max_flow_idx, :]
        max_flow_knob_vector = flow_test[max_flow_idx]
        knob_pts = seg_pcd
        if not max_flow_knob_pt.any():
            return None, None, None, None  # type: ignore

        return (knob_pts, max_flow_knob_pt[0], chain, max_flow_knob_vector)
