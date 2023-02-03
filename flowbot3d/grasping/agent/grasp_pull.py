import abc
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import numpy.typing as npt
import torch
from rpad.partnet_mobility_utils.articulate import articulate_points

from flowbot3d.models.flowbot3d import ArtFlowNet
from flowbot3d.visualizations import FlowNetAnimation

DEFAULT_GLOBAL_PULL_VECTOR = np.array([[-1, 0, 0]])


class PCAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, obs) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def reset(self, obs):
        pass


class ContactDetector(Protocol):
    @abc.abstractmethod
    def detect_contact_point(
        self, obs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        pass


class PullDirectionDetector(Protocol):
    @abc.abstractmethod
    def choose_pull_direction(self, obs) -> npt.NDArray[np.float32]:
        pass


class FlowBot3DDetector:
    def __init__(
        self,
        ckpt_path,
        device,
        cam_frame: bool,
        animation: Optional[FlowNetAnimation] = None,
    ):
        self.model = ArtFlowNet.load_from_checkpoint(ckpt_path).to(device)
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
        animation: Optional[FlowNetAnimation],
        cam_frame=False,
    ):
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
        max_flow_vector = pred_flow[max_flow_idx]
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

        return (
            max_flow_pt.reshape(
                3,
            ),
            max_flow_vector / np.linalg.norm(max_flow_vector),
        )


class UMPNetPullDirectionDetector:
    pass


class NormalPullDirectionDetector:
    pass


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
    def max_flow_pt_calc_no_ransac(obs, door, top_k=1):
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
            return None, None, None, None

        return (knob_pts, max_flow_knob_pt[0], chain, max_flow_knob_vector)


class GraspPullAgent(PCAgent):
    def __init__(
        self,
        contact_detector: ContactDetector,
        pull_dir_detector: PullDirectionDetector,
        device,
        animation: FlowNetAnimation,
    ):
        self.device = device
        self.animation = animation
        self.contact_detector = contact_detector
        self.pull_dir_detector = pull_dir_detector

        self.gripper_horizontal = None
        self.gripper_vertical = None
        self.contact_point = None
        self.pull_vector = None
        self.phase_counter = None
        self.T_pose_back = None
        self.phase = None

        self.global_pull_vector = DEFAULT_GLOBAL_PULL_VECTOR

    def reset(self, obs):
        # Find the contact point.
        contact_point, contact_pull_vector = self.contact_detector.detect_contact_point(
            obs
        )
        if contact_pull_vector is None:
            contact_pull_vector = np.array([[0, 0, 0]])
        if abs(contact_pull_vector[0, 2]) > abs(contact_pull_vector[0, 1]) and abs(
            contact_pull_vector[0, 2]
        ) > abs(contact_pull_vector[0, 0]):
            self.gripper_vertical = True
        else:
            self.gripper_vertical = False

        pcd = obs["pointcloud"]["xyz"]
        pcd = pcd[np.where(pcd[:, 2] > 0.1)]
        self.contact_point = contact_point

        self.gripper_horizontal = True
        self.phase = 1

        # Stepping in gym
        self.phase_counter = 0
        # trans_m_w_matrix = T_m_w
        T_org_pose = obs["T_org_pose"]
        self.T_pose_back = np.linalg.inv(T_org_pose)

        # Pull vector as the flow direction of the largest flow vector
        self.pull_vector = contact_pull_vector

    def get_action(self, obs):
        action, self.phase, self.phase_counter, add_dr = self.grasp_and_pull_policy(
            obs,
            self.contact_point,
            self.pull_vector,
            self.phase,
            self.phase_counter,
            self.T_pose_back,
            self.gripper_vertical,
        )

        extras = {}
        if add_dr:
            extras["drive"] = True

        return action, extras

    def grasp_and_pull_policy(
        self,
        obs,
        max_flow_pt,
        pull_vector,
        phase,
        phase_counter,
        aux_T,
        gripper_vertical,
    ):
        ee_coords = obs["ee_coords"]
        robot_qpos = obs["robot_qpos"]
        ee_vels = obs["ee_vels"]

        # Define primitives
        action = np.zeros(8)
        ee_center = 0.5 * (ee_coords[0] + ee_coords[1])
        # Phase 1: Grasp
        if phase == 1:

            delta_T = (max_flow_pt - ee_center) / np.linalg.norm(
                max_flow_pt - ee_center
            )
            # delta_R = 0.2 * R.from_matrix(T_robot_goal[:3, :3]).as_euler("xyz")
            action = np.zeros(8)

            # if gripper_horizontal and robot_qpos[5] < np.pi / 2:
            if gripper_vertical and robot_qpos[3] > -np.pi / 2:
                action[3] = -1
                # print("Rotating Gripper")
            if robot_qpos[5] < np.pi / 2:
                action[5] = 1
                # print("Rotating Gripper")
            if not np.isclose(max_flow_pt, ee_center, atol=0.1).all():
                # print("MOVING EE")
                action[:3] = aux_T[:3, :3] @ delta_T
                # print(action)
                vel = np.linalg.norm(ee_vels.mean(axis=0))
                if vel < 1e-2:
                    # print(vel)
                    # print(phase_counter)
                    if phase_counter < 10:
                        phase_counter += 1
                    else:
                        action[:3] = aux_T[:3, :3] @ np.array([-1, 0, 1])
                elif phase_counter >= 1:
                    phase_counter -= 1
                    action[:3] = aux_T[:3, :3] @ np.array([-1, 0, 1])
                else:
                    phase_counter = 0
            else:
                # print("EE HOLD")
                if phase_counter >= 10:
                    phase = 2
                    phase_counter = 0
                else:
                    if not np.isclose(max_flow_pt, ee_center).all():
                        action[:3] = aux_T[:3, :3] @ delta_T
                    phase_counter += 1
            return action, phase, phase_counter, None
        # Phase 2: Back Up
        else:
            action = np.zeros(8)
            if phase_counter >= 10:
                add_dr = True

                pull_vector = self.pull_dir_detector.choose_pull_direction(obs)
                pull_vector = pull_vector.reshape(1, 3)
                # print("pull vec: ", pull_vector)
                if pull_vector is not None:
                    angle = np.dot(
                        pull_vector.reshape(
                            3,
                        )
                        / np.linalg.norm(pull_vector),
                        self.global_pull_vector.reshape(
                            3,
                        )
                        / (1e-7 + np.linalg.norm(self.global_pull_vector)),
                    )

                else:
                    pull_vector = self.global_pull_vector
                    angle = 1

                angle = abs(np.arccos(angle))
                v = np.cross(
                    self.global_pull_vector.reshape(
                        3,
                    )
                    / (1e-7 + np.linalg.norm(self.global_pull_vector)),
                    pull_vector.reshape(
                        3,
                    )
                    / (1e-7 + np.linalg.norm(pull_vector)),
                )
                # v = v / np.linalg.norm(v + 1e-7)
                if abs(pull_vector[0]).argmin() == 2:
                    vn = np.array([0, 0, 1])
                elif abs(pull_vector[0]).argmin() == 1:
                    vn = np.array([0, 1, 0])
                elif abs(pull_vector[0]).argmin() == 0:
                    vn = np.array([1, 0, 0])
                if np.dot(vn, v) > 0:
                    angle = -angle

                if abs(pull_vector[0, 1]) > abs(pull_vector[0, 2]):
                    action[4] = 0.5 * np.sign(angle)
                elif abs(pull_vector[0, 2]) > abs(pull_vector[0, 1]):
                    action[3] = 0.5 * np.sign(angle)
                self.global_pull_vector = pull_vector
                action[0:3] = (
                    1
                    * (aux_T[:3, :3] @ pull_vector.reshape(3, 1)).squeeze()
                    / np.linalg.norm(pull_vector)
                )
                action[3:6] += 1e-5
                phase_counter += 1

            else:
                add_dr = False
                phase_counter += 1
                action[0] = 0
            # print("PHASE 3 GRASPING AND BACKING UP")
            return action, phase, phase_counter, add_dr
