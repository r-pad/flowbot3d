import abc
from typing import Protocol, Tuple

import numpy as np
import numpy.typing as npt

DEFAULT_GLOBAL_PULL_VECTOR = np.array([[-1, 0, 0]])


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


class GraspPullAgent:
    def __init__(
        self,
        contact_detector: ContactDetector,
        pull_dir_detector: PullDirectionDetector,
        device,
    ):
        self.device = device
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
