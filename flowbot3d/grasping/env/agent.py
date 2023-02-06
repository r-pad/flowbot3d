import numpy as np
from mani_skill.agent.agent import Agent, get_actor_by_name
from mani_skill.agent.controllers import PositionController, VelocityController
from sapien.core import Pose


def concat_vec_in_dict(d, key_list):
    return np.concatenate(
        [
            d[key] if isinstance(d[key], np.ndarray) else np.array([d[key]])
            for key in key_list
        ]
    )


class FloatingGripperAgent(Agent):
    """This abstracts away all the getting/setting required for messing with the
    simulation directly, and controls a flying gripper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "floating_gripper"
        self.finger1_joint, self.finger2_joint = get_actor_by_name(
            self.robot.get_joints(),
            ["panda_finger_joint1", "panda_finger_joint2"],
        )
        self.finger1_link, self.finger2_link = get_actor_by_name(
            self.robot.get_links(),
            ["panda_leftfinger", "panda_rightfinger"],
        )
        self.hand = get_actor_by_name(self.robot.get_links(), "panda_hand")
        self.num_ee = 1
        self.full_state_len = len(
            self.get_state(by_dict=False, with_controller_state=True)
        )

    def get_state(self, by_dict=False, with_controller_state=False):
        state_dict = {}
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        state_dict["qpos"] = qpos
        state_dict["qvel"] = qvel
        if by_dict:
            return state_dict
        return np.concatenate(list(state_dict.values()))

    def get_base_link(self):
        return self.robot.get_links()[0]

    def set_state(self, state, by_dict=False):
        if not by_dict:
            assert (
                len(state) == self.full_state_len
            ), "length of state is not correct, probably because controller states are missing"
            state_dict = {
                "qpos": state[: self.robot.dof],
                "qvel": state[self.robot.dof : 2 * self.robot.dof],
                # "controller_state": state[2 * self.robot.dof :],
            }
        else:
            state_dict = state
        if "qpos" in state_dict:
            self.robot.set_qpos(state_dict["qpos"])
        if "qvel" in state_dict:
            self.robot.set_qvel(state_dict["qvel"])
        if "controller_state" in state_dict:
            # idx = 2*self.robot.dof
            state = state_dict["controller_state"]
            idx = 0
            for controller in self.controllers:
                if type(controller) == PositionController:
                    if state[idx]:
                        controller.velocity_pid._prev_err = state[idx + 1]
                    else:
                        controller.velocity_pid._prev_err = None
                    controller.velocity_pid._cum_err = state[idx + 2]
                    controller.lp_filter.y = state[idx + 3]
                    idx = idx + 4
                elif type(controller) == VelocityController:
                    controller.lp_filter.y = state[idx]
                    idx = idx + 1

    def get_ee_vels(self):
        finger_vels = [
            self.finger2_link.get_velocity(),
            self.finger1_link.get_velocity(),
        ]
        return np.array(finger_vels)

    def get_ee_coords(self):
        finger_tips = [
            self.finger2_joint.get_global_pose().transform(Pose([0, 0.035, 0])).p,
            self.finger1_joint.get_global_pose().transform(Pose([0, -0.035, 0])).p,
        ]
        return np.array(finger_tips)

    def get_body_link(self):
        raise NotImplementedError

    def get_ee_coords_sample(self):
        l = 0.0355
        r = 0.052
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                self.finger2_joint.get_global_pose().transform(Pose([0, x, 0])).p,
                self.finger1_joint.get_global_pose().transform(Pose([0, -x, 0])).p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))
