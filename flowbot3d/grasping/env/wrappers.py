import gym


class FlowBot3DWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        cam_mat = (
            self.env.cameras[1].sub_cameras[0].get_pose().to_transformation_matrix()
        )
        T_org_pose = self.env.agent.robot.get_root_pose().to_transformation_matrix()

        return {
            **observation,
            "ee_coords": self.env.agent.get_ee_coords(),
            "cam_mat": cam_mat,
            "T_org_pose": T_org_pose,
            "robot_qpos": self.env.agent.robot.get_qpos(),
            "ee_vels": self.env.agent.get_ee_vels(),
        }
