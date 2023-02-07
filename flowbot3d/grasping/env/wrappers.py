import gym
from rpad.partnet_mobility_utils.urdf import PMTree


class FlowBot3DWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        cam_mat = (
            self.env.cameras[1].sub_cameras[0].get_pose().to_transformation_matrix()
        )
        T_org_pose = self.env.agent.robot.get_root_pose().to_transformation_matrix()

        T_org_object = self.env.cabinet.get_root_pose().to_transformation_matrix()

        # obs = process_mani_skill_base(obs, env)
        urdf = PMTree.parse_urdf_from_string(self.env.cabinet.export_urdf())
        chain = urdf.get_chain(self.env.target_link.name)

        return {
            **observation,
            "ee_coords": self.env.agent.get_ee_coords(),
            "cam_mat": cam_mat,
            "T_org_pose": T_org_pose,
            "T_org_object": T_org_object,
            "robot_qpos": self.env.agent.robot.get_qpos(),
            "ee_vels": self.env.agent.get_ee_vels(),
            "chain": chain,
            "cam_pos": self.env.cameras[1].sub_cameras[0].get_pose().p,
        }
