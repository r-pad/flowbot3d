"""This file modifies the 7-dof franka panda arm environments to get
flying gripper environments."""
import pathlib
from copy import deepcopy

import numpy as np
import sapien
from sapien.core import Articulation, Pose

from flowbot3d.datasets.calc_art import compute_new_points
from flowbot3d.datasets.pm.pm_raw import parse_urdf_from_string
from third_party.ManiSkill.mani_skill.env.open_cabinet_door_drawer import (
    OpenCabinetEnvBase,
)
from third_party.ManiSkill.mani_skill.utils.config_parser import (
    process_variables,
    process_variants,
)
from third_party.ManiSkill.mani_skill.utils.misc import (
    get_actor_state,
    get_pad_articulation_state,
)

_this_file = pathlib.Path(__file__).resolve()


class OpenCabinetDoorGripperEnv(OpenCabinetEnvBase):
    def __init__(self, *args, split="train", **kwargs):
        super().__init__(
            _this_file.parent.joinpath(
                "assets/config_files/open_cabinet_door_floating.yml"
            ),
            *args,
            **kwargs,
        )

    def transform_pcd(self, obs, link, chain, magnitude):
        """Transform the PCD observed in ManiSkill Env"""
        pcd = obs["pointcloud"]["xyz"]
        seg = self.seg_all[link]

        # Get the seg pcd.
        seg_pcd = pcd[np.where(seg)[0], :]
        org_config = np.zeros(len(chain))
        target_config = np.ones(len(chain)) * magnitude

        p_world_flowedpts = compute_new_points(
            seg_pcd, np.eye(4), chain, org_config, target_config
        )

        flow_local = p_world_flowedpts - seg_pcd
        flow = np.zeros_like(pcd)
        flow[np.where(seg)[0], :] = flow_local

        return flow

    def reset(self, level=None):
        if level is None:
            level = self._main_rng.randint(2**32)
        self.level = level
        self._level_rng = np.random.RandomState(seed=self.level)

        # recreate scene
        scene_config = sapien.core.SceneConfig()
        scene_config.gravity = [0, 0, 0]
        for p, v in self.yaml_config["physics"].items():
            if p != "simulation_frequency":
                setattr(scene_config, p, v)
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(self.timestep)

        config = deepcopy(self.yaml_config)
        config = process_variables(config, self._level_rng)
        self.all_model_ids = list(
            config["layout"]["articulations"][0]["_variants"]["options"].keys()
        )
        self.level_config, self.level_variant_config = process_variants(
            config, self._level_rng, self.variant_config
        )

        # load everything
        self._setup_renderer()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_actors()
        self._load_articulations()
        self._setup_objects()
        self._load_agent()
        self._load_custom()
        self._setup_cameras()
        if self._viewer is not None:
            self._setup_viewer()

        self._init_eval_record()
        self.step_in_ep = 0

        self.cabinet: Articulation = self.articulations["cabinet"]["articulation"]

        self._place_cabinet()
        self._close_all_parts()
        self._find_handles_from_articulation()
        self._place_robot()
        self._choose_target_link()
        self._ignore_collision()
        self._set_joint_physical_parameters()
        self._prepare_for_obs()

        [[lmin, lmax]] = self.target_joint.get_limits()

        self.target_qpos = lmin + (lmax - lmin) * self.custom["open_extent"]
        self.pose_at_joint_zero = self.target_link.get_pose()

        # Get all movable parts
        for i, j in enumerate(self.cabinet.get_active_joints()):
            if j.get_child_link().name == self.target_link.name:
                joint_type = j.type
                break

        self.flowed_joints = []
        for i, j in enumerate(self.cabinet.get_active_joints()):
            if j.type == joint_type:
                [[l_min, l_max]] = j.get_limits()
                if np.inf != l_max and np.inf != l_min:
                    self.flowed_joints.append(j)

        return self.get_obs()

    def _choose_target_link(self):
        super()._choose_target_link("revolute")

    def get_obs(self, seg="both", **kwargs):  # seg can be 'visual', 'actor', 'both'
        if self.obs_mode == "custom":
            return self.get_custom_observation()
        if self.obs_mode == "state":
            return self.get_state(with_controller_state=False)
        elif self.obs_mode == "rgbd":
            obs = {
                "agent": self.agent.get_state(with_controller_state=False),
                "rgbd": self.render(
                    mode="color_image",
                    depth=True,
                    camera_names=["robot"],
                    seg=seg,
                    **kwargs,
                ),
            }
        elif self.obs_mode == "pointcloud":
            obs = {
                "agent": self.agent.get_state(with_controller_state=False),
                "pointcloud": self.render(
                    mode="pointcloud", camera_names=["robot"], seg=seg, **kwargs
                ),
            }
        # post processing
        if self.obs_mode == "pointcloud" or self.obs_mode == "rgbd":
            views = obs[
                self.obs_mode
            ]  # views is also a dict, keys including 'robot', 'world', ...
            for cam_name, view in views.items():
                if isinstance(view, list):
                    for view_dict in view:
                        self._post_process_view(view_dict)
                    combined_view = {}
                    for key in view[0].keys():
                        combined_view[key] = np.concatenate(
                            [view_dict[key] for view_dict in view], axis=-1
                        )
                    views[cam_name] = combined_view
                else:  # view is a dict
                    self._post_process_view(view)
            if len(views) == 1:
                view = next(iter(views.values()))
                obs[self.obs_mode] = view
                if self.obs_mode == "pointcloud":
                    # Get flow
                    # urdf = parse_urdf(
                    #     "./partnet-mobility-dataset/{}/mobility.urdf".format(self.selected_id))
                    urdf = parse_urdf_from_string(self.cabinet.export_urdf())
                    chain = urdf.get_chain(self.target_link.name)
                    flow = self.transform_pcd(obs, self.target_link.name, chain, 0.1)
                    flow_all = {}
                    for link in self.seg_all.keys():
                        seg = self.seg_all[link][:, :-1]
                        # import pdb
                        #
                        # pdb.set_trace()
                        chain = urdf.get_chain(link)
                        flow_all[link] = self.transform_pcd(obs, link, chain, 0.1)
                    # Get delta theta
                    info, done = self._eval()
                    if not info["open_enough"]:
                        obs[self.obs_mode]["d_theta"] = np.array([0.1])
                        obs[self.obs_mode]["flow"] = flow
                        obs[self.obs_mode]["flow_all"] = flow_all
                    else:
                        obs[self.obs_mode]["d_theta"] = np.array([0.0])
                        obs[self.obs_mode]["flow"] = flow
                        obs[self.obs_mode]["flow_all"] = flow_all

        return obs

    def _post_process_view(self, view_dict):
        visual_id_seg = view_dict["seg"][..., 0]  # (n, m)
        actor_id_seg = view_dict["seg"][..., 1]  # (n, m)

        masks = [np.zeros(visual_id_seg.shape, dtype=np.bool) for _ in range(3)]

        for visual_id in self.handle_visual_ids:
            masks[0] = masks[0] | (visual_id_seg == visual_id)
        for actor_id in self.target_link_ids:
            masks[1] = masks[1] | (actor_id_seg == actor_id)
        for actor_id in self.robot_link_ids:
            masks[2] = masks[2] | (actor_id_seg == actor_id)

        # All movable parts
        seg_all = {}
        for j in self.flowed_joints:
            actor_id = j.get_child_link().get_id()
            masks_all = [np.zeros(visual_id_seg.shape, dtype=np.bool) for _ in range(3)]
            for visual_id in self.handle_visual_ids:
                masks_all[0] = masks_all[0] | (actor_id == visual_id)
            masks_all[1] = masks_all[1] | (actor_id_seg == actor_id)
            for actor_id in self.robot_link_ids:
                masks_all[2] = masks_all[2] | (actor_id_seg == actor_id)
            seg_all[j.get_child_link().name] = np.stack(masks_all, axis=-1)

        view_dict["seg"] = np.stack(masks, axis=-1)
        self.seg_all = seg_all

    def get_state(self, with_controller_state=True):
        actors, arts = self.get_all_objects_in_state()

        actors_state = [get_actor_state(actor) for actor in actors]
        arts_state = [get_pad_articulation_state(art, max_dof) for art, max_dof in arts]

        return np.concatenate(
            actors_state
            + arts_state
            + [
                self.get_additional_task_info(obs_mode="state"),
                self.agent.get_state(with_controller_state=with_controller_state),
            ]
        )

    def set_state(self, state):
        # set actors
        actors, arts = self.get_all_objects_in_state()
        for actor in actors:
            actor.set_pose(pose=Pose(p=state[:3], q=state[3:7]))
            actor.set_velocity(state[7:10])
            actor.set_angular_velocity(state[10:13])
            state = state[13:]

        # set articulations
        for art, max_dof in arts:
            art.set_root_pose(Pose(state[0:3], state[3:7]))
            art.set_root_velocity(state[7:10])
            art.set_root_angular_velocity(state[10:13])
            state = state[13:]
            # import pdb; pdb.set_trace()
            art.set_qpos(state[0 : art.dof])
            art.set_qvel(state[max_dof : max_dof + art.dof])
            state = state[2 * max_dof :]

        # skip task info
        task_info_len = len(self.get_additional_task_info(obs_mode="state"))
        state = state[task_info_len:]

        # set robot state
        self.agent.set_state(state)

        return self.get_obs()

    @property
    def num_target_links(self):
        return super().num_target_links("revolute")


class OpenCabinetDrawerGripperEnv(OpenCabinetEnvBase):
    def __init__(
        self,
        yaml_file_path=_this_file.parent.joinpath(
            "assets/config_files/open_cabinet_drawer_floating.yml"
        ),
        *args,
        **kwargs,
    ):
        super().__init__(yaml_file_path=yaml_file_path, *args, **kwargs)

    def transform_pcd(self, obs, link, chain, magnitude):
        """Transform the PCD observed in ManiSkill Env"""
        pcd = obs["pointcloud"]["xyz"]
        seg = self.seg_all[link]

        # Get the seg pcd.
        seg_pcd = pcd[np.where(seg)[0], :]
        org_config = np.zeros(len(chain))
        target_config = np.ones(len(chain)) * magnitude

        p_world_flowedpts = compute_new_points(
            seg_pcd, np.eye(4), chain, org_config, target_config
        )

        flow_local = p_world_flowedpts - seg_pcd
        flow = np.zeros_like(pcd)
        flow[np.where(seg)[0], :] = flow_local

        return flow

    def reset(self, level=None):
        if level is None:
            level = self._main_rng.randint(2**32)
        self.level = level
        self._level_rng = np.random.RandomState(seed=self.level)

        # recreate scene
        scene_config = sapien.core.SceneConfig()
        scene_config.gravity = [0, 0, 0]
        for p, v in self.yaml_config["physics"].items():
            if p != "simulation_frequency":
                setattr(scene_config, p, v)
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(self.timestep)

        config = deepcopy(self.yaml_config)
        config = process_variables(config, self._level_rng)
        self.all_model_ids = list(
            config["layout"]["articulations"][0]["_variants"]["options"].keys()
        )
        self.level_config, self.level_variant_config = process_variants(
            config, self._level_rng, self.variant_config
        )

        # load everything
        self._setup_renderer()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_actors()
        self._load_articulations()
        self._setup_objects()
        self._load_agent()
        self._load_custom()
        self._setup_cameras()
        if self._viewer is not None:
            self._setup_viewer()

        self._init_eval_record()
        self.step_in_ep = 0

        self.cabinet: Articulation = self.articulations["cabinet"]["articulation"]

        self._place_cabinet()
        self._close_all_parts()
        self._find_handles_from_articulation()
        self._place_robot()
        self._choose_target_link()
        self._ignore_collision()
        self._set_joint_physical_parameters()
        self._prepare_for_obs()

        [[lmin, lmax]] = self.target_joint.get_limits()

        self.target_qpos = lmin + (lmax - lmin) * self.custom["open_extent"]
        self.pose_at_joint_zero = self.target_link.get_pose()

        # Get all movable parts
        for i, j in enumerate(self.cabinet.get_active_joints()):
            if j.get_child_link().name == self.target_link.name:
                joint_type = j.type
                break

        self.flowed_joints = []
        for i, j in enumerate(self.cabinet.get_active_joints()):
            if j.type == joint_type:
                [[l_min, l_max]] = j.get_limits()
                if np.inf != l_max and np.inf != l_min:
                    self.flowed_joints.append(j)

        return self.get_obs()

    def _choose_target_link(self):
        super()._choose_target_link("prismatic")

    def get_obs(self, seg="both", **kwargs):  # seg can be 'visual', 'actor', 'both'
        if self.obs_mode == "custom":
            return self.get_custom_observation()
        if self.obs_mode == "state":
            return self.get_state(with_controller_state=False)
        elif self.obs_mode == "rgbd":
            obs = {
                "agent": self.agent.get_state(with_controller_state=False),
                "rgbd": self.render(
                    mode="color_image",
                    depth=True,
                    camera_names=["robot"],
                    seg=seg,
                    **kwargs,
                ),
            }
        elif self.obs_mode == "pointcloud":
            # self.render("pointcloud", False, seg, ["robot"])
            obs = {
                "agent": self.agent.get_state(with_controller_state=False),
                "pointcloud": self.render(
                    mode="pointcloud", camera_names=["robot"], seg=seg, **kwargs
                ),
            }
        # post processing
        if self.obs_mode == "pointcloud" or self.obs_mode == "rgbd":
            views = obs[
                self.obs_mode
            ]  # views is also a dict, keys including 'robot', 'world', ...
            for cam_name, view in views.items():
                if isinstance(view, list):
                    for view_dict in view:
                        self._post_process_view(view_dict)
                    combined_view = {}
                    for key in view[0].keys():
                        combined_view[key] = np.concatenate(
                            [view_dict[key] for view_dict in view], axis=-1
                        )
                    views[cam_name] = combined_view
                else:  # view is a dict
                    self._post_process_view(view)
            if len(views) == 1:
                view = next(iter(views.values()))
                obs[self.obs_mode] = view
                if self.obs_mode == "pointcloud":
                    # Get flow
                    # urdf = parse_urdf(
                    #     "./partnet-mobility-dataset/{}/mobility.urdf".format(self.selected_id))
                    urdf = parse_urdf_from_string(self.cabinet.export_urdf())
                    chain = urdf.get_chain(self.target_link.name)
                    flow = self.transform_pcd(obs, self.target_link.name, chain, 0.1)
                    flow_all = {}
                    for link in self.seg_all.keys():
                        seg = self.seg_all[link][:, :-1]
                        chain = urdf.get_chain(link)
                        flow_all[link] = self.transform_pcd(obs, link, chain, 0.1)
                    # Get delta theta
                    info, done = self._eval()
                    if not info["open_enough"]:
                        obs[self.obs_mode]["d_theta"] = np.array([0.1])
                        obs[self.obs_mode]["flow"] = flow
                        obs[self.obs_mode]["flow_all"] = flow_all
                    else:
                        obs[self.obs_mode]["d_theta"] = np.array([0.0])
                        obs[self.obs_mode]["flow"] = flow
                        obs[self.obs_mode]["flow_all"] = flow_all
        return obs

    def _post_process_view(self, view_dict):
        visual_id_seg = view_dict["seg"][..., 0]  # (n, m)
        actor_id_seg = view_dict["seg"][..., 1]  # (n, m)

        masks = [np.zeros(visual_id_seg.shape, dtype=np.bool) for _ in range(3)]

        for visual_id in self.handle_visual_ids:
            masks[0] = masks[0] | (visual_id_seg == visual_id)
        for actor_id in self.target_link_ids:
            masks[1] = masks[1] | (actor_id_seg == actor_id)
        for actor_id in self.robot_link_ids:
            masks[2] = masks[2] | (actor_id_seg == actor_id)

        # All movable parts
        seg_all = {}
        for j in self.flowed_joints:
            actor_id = j.get_child_link().get_id()
            masks_all = [np.zeros(visual_id_seg.shape, dtype=np.bool) for _ in range(3)]
            for visual_id in self.handle_visual_ids:
                masks_all[0] = masks_all[0] | (actor_id == visual_id)
            masks_all[1] = masks_all[1] | (actor_id_seg == actor_id)
            for actor_id in self.robot_link_ids:
                masks_all[2] = masks_all[2] | (actor_id_seg == actor_id)
            seg_all[j.get_child_link().name] = np.stack(masks_all, axis=-1)

        view_dict["seg"] = np.stack(masks, axis=-1)
        self.seg_all = seg_all

    def get_state(self, with_controller_state=True):
        actors, arts = self.get_all_objects_in_state()

        actors_state = [get_actor_state(actor) for actor in actors]
        arts_state = [get_pad_articulation_state(art, max_dof) for art, max_dof in arts]

        return np.concatenate(
            actors_state
            + arts_state
            + [
                self.get_additional_task_info(obs_mode="state"),
                self.agent.get_state(with_controller_state=with_controller_state),
            ]
        )

    def set_state(self, state):
        # set actors
        actors, arts = self.get_all_objects_in_state()
        for actor in actors:
            actor.set_pose(pose=Pose(p=state[:3], q=state[3:7]))
            actor.set_velocity(state[7:10])
            actor.set_angular_velocity(state[10:13])
            state = state[13:]

        # set articulations
        for art, max_dof in arts:
            art.set_root_pose(Pose(state[0:3], state[3:7]))
            art.set_root_velocity(state[7:10])
            art.set_root_angular_velocity(state[10:13])
            state = state[13:]
            # import pdb; pdb.set_trace()
            art.set_qpos(state[0 : art.dof])
            art.set_qvel(state[max_dof : max_dof + art.dof])
            state = state[2 * max_dof :]

        # skip task info
        task_info_len = len(self.get_additional_task_info(obs_mode="state"))
        state = state[task_info_len:]

        # set robot state
        self.agent.set_state(state)

        return self.get_obs()

    @property
    def num_target_links(self):
        return super().num_target_links("prismatic")
