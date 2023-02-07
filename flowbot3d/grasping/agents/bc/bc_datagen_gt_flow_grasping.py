import os
import shutil

import gym
import numpy as np
import pyransac3d as pyrsc
from h5py import File
from mani_skill_learn.env.env_utils import get_env_state, true_done
from mani_skill_learn.env.replay_buffer import ReplayMemory
from mani_skill_learn.utils.data import compress_size, flatten_dict, to_np
from mani_skill_learn.utils.data.compression import compress_size
from rpad.partnet_mobility_utils.articulate import articulate_points
from rpad.partnet_mobility_utils.urdf import PMTree

import flowbot3d.grasping.env  # noqa


def transform_pcd(obs, chain, magnitude):
    """Transform the PCD observed in ManiSkill Env"""
    pcd = obs["pointcloud"]["xyz"]
    clr = obs["pointcloud"]["rgb"]
    seg = obs["pointcloud"]["seg"][:, :-1]

    # Get the seg pcd.
    seg_pcd = pcd[np.where(seg)[0], :]
    seg_clr = pcd[np.where(clr)[0], :]
    org_config = np.zeros(len(chain))
    target_config = np.ones(len(chain)) * magnitude

    p_world_flowedpts = articulate_points(
        seg_pcd, np.eye(4), chain, org_config, target_config
    )

    flow = p_world_flowedpts - seg_pcd

    return p_world_flowedpts, flow, seg_pcd, seg_clr


def flow_w2a(flow_art, chain, magnitude):
    org_config = np.zeros(len(chain))
    target_config = np.ones(len(chain)) * magnitude

    p_world_flowedpts = articulate_points(
        flow_art, np.eye(4), chain, org_config, target_config
    )

    flow = p_world_flowedpts - flow_art

    return flow


GLOBAL_PULL_VECTOR = np.array([-1, 0, 0])


def vanilla_grasping_policy(
    env,
    max_flow_pt,
    pull_vector,
    phase,
    phase_counter,
    aux_T,
    gripper_horizontal,
    gripper_vertical,
    env_name,
):
    obs = env.get_obs()
    # Define primitives
    action = np.zeros(8)
    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])
    global GLOBAL_PULL_VECTOR
    # Phase 1: Grasp
    if phase == 1:
        delta_T = (
            0.5 * (max_flow_pt - ee_center) / np.linalg.norm(max_flow_pt - ee_center)
        )
        # delta_R = 0.2 * R.from_matrix(T_robot_goal[:3, :3]).as_euler("xyz")
        action = np.zeros(8)
        print(ee_center)

        # if gripper_horizontal and env.agent.robot.get_qpos()[5] < np.pi / 2:
        if gripper_vertical and env.agent.robot.get_qpos()[3] > -np.pi / 2:
            action[3] = -1
            print("Rotating Gripper")
        if env.agent.robot.get_qpos()[5] < np.pi / 2:
            action[5] = 1
            print("Rotating Gripper")
        if not np.isclose(max_flow_pt, ee_center, atol=0.1).all():
            print("MOVING EE")
            action[:3] = aux_T[:3, :3] @ delta_T
            print(action)
        else:
            print("EE HOLD")
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
        dr = None
        if phase_counter >= 10:
            scene = env._scene
            dr = scene.create_drive(
                env.agent.robot.get_links()[-1],
                env.agent.robot.get_links()[-1].get_cmass_local_pose(),
                env.target_link,
                env.target_link.get_cmass_local_pose(),
            )
            dr.set_x_properties(stiffness=25000, damping=0)
            dr.set_y_properties(stiffness=25000, damping=0)
            dr.set_z_properties(stiffness=25000, damping=0)
            _, _, _, pull_vector = max_flow_pt_calc_no_ransac(env, env_name, 1)
            if pull_vector is not None:
                angle = np.dot(
                    pull_vector.reshape(
                        3,
                    )
                    / np.linalg.norm(pull_vector),
                    GLOBAL_PULL_VECTOR.reshape(
                        3,
                    )
                    / (1e-4 + np.linalg.norm(GLOBAL_PULL_VECTOR)),
                )

            else:
                pull_vector = GLOBAL_PULL_VECTOR
                angle = 1

            angle = abs(np.arccos(angle))
            v = np.cross(
                GLOBAL_PULL_VECTOR.reshape(
                    3,
                ),
                pull_vector.reshape(
                    3,
                ),
            )
            v = v / np.linalg.norm(v + 1e-4)
            if abs(pull_vector[0]).argmin() == 2:
                vn = np.array([0, 0, 1])
            elif abs(pull_vector[0]).argmin() == 1:
                vn = np.array([0, 1, 0])
            elif abs(pull_vector[0]).argmin() == 0:
                vn = np.array([1, 0, 0])
            if np.dot(vn, v) > 0:
                angle = -angle

            if abs(pull_vector[0, 1]) > abs(pull_vector[0, 2]):
                action[4] = 0.3 * np.sign(angle)
            elif abs(pull_vector[0, 2]) > abs(pull_vector[0, 1]):
                action[3] = 0.3 * np.sign(angle)
            GLOBAL_PULL_VECTOR = pull_vector
            action[0:3] = (
                # was 0.25
                1
                * (aux_T[:3, :3] @ pull_vector.reshape(3, 1)).squeeze()
                / np.linalg.norm(pull_vector)
            )
            print(pull_vector)
            print(action[3:6])
            phase_counter += 1
        else:
            phase_counter += 1
            action[0] = 0
        print("PHASE 3 GRASPING AND BACKING UP")
        return action, phase, phase_counter, dr


def max_flow_pt_calc(env, env_name, top_k=1):
    obs = env.get_obs()
    urdf = PMTree.parse_urdf_from_string(env.cabinet.export_urdf())
    chain = urdf.get_chain(env.target_link.name)
    _, flow_test, seg_pcd, seg_clr = transform_pcd(obs, chain, 0.1)
    flow_norm_allpts = np.linalg.norm(flow_test, axis=1)
    max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]
    ransac_plane = pyrsc.Plane()
    coeffs, inliers = ransac_plane.fit(seg_pcd, 0.01)
    all_idx = np.arange(seg_pcd.shape[0])

    # Get the indices that correspond to the knob
    seg_knob_idx_temp = np.delete(all_idx, inliers, 0)

    # Further filter out the points behind the door
    avg_plane_x = np.mean(seg_pcd[inliers, 0])
    seg_knob_idx = []
    for i in seg_knob_idx_temp:
        if avg_plane_x > seg_pcd[i, 0]:
            seg_knob_idx.append(i)
    top_k = min(top_k, len(seg_knob_idx))
    # Find the index of max flow on the knob
    seg_knob_idx = np.array(seg_knob_idx)
    if len(seg_knob_idx) == 0:
        return None, None, chain, None
    max_flow_knob_idx_temp = np.argpartition(flow_norm_allpts[seg_knob_idx], -top_k)[
        -top_k:
    ]

    # Map the argmax back to the original seg pcd
    max_flow_knob_pt = np.mean(seg_pcd[seg_knob_idx[max_flow_knob_idx_temp], :], axis=0)
    max_flow_knob_vector = flow_test[seg_knob_idx[max_flow_knob_idx_temp]]
    knob_pts = seg_pcd[seg_knob_idx]

    return (knob_pts, max_flow_knob_pt, chain, max_flow_knob_vector)


def max_flow_pt_calc_no_ransac(env, env_name, top_k=1):
    obs = env.get_obs()
    urdf = PMTree.parse_urdf_from_string(env.cabinet.export_urdf())
    chain = urdf.get_chain(env.target_link.name)
    _, flow_test, seg_pcd, seg_clr = transform_pcd(obs, chain, 0.1)
    flow_norm_allpts = np.linalg.norm(flow_test, axis=1)
    max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]

    # Map the argmax back to the original seg pcd
    max_flow_knob_pt = seg_pcd[max_flow_idx, :]
    max_flow_knob_vector = flow_test[max_flow_idx]
    knob_pts = seg_pcd
    if not max_flow_knob_pt.any():
        return None, None, None, None

    return (knob_pts, max_flow_knob_pt[0], chain, max_flow_knob_vector)


def flow_grasp_weighted_sum(flow_vecs, grasp_scores):
    """Weighted sum of the flow magnitude and graspability score"""
    assert flow_vecs.shape[0] == grasp_scores.shape[0]
    flow_mags = np.array([np.linalg.norm(f) for f in flow_vecs])
    weighted_sum = 0.2 * flow_mags / flow_mags.max() + 0.8 * grasp_scores
    return weighted_sum.argmax()


def run(env_name, trials, save_video):
    datagen_traj = os.path.join(os.getcwd(), "umpnet_bc_demo_traj")
    if not os.path.exists(datagen_traj):
        os.mkdir(datagen_traj)
    datagen_traj = os.path.join(os.getcwd(), "umpnet_bc_demo_traj", env_name)
    if not os.path.exists(datagen_traj):
        os.mkdir(datagen_traj)
    trajectory_path = os.path.join(datagen_traj, "trajectory.h5")
    h5_file = File(trajectory_path, "w")
    for tr in range(trials):
        env = gym.make(env_name)
        env.set_env_mode(obs_mode="pointcloud", reward_type="dense")
        env.reset(level=tr)

        T_o_org = env.agent.robot.get_root_pose().to_transformation_matrix()
        knob_pts, max_flow_knob_pt, chain, flow_vec = max_flow_pt_calc_no_ransac(
            env, env_name, 1
        )
        if flow_vec is None:
            flow_vec = np.array([[0, 0, 0]])
        if abs(flow_vec[0, 2]) > abs(flow_vec[0, 1]) and abs(flow_vec[0, 2]) > abs(
            flow_vec[0, 0]
        ):
            gripper_vertical = True
        else:
            gripper_vertical = False
        print(gripper_vertical)
        if knob_pts is None:
            print("RANSAC FAILED")
            return False

        pcd = env.get_obs()["pointcloud"]["xyz"]
        pcd = pcd[np.where(pcd[:, 2] > 0.1)]
        w2a_max_score_pt = max_flow_knob_pt

        # Determine the alignment of the gripper (horizontal or vertical)
        knob_w = knob_pts[:, 1].max() - knob_pts[:, 1].min()
        knob_h = knob_pts[:, 2].max() - knob_pts[:, 2].min()
        gripper_horizontal = knob_w < knob_h
        phase = 1

        # Stepping in gym
        phase_counter = 0
        # trans_m_w_matrix = T_m_w
        T_org_pose = env.agent.robot.get_root_pose().to_transformation_matrix()
        T_pose_back = np.linalg.inv(T_org_pose)

        # Pull vector as the flow direction of the largest flow vector
        pull_vector = flow_vec

        video_writer = None
        recent_obs = env.reset()
        data_episode = None
        horizon = 150

        for i in range(150):
            """
            Initialize curr step saved data
            """
            data_to_store = {"obs": recent_obs}
            env_state = get_env_state(env)
            for key in env_state:
                data_to_store[key] = env_state[key]
            data_to_store.update(env_state)

            if save_video:
                # env.render("human")
                image = env.render(mode="color_image")["world"]["rgb"]
                image = image[..., ::-1]
                if video_writer is None:
                    import cv2

                    video_file = os.path.join(
                        os.getcwd(), "pm_obj_datagen_vid", f"{env_name}.mp4"
                    )
                    video_writer = cv2.VideoWriter(
                        video_file,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20,
                        (image.shape[1], image.shape[0]),
                    )
                video_writer.write(np.uint8(image * 255))
            action, phase, phase_counter, dr = vanilla_grasping_policy(
                env,
                w2a_max_score_pt,
                pull_vector,
                phase,
                phase_counter,
                T_pose_back,
                gripper_horizontal,
                gripper_vertical,
                env_name,
            )
            next_obs, reward, done, info = env.step(action)
            """
            Populate the curr step saved data
            """
            episode_done = done
            done = true_done(done, info)
            eval_info = info
            info = {"info": info}
            data_to_store["actions"] = compress_size(action)
            data_to_store["next_obs"] = compress_size(next_obs)
            data_to_store["rewards"] = compress_size(reward)
            data_to_store["dones"] = done
            data_to_store["episode_dones"] = episode_done
            data_to_store.update(compress_size(to_np(flatten_dict(info))))
            env_state = get_env_state(env)
            for key in env_state:
                data_to_store[f"next_{key}"] = env_state[key]

            # env.render('human')
            recent_obs = next_obs
            if data_episode is None:
                data_episode = ReplayMemory(horizon)
            data_episode.push(**data_to_store)
            if dr:
                env._scene.remove_drive(dr)
            if eval_info["eval_info"]["success"]:
                print("SUCCESS")
                break
        env.close()
        group = h5_file.create_group(f"traj_{tr}")
        data_episode.to_h5(group, with_traj_index=False)
        data_episode = None
    h5_file.close()


if __name__ == "__main__":
    import json
    import re

    ump_split_f = open(os.path.join(os.getcwd(), "umpnet_data_split.json"))
    mode = "train"
    data = json.load(ump_split_f)
    classes = data[mode].keys()
    file = open(
        os.path.join(os.getcwd(), "umpnet_obj_splits/{}_test_split.txt".format(mode)),
        "r",
    )
    start_ind = 125
    available_envs = []
    for i in file.readlines():
        available_envs.append(i)
    for num, i in enumerate(available_envs[start_ind:]):
        i = i[:-1]
        for cl in classes:
            idx = [m.start() for m in re.finditer("_", i)]
            env_id = i[idx[0] + 1 : idx[1]]
            if env_id in data[mode][cl]["test"]:
                temp = cl

    for num, i in enumerate(available_envs[start_ind:]):
        i = i[:-1]
        print(" Executing task for env: ", i)
        print("Index: ", num + start_ind)
        print("class: ", temp)
        run(i, 1, False)
        cwd = os.getcwd()
        os.chdir(f"/home/{os.getlogin()}/ManiSkill-Learn")
        trans_cmd = f"python tools/convert_demo_pcd.py --max-num-traj=-1 --env-name={i} --traj-name={os.getcwd()}/umpnet_bc_demo_traj/{i}/trajectory.h5 --output-name=./umpnet_pcd_demo_traj/{i}/traj_pcd.h5 --obs-mode pointcloud"
        os.system(trans_cmd)
        os.chdir(cwd)
        shutil.rmtree(f"{os.getcwd()}/umpnet_bc_demo_traj/{i}")
