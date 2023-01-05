import os

import gym
import numpy as np
import torch
from h5py import File
from mani_skill_learn.env.env_utils import get_env_state
from mani_skill_learn.env.replay_buffer import ReplayMemory
from mani_skill_learn.utils.data import compress_size, flatten_dict, to_np, unsqueeze
from mani_skill_learn.utils.data.compression import compress_size
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

import flowbot3d.grasping.env  # noqa
from flowbot3d.baselines.bc_datagen_gt_flow_grasping import GLOBAL_PULL_VECTOR
from flowbot3d.baselines.gt_flow_grasping_ump_metric import max_flow_pt_calc_no_ransac

GLOBAL_ACT = None
GLOBAL_PULL_VECTOR = np.array([-1, 0, 0])


def dagger_data_collect(env, env_name):
    global GLOBAL_PULL_VECTOR
    obs = env.get_obs()
    data_to_store = {"obs": obs}
    env_state = get_env_state(env)
    for key in env_state:
        data_to_store[key] = env_state[key]
    data_to_store.update(env_state)

    flow = obs["pointcloud"]["flow"]
    max_flow_vec = flow[np.argmax(np.linalg.norm(flow, axis=1))]
    pull_vector = max_flow_vec
    T_org_pose = env.agent.robot.get_root_pose().to_transformation_matrix()
    T_pose_back = np.linalg.inv(T_org_pose)
    aux_T = T_pose_back
    action = np.zeros(8)
    action[0:3] = (
        # was 0.25
        1
        * (aux_T[:3, :3] @ pull_vector.reshape(3, 1)).squeeze()
        / np.linalg.norm(pull_vector)
    )
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

    pull_vector = pull_vector.reshape(1, 3)
    if abs(pull_vector[0, 1]) > abs(pull_vector[0, 2]):
        action[4] = 0.3 * np.sign(angle)
    elif abs(pull_vector[0, 2]) > abs(pull_vector[0, 1]):
        action[3] = 0.3 * np.sign(angle)
    GLOBAL_PULL_VECTOR = pull_vector

    # Step for data collection purpose
    next_obs, reward, done, info = env.step(action)

    # Query expert policy (oracle GT flow)
    data_to_store["actions"] = compress_size(action)
    data_to_store["next_obs"] = compress_size(next_obs)
    data_to_store["rewards"] = compress_size(reward)
    data_to_store["dones"] = True
    data_to_store["episode_dones"] = done
    info = {"info": info}
    data_to_store.update(compress_size(to_np(flatten_dict(info))))
    env_state = get_env_state(env)
    for key in env_state:
        data_to_store[f"next_{key}"] = env_state[key]

    # Undo
    env.step(-action)

    return data_to_store


def bc_grasping_policy(
    env,
    max_flow_pt,
    agent,
    phase,
    phase_counter,
):

    obs = env.get_obs()
    state = obs["agent"]
    del obs["agent"]
    del obs["pointcloud"]["flow_all"]
    obs["state"] = state

    vel = np.linalg.norm(env.agent.get_ee_vels().mean(axis=0))
    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])
    T_org_pose = env.agent.robot.get_root_pose().to_transformation_matrix()
    aux_T = np.linalg.inv(T_org_pose)
    if phase == 1:

        if max_flow_pt is None:
            max_flow_pt = env.target_link.pose.p

        delta_T = (max_flow_pt - ee_center) / np.linalg.norm(max_flow_pt - ee_center)
        # delta_R = 0.2 * R.from_matrix(T_robot_goal[:3, :3]).as_euler("xyz")
        action = np.zeros(8)
        print(ee_center)

        # if gripper_horizontal and env.agent.robot.get_qpos()[5] < np.pi / 2:
        if env.agent.robot.get_qpos()[5] < np.pi / 2:
            action[5] = 1
            print("Rotating Gripper")
        if not np.isclose(max_flow_pt, ee_center, atol=0.1).all():
            print("MOVING EE")
            action[:3] = aux_T[:3, :3] @ delta_T
            print(action)
            vel = np.linalg.norm(env.agent.get_ee_vels().mean(axis=0))
            if vel < 1e-2:
                print(vel)
                print(phase_counter)
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
            print("EE HOLD")
            if phase_counter >= 10:
                phase = 2
                phase_counter = 0
            else:
                if not np.isclose(max_flow_pt, ee_center).all():
                    action[:3] = aux_T[:3, :3] @ delta_T
                phase_counter += 1
        return action, phase, phase_counter, None

    dr = None
    if phase == 2:
        with torch.no_grad():
            action = agent(unsqueeze(obs, axis=0)).to("cpu").numpy().squeeze()
        print("\ndrive created\n")
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
    print(action[:3])
    print(phase, phase_counter)
    print(vel)
    GLOBAL_ACT = action
    return action, phase, phase_counter, dr


def run(env_name, level, cfg, bc_model, save_video, door, ajar, result_dir, dagger):
    import math

    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="sparse")
    env.reset(level=level)
    bc_agent = load_model(cfg, bc_model, env_name)

    if door:
        # Special handling of objects with convex hull issues
        r = R.from_euler("x", 360, degrees=True)
        rot_mat = r.as_matrix()
        r = r.as_quat()
        env.cabinet.set_root_pose(Pose(env.cabinet.get_root_pose().p, r))

    if ajar:
        qpos = []
        for joint in env.cabinet.get_active_joints():
            [[lmin, lmax]] = joint.get_limits()
            if joint.name == env.target_joint.name:
                qpos.append(lmin + 0.2 * (lmax - lmin))
            else:
                qpos.append(lmin)
        env.cabinet.set_qpos(np.array(qpos))

    for i, j in enumerate(env.cabinet.get_active_joints()):
        if j.get_child_link().name == env.target_link.name:
            q_limit = env.cabinet.get_qlimits()[i][1]
            q_idx = i
            qpos_init = env.cabinet.get_qpos()[i]
            total_qdist = abs(q_limit - qpos_init)
            break

    if dagger:
        datagen_traj = os.path.join(os.getcwd(), "umpnet_dagger_demo_traj")
        if not os.path.exists(datagen_traj):
            os.mkdir(datagen_traj)
        datagen_traj = os.path.join(os.getcwd(), "umpnet_dagger_demo_traj", env_name)
        if not os.path.exists(datagen_traj):
            os.mkdir(datagen_traj)
        trajectory_path = os.path.join(datagen_traj, "trajectory.h5")
        h5_file = File(trajectory_path, "w")

    pcd = env.get_obs()["pointcloud"]["xyz"]
    pcd = pcd[np.where(pcd[:, 2] > 1e-2)]
    filter_2 = np.random.permutation(np.arange(pcd.shape[0]))[:1200]
    pcd = pcd[filter_2]

    phase = 1

    # Stepping in gym
    phase_counter = 0

    video_writer = None
    data_episode = None
    knob_pts, max_flow_knob_pt, chain, flow_vec = max_flow_pt_calc_no_ransac(
        env, env_name, door, 1
    )

    for i in range(150):
        # env.render("human")
        if save_video:
            image = env.render(mode="color_image")["world"]["rgb"]
            image = image[..., ::-1]
            if video_writer is None:
                import cv2

                video_file = result_dir
                video_file = os.path.join(os.getcwd(), result_dir, f"{env_name}.mp4")
                video_writer = cv2.VideoWriter(
                    video_file,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (image.shape[1], image.shape[0]),
                )
            video_writer.write(np.uint8(image * 255))

        action, phase, phase_counter, dr = bc_grasping_policy(
            env, max_flow_knob_pt, bc_agent, phase, phase_counter
        )

        if dagger and phase == 2:
            data_to_store = dagger_data_collect(env, env_name)
            if data_episode is None:
                data_episode = ReplayMemory(150)
            data_episode.push(**data_to_store)

        obs, reward, done, info = env.step(action)
        if dr:
            env._scene.remove_drive(dr)
        # UMPNet Metrics
        if not math.isnan(env.cabinet.get_qpos()[0]):
            print("curr qpos:", env.cabinet.get_qpos()[q_idx])
            print("qlimit: ", q_limit)
            norm_dist = abs(q_limit - env.cabinet.get_qpos()[q_idx]) / (
                1e-6 + total_qdist
            )
            print("DIST: ", norm_dist)
            if np.isclose(norm_dist, 0.0, atol=1e-5):
                print("SUCCESS")
                break
    if dagger and phase == 2:
        group = h5_file.create_group(f"traj_0")
        data_episode.to_h5(group, with_traj_index=False)
    env.close()
    if not math.isnan(env.cabinet.get_qpos()[0]):
        return norm_dist
    else:
        return 1 - int(info["eval_info"]["success"])


def load_model(cfg, model_path, env_name):
    from mani_skill_learn.env import get_env_info
    from mani_skill_learn.methods.builder import build_brl
    from mani_skill_learn.utils.torch import load_checkpoint

    cfg.env_cfg["env_name"] = env_name
    obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
    cfg.agent["obs_shape"] = obs_shape
    cfg.agent["action_shape"] = action_shape
    cfg.agent["action_space"] = action_space

    agent = build_brl(cfg.agent)
    load_checkpoint(agent, model_path, map_location="cpu")
    agent.to("cuda")
    agent.eval()
    return agent


if __name__ == "__main__":

    import argparse
    import json
    import re
    import shutil
    from collections import defaultdict

    from mani_skill_learn.utils.meta import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="bc_mem")
    parser.add_argument("--ind_start", type=int, default="0")
    parser.add_argument("--ind_end", type=int, default="119")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dagger", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    model_name = args.model_name
    dagger = args.dagger
    if dagger:
        result_dir = os.path.join(
            os.getcwd(), "umpmetric_results_master", args.model_name + "_oracle"
        )
    else:
        result_dir = os.path.join(
            os.getcwd(), "umpmetric_results_master", args.model_name + "_oracle"
        )
    debug = args.debug

    if "flow" in model_name:
        config = f"/home/{os.getlogin()}/ManiSkill-Learn/configs/bc/mani_skill_point_cloud_transformer_flow.py"
    else:
        config = f"/home/{os.getlogin()}/ManiSkill-Learn/configs/bc/mani_skill_point_cloud_transformer.py"
    if "dagger" in model_name:
        bc_model = f"/home/{os.getlogin()}/ManiSkill-Learn/baselines_comparisons_results/{model_name}/BC/models/model_10000.ckpt"
    else:
        bc_model = f"/home/{os.getlogin()}/ManiSkill-Learn/baselines_comparisons_results/{model_name}/BC/models/model_100000.ckpt"
    if debug:
        cfg = Config.fromfile(config)
        run(
            "OpenCabinetDrawerGripper_{}_link_0-v0".format("46549"),
            0,
            cfg,
            bc_model,
            True,
            False,
            False,
            result_dir,
            dagger,
        )
        exit(0)

    if not os.path.exists(result_dir):
        print("Creating result directory")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(os.path.join(result_dir, "succ"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "fail"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "succ_unseen"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "fail_unseen"), exist_ok=True)
        print("Directory created")
    if mode == "train":
        succ_res_dir = os.path.join(result_dir, "succ")
        fail_res_dir = os.path.join(result_dir, "fail")
    else:
        succ_res_dir = os.path.join(result_dir, "succ_unseen")
        fail_res_dir = os.path.join(result_dir, "fail_unseen")

    start_ind = args.ind_start
    end_ind = args.ind_end
    model_name = args.model_name

    bad_doors = [
        "8930",
        "9003",
        "9016",
        "9107",
        "9164",
        "9168",
        "9386",
        "9388",
        "9410",
    ]
    ump_split_f = open(os.path.join(os.getcwd(), "umpnet_data_split.json"))
    data = json.load(ump_split_f)
    classes = data[mode].keys()
    results = defaultdict(list)
    file = open(
        os.path.join(os.getcwd(), "umpnet_obj_splits/{}_test_split.txt".format(mode)),
        "r",
    )
    available_envs = []
    for i in file.readlines():
        available_envs.append(i)
    for num, i in enumerate(available_envs[start_ind:end_ind]):
        i = i[:-1]
        for cl in classes:
            idx = [m.start() for m in re.finditer("_", i)]
            env_id = i[idx[0] + 1 : idx[1]]
            if env_id in data[mode][cl]["test"]:
                temp = cl
        ajar = False
        if not (
            temp != "Door"
            and temp != "Box"
            and temp != "Table"
            and temp != "Phone"
            and temp != "Bucket"
        ):
            ajar = False

        print("Executing task for env: ", i)
        print("Index: ", num + start_ind)
        bd = False
        for d in bad_doors:
            if d in i:
                bd = True
                break
        cfg = Config.fromfile(config)
        if bd:
            succ = run(i, 0, cfg, bc_model, True, True, ajar, result_dir, dagger)
        else:
            succ = run(i, 0, cfg, bc_model, True, False, ajar, result_dir, dagger)

        if dagger and os.path.isfile(
            f"/home/{os.getlogin()}/flowbot3d/umpnet_dagger_demo_traj/{i}/trajectory.h5"
        ):
            cwd = os.getcwd()
            os.chdir(f"/home/{os.getlogin()}/ManiSkill-Learn")
            trans_cmd = f"python tools/convert_demo_pcd.py --max-num-traj=-1 --env-name={i} --traj-name=/home/{os.getlogin()}/discriminative_embeddings/umpnet_dagger_demo_traj/{i}/trajectory.h5 --output-name=./umpnet_dagger_demo_traj/{i}/traj_pcd.h5 --obs-mode pointcloud"
            os.system(trans_cmd)
            os.chdir(cwd)
            shutil.rmtree(
                f"/home/{os.getlogin()}/flowbot3d/umpnet_dagger_demo_traj/{i}"
            )

        if os.path.isfile(os.path.join(succ_res_dir, "{}.mp4".format(i))):
            print("found duplicates")
            os.remove(os.path.join(succ_res_dir, "{}.mp4".format(i)))
        if os.path.isfile(os.path.join(fail_res_dir, "{}.mp4".format(i))):
            print("found duplicates")
            os.remove(os.path.join(fail_res_dir, "{}.mp4".format(i)))

        if succ < 0.1:
            if os.path.isfile(os.path.join(result_dir, "{}.mp4".format(i))):
                shutil.move(
                    os.path.join(result_dir, "{}.mp4".format(i)),
                    os.path.join(succ_res_dir, "{}.mp4".format(i)),
                )
        else:
            if os.path.isfile(os.path.join(result_dir, "{}.mp4".format(i))):
                shutil.move(
                    os.path.join(result_dir, "{}.mp4".format(i)),
                    os.path.join(fail_res_dir, "{}.mp4".format(i)),
                )

        for cl in classes:
            idx = [m.start() for m in re.finditer("_", i)]
            env_id = i[idx[0] + 1 : idx[1]]
            if env_id in data[mode][cl]["test"]:
                print("Class: ", cl)
                print(succ)
                results[cl].append(int(succ))
                if mode == "test":
                    res_file = open(os.path.join(result_dir, "test_test_res.txt"), "a")
                else:
                    res_file = open(os.path.join(result_dir, "train_test_res.txt"), "a")
                print("{}: {}".format(cl, succ), file=res_file)
                res_file.close()
