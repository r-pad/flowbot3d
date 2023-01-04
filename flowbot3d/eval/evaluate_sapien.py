import math
import os
import sys

import gym
import numpy as np
import torch
from part_embedding.flow_prediction.animate import FlowNetAnimation
from part_embedding.flow_prediction.latest_models import load_model

# from mani_skill_learn.env.observation_process import process_mani_skill_base
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

import flowbot3d.grasping.env  # noqa

GLOBAL_PULL_VECTOR = np.array([[-1, 0, 0]])


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
    flow_model,
    animation,
    cam_frame,
):

    obs = env.get_obs()
    # Define primitives
    action = np.zeros(8)
    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])
    global GLOBAL_PULL_VECTOR
    # Phase 1: Grasp
    if phase == 1:

        delta_T = (max_flow_pt - ee_center) / np.linalg.norm(max_flow_pt - ee_center)
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
            dr.set_x_properties(stiffness=45000, damping=0)
            dr.set_y_properties(stiffness=45000, damping=0)
            dr.set_z_properties(stiffness=45000, damping=0)
            _, pull_vector = max_flow_pt_calc(
                env, env_name, flow_model, 1, phase_counter, animation, cam_frame
            )
            pull_vector = pull_vector.reshape(1, 3)
            print("pull vec: ", pull_vector)
            if pull_vector is not None:
                angle = np.dot(
                    pull_vector.reshape(
                        3,
                    )
                    / np.linalg.norm(pull_vector),
                    GLOBAL_PULL_VECTOR.reshape(
                        3,
                    )
                    / (1e-7 + np.linalg.norm(GLOBAL_PULL_VECTOR)),
                )

            else:
                pull_vector = GLOBAL_PULL_VECTOR
                angle = 1

            angle = abs(np.arccos(angle))
            v = np.cross(
                GLOBAL_PULL_VECTOR.reshape(
                    3,
                )
                / (1e-7 + np.linalg.norm(GLOBAL_PULL_VECTOR)),
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
            GLOBAL_PULL_VECTOR = pull_vector
            action[0:3] = (
                1
                * (aux_T[:3, :3] @ pull_vector.reshape(3, 1)).squeeze()
                / np.linalg.norm(pull_vector)
            )
            action[3:6] += 1e-5
            phase_counter += 1

        else:
            phase_counter += 1
            action[0] = 0
        print("PHASE 3 GRASPING AND BACKING UP")
        return action, phase, phase_counter, dr


def max_flow_pt_calc(env, env_name, model, top_k, id, animation, cam_frame=False):
    """For the initial grasping point selection"""
    obs = env.get_obs()
    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])

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

        pred_flow = model.predict(
            torch.from_numpy(xyz).to("cuda"),
            torch.from_numpy(mask_1[mask_meta]).float(),
        )
        pred_flow = pred_flow.cpu().numpy()

    else:
        cam_mat = env.cameras[1].sub_cameras[0].get_pose().to_transformation_matrix()
        pred_flow = model.predict(
            torch.from_numpy(pcd @ cam_mat[:3, :3] + cam_mat[:3, -1]).to("cuda"),
            torch.from_numpy(mask_1[mask_meta]).float(),
        )
        pred_flow = pred_flow.cpu().numpy()
        pred_flow = pred_flow @ np.linalg.inv(cam_mat)[:3, :3]

    flow_norm_allpts = np.linalg.norm(pred_flow, axis=1)
    flow_norm_allpts = np.divide(flow_norm_allpts, ee_to_pt_dist)
    max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]
    max_flow_pt = pcd[max_flow_idx]
    max_flow_vector = pred_flow[max_flow_idx]
    print("max_flow: ", max_flow_vector)
    if True:
        if animation:
            temp = animation.add_trace(
                torch.as_tensor(pcd),
                torch.as_tensor([pcd[mask_1[mask_meta]]]),
                torch.as_tensor(
                    [pred_flow[mask_1[mask_meta]] / np.linalg.norm(max_flow_vector)]
                ),
                "red",
            )
            animation_module.append_gif_frame(temp)

    return (
        max_flow_pt.reshape(
            3,
        ),
        max_flow_vector / np.linalg.norm(max_flow_vector),
    )


def flow_grasp_weighted_sum(flow_vecs, grasp_scores):
    """Weighted sum of the flow magnitude and graspability score"""
    assert flow_vecs.shape[0] == grasp_scores.shape[0]
    flow_mags = np.array([np.linalg.norm(f) for f in flow_vecs])
    weighted_sum = 0.2 * flow_mags / flow_mags.max() + 0.8 * grasp_scores
    return weighted_sum.argmax()


def run(
    env_name,
    level,
    save_video,
    door,
    flow_model,
    result_dir,
    animation,
    ajar,
    cam_frame,
):
    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="sparse")
    env.reset(level=level)
    # randomize_camera(env)

    if door:
        # Special handling of objects with convex hull issues
        r = R.from_euler("x", 360, degrees=True)
        rot_mat = r.as_matrix()
        r = r.as_quat()
        env.cabinet.set_root_pose(Pose(env.cabinet.get_root_pose().p, r))

    for i, j in enumerate(env.cabinet.get_active_joints()):
        if j.get_child_link().name == env.target_link.name:
            q_limit = env.cabinet.get_qlimits()[i][1]
            q_idx = i
            qpos_init = env.cabinet.get_qpos()[i]
            total_qdist = abs(q_limit - qpos_init)
            break

    if ajar:
        qpos = []
        for joint in env.cabinet.get_active_joints():
            [[lmin, lmax]] = joint.get_limits()
            if joint.name == env.target_joint.name:
                qpos.append(lmin + 0.2 * (lmax - lmin))
            # qpos.append(-1.1)
            else:
                qpos.append(lmin)
        env.cabinet.set_qpos(np.array(qpos))

    max_flow_knob_pt, flow_vec = max_flow_pt_calc(
        env, env_name, flow_model, 1, 0, animation, cam_frame
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

    pcd = env.get_obs()["pointcloud"]["xyz"]
    pcd = pcd[np.where(pcd[:, 2] > 0.1)]
    w2a_max_score_pt = max_flow_knob_pt

    gripper_horizontal = True
    phase = 1

    # Stepping in gym
    phase_counter = 0
    # trans_m_w_matrix = T_m_w
    T_org_pose = env.agent.robot.get_root_pose().to_transformation_matrix()
    T_pose_back = np.linalg.inv(T_org_pose)

    # Pull vector as the flow direction of the largest flow vector
    pull_vector = flow_vec

    video_writer = None

    for i in range(150):
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
            flow_model,
            animation,
            cam_frame,
        )
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
    env.close()
    if not math.isnan(env.cabinet.get_qpos()[0]):
        return norm_dist
    else:
        return 1 - int(info["eval_info"]["success"])


if __name__ == "__main__":
    import argparse
    import json
    import re
    import shutil
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="4_to_4_umpnet")
    parser.add_argument("--ind_start", type=int, default="0")
    parser.add_argument("--ind_end", type=int, default="0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ajar", action="store_true")
    parser.add_argument("--cf", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    ajar = args.ajar
    cam_frame = args.cf
    result_dir = os.path.join(os.getcwd(), "umpmetric_results_master", args.model_name)

    # Create directories
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

    if "no" in model_name:
        print("NO MASK")
        flow_model = load_model(model_name, False)
    else:
        flow_model = load_model(model_name, True)
    flow_model = flow_model.to("cuda")  # type: ignore

    debug = args.debug

    if debug:
        animation_module = FlowNetAnimation()
        run(
            "OpenCabinetDoorGripper_{}_link_0-v0".format("101377"),
            0,
            True,
            False,
            flow_model,
            result_dir,
            animation_module,
            ajar,
            cam_frame,
        )
        save_html = animation_module.animate()
        if save_html:
            save_html.write_html("debug.html")
    else:
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
            os.path.join(
                os.getcwd(), "umpnet_obj_splits/{}_test_split.txt".format(mode)
            ),
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
            if not (
                temp != "Door"
                and temp != "Box"
                and temp != "Table"
                and temp != "Phone"
                and temp != "Bucket"
            ):
                ajar = True
            if temp != "Box":
                continue
            print("Executing task for env: ", i)
            print("Index: ", num + start_ind)
            bd = False
            for d in bad_doors:
                if d in i:
                    bd = True
                    break
            animation_module = FlowNetAnimation()
            if bd:
                succ = run(
                    i,
                    0,
                    True,
                    True,
                    flow_model,
                    result_dir,
                    animation_module,
                    ajar,
                    cam_frame,
                )
            else:
                succ = run(
                    i,
                    0,
                    True,
                    False,
                    flow_model,
                    result_dir,
                    animation_module,
                    ajar,
                    cam_frame,
                )
            save_html = animation_module.animate()

            if os.path.isfile(
                os.path.join(succ_res_dir, "{}.mp4".format(i))
            ) and os.path.isfile(os.path.join(succ_res_dir, "{}.html".format(i))):
                print("found duplicates")
                os.remove(os.path.join(succ_res_dir, "{}.mp4".format(i)))
                os.remove(os.path.join(succ_res_dir, "{}.html".format(i)))
            if os.path.isfile(
                os.path.join(fail_res_dir, "{}.mp4".format(i))
            ) and os.path.isfile(os.path.join(succ_res_dir, "{}.html".format(i))):
                print("found duplicates")
                os.remove(os.path.join(fail_res_dir, "{}.mp4".format(i)))
                os.remove(os.path.join(fail_res_dir, "{}.html".format(i)))

            if succ < 0.1:
                if os.path.isfile(os.path.join(result_dir, "{}.mp4".format(i))):
                    shutil.move(
                        os.path.join(result_dir, "{}.mp4".format(i)),
                        os.path.join(succ_res_dir, "{}.mp4".format(i)),
                    )
                    if save_html:
                        save_html.write_html(
                            os.path.join(succ_res_dir, "{}.html".format(i))
                        )
            else:
                if os.path.isfile(os.path.join(result_dir, "{}.mp4".format(i))):
                    shutil.move(
                        os.path.join(result_dir, "{}.mp4".format(i)),
                        os.path.join(fail_res_dir, "{}.mp4".format(i)),
                    )
                    if save_html:
                        save_html.write_html(
                            os.path.join(fail_res_dir, "{}.html".format(i))
                        )

            for cl in classes:
                idx = [m.start() for m in re.finditer("_", i)]
                env_id = i[idx[0] + 1 : idx[1]]
                if env_id in data[mode][cl]["test"]:
                    print("Class: ", cl)
                    print(succ)
                    results[cl].append(int(succ))
                    if mode == "test":
                        res_file = open(
                            os.path.join(result_dir, "test_test_res.txt"), "a"
                        )
                    else:
                        res_file = open(
                            os.path.join(result_dir, "train_test_res.txt"), "a"
                        )
                    print("{}: {}".format(cl, succ), file=res_file)
                    res_file.close()
