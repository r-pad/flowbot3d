import math
import os
import sys

import gym
import numpy as np
import pybullet as p
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R
from trimesh import viewer  # noqa

import flowbot3d.grasping.env  # noqa
from flowbot3d.datasets.calc_art import compute_new_points
from flowbot3d.datasets.pm.pm_raw import parse_urdf_from_string


def transform_pcd(env, obs, chain, magnitude):
    """Transform the PCD observed in ManiSkill Env"""
    pcd = obs["pointcloud"]["xyz"]
    clr = obs["pointcloud"]["rgb"]
    seg = obs["pointcloud"]["seg"][:, :-1]
    seg_all = env.seg_all

    # Get the seg pcd.
    seg_pcd = pcd[np.where(seg)[0], :]
    seg_clr = pcd[np.where(clr)[0], :]
    org_config = np.zeros(len(chain))
    target_config = np.ones(len(chain)) * magnitude * 5

    p_world_flowedpts = compute_new_points(
        seg_pcd, np.eye(4), chain, org_config, target_config
    )

    flow = p_world_flowedpts - seg_pcd

    return p_world_flowedpts, flow, seg_pcd, seg_clr


def transform_pcd_door(env, obs, chain, magnitude):
    """
    Special case to deal with Door objects' convex hull issues
    """

    pcd = obs["pointcloud"]["xyz"]
    clr = obs["pointcloud"]["rgb"]
    seg = obs["pointcloud"]["seg"][:, :-1]
    rot_mat = env.cabinet.get_root_pose().to_transformation_matrix()
    rot_mat = np.linalg.inv(rot_mat)
    pcd = pcd @ rot_mat[:3, :3] + rot_mat[:3, -1]

    # Get the seg pcd.
    seg_pcd = pcd[np.where(seg)[0], :]
    seg_clr = pcd[np.where(clr)[0], :]
    org_config = np.zeros(len(chain))
    target_config = np.ones(len(chain)) * magnitude * 5

    p_world_flowedpts = compute_new_points(
        seg_pcd, np.eye(4), chain, org_config, target_config
    )

    flow = p_world_flowedpts - seg_pcd
    p_world_flowedpts = (
        p_world_flowedpts @ np.linalg.inv(rot_mat)[:3, :3]
        + np.linalg.inv(rot_mat)[:3, -1]
    )
    flow = flow @ np.linalg.inv(rot_mat)[:3, :3]
    seg_pcd = seg_pcd @ np.linalg.inv(rot_mat)[:3, :3] + np.linalg.inv(rot_mat)[:3, -1]

    return p_world_flowedpts, flow, seg_pcd, seg_clr


GLOBAL_PULL_VECTOR = np.array([[-1, 0, 0]])
GLOBAL_ANGLE = None


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
    global GLOBAL_ANGLE
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
            _, _, _, pull_vector = max_flow_pt_calc_no_ransac(env, env_name, 1)
            print(pull_vector)
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
            if phase_counter == 10:
                GLOBAL_ANGLE = action[3:6]
            GLOBAL_PULL_VECTOR = pull_vector
            action[0:3] = (
                10
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


def max_flow_pt_calc_no_ransac(env, env_name, door, top_k=1):
    obs = env.get_obs()
    # obs = process_mani_skill_base(obs, env)
    urdf = parse_urdf_from_string(env.cabinet.export_urdf())
    chain = urdf.get_chain(env.target_link.name)
    # urdf = parse_urdf(
    #     "./partnet-mobility-dataset/{}/mobility.urdf".format(env.selected_id))
    # chain = urdf.get_chain(env.target_link.name)
    # print(chain)
    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])
    if door:
        _, flow_test, seg_pcd, seg_clr = transform_pcd_door(env, obs, chain, 0.1)
    else:
        _, flow_test, seg_pcd, seg_clr = transform_pcd(env, obs, chain, 0.1)
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


def flow_grasp_weighted_sum(flow_vecs, grasp_scores):
    """Weighted sum of the flow magnitude and graspability score"""
    assert flow_vecs.shape[0] == grasp_scores.shape[0]
    flow_mags = np.array([np.linalg.norm(f) for f in flow_vecs])
    weighted_sum = 0.2 * flow_mags / flow_mags.max() + 0.8 * grasp_scores
    return weighted_sum.argmax()


def sample_az_ele(radius, az_lo, az_hi, ele_lo, ele_hi):
    """Sample random azimuth elevation pair and convert to cartesian."""
    azimuth = np.random.uniform(az_lo, az_hi)
    elevation = np.random.uniform(ele_lo, ele_hi)

    print("Sampled elevation: ", np.rad2deg(elevation))
    print("Sampled azimuth: ", np.rad2deg(azimuth))

    x = -radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation) * np.sin(azimuth)
    y = radius * np.cos(azimuth)

    return x, y, z, azimuth, elevation


def randomize_camera(env):
    """Randomize random camera viewpoints"""
    target = env.cabinet.get_pose().p
    can_cam_loc = env.cameras[1].sub_cameras[0].get_pose().p
    radius = np.linalg.norm(can_cam_loc - target)
    x, y, z, az, ele = sample_az_ele(
        radius, np.deg2rad(30), np.deg2rad(150), np.deg2rad(30), np.deg2rad(60)
    )
    eye = np.array([x, y, z])
    up = [0, 0, 1]
    view_list = p.computeViewMatrix(eye, target, up)
    T_camgl2world = np.asarray(view_list).reshape(4, 4).T
    T_world2camgl = np.linalg.inv(T_camgl2world)
    T_camgl2cam = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    T_world2cam = T_world2camgl @ T_camgl2cam
    if az > 90:
        Ttemp = R.from_euler("zyx", [90, 0, 0], degrees=True).as_matrix()
        T_world2cam[:3, :3] = Ttemp @ T_world2cam[:3, :3]
    else:
        Ttemp = R.from_euler("zyx", [90, 0, 180], degrees=True).as_matrix()
        T_world2cam[:3, :3] = Ttemp @ T_world2cam[:3, :3]

    new_loc = T_world2cam[:3, -1]
    new_rot = T_world2cam[:3, :3]

    can_cam_rot = env.cameras[1].sub_cameras[0].get_pose().q
    can_cam_rot = R.from_quat(can_cam_rot).as_matrix()
    rot_diff = np.linalg.inv(can_cam_rot) @ new_rot
    for i in range(3):
        cam = env.cameras[1].sub_cameras[i]
        if i == 0:
            rot_transformed = R.from_matrix(new_rot).as_quat()
            new_pose = Pose(new_loc, rot_transformed)
            cam.set_initial_pose(new_pose)
        else:
            curr_rot = cam.get_pose().q
            curr_rot = R.from_quat(curr_rot).as_matrix()
            rot_transformed = curr_rot @ rot_diff
            rot_transformed = R.from_matrix(rot_transformed).as_quat()
            new_pose = Pose(new_loc, rot_transformed)
            cam.set_initial_pose(new_pose)

    return


def run(env_name, level, save_video, door=False):

    from sapien.core import Pose

    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="sparse")
    env.reset(level=level)

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

    qpos = []
    for joint in env.cabinet.get_active_joints():
        [[lmin, lmax]] = joint.get_limits()
        if joint.name == env.target_joint.name:
            qpos.append(lmin + 0.1 * (lmax - lmin))
        # qpos.append(-1.1)
        else:
            qpos.append(lmin)
    env.cabinet.set_qpos(np.array(qpos))

    T_o_org = env.agent.robot.get_root_pose().to_transformation_matrix()
    knob_pts, max_flow_knob_pt, chain, flow_vec = max_flow_pt_calc_no_ransac(
        env, env_name, door, 1
    )
    if flow_vec is None:
        flow_vec = np.array([[0, 0, 0]])
    if abs(flow_vec[0, 2]) > abs(flow_vec[0, 1]) and abs(flow_vec[0, 2]) > abs(
        flow_vec[0, 0]
    ):
        gripper_vertical = True
    else:
        gripper_vertical = False
    if knob_pts is None:
        return False
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
        env.render("human")
        if save_video:
            image = env.render(mode="color_image")["world"]["rgb"]
            image = image[..., ::-1]
            if video_writer is None:
                import cv2

                video_file = os.path.join(
                    os.getcwd(), "umpmetric_exec_vid", f"{env_name}.mp4"
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
                return 0
    env.close()
    if not math.isnan(env.cabinet.get_qpos()[0]):
        return norm_dist
    else:
        return 1 - int(info["eval_info"]["success"])


if __name__ == "__main__":
    import json
    import re
    import shutil
    from collections import defaultdict

    # These are the doors with convex hull issues, where we transform the door first...
    bad_doors = ["8930", "9003", "9016", "9107", "9164", "9168", "9386", "9388", "9410"]

    # run("OpenCabinetDoorGripper_{}_link_0-v0".format("100243"), 0, False, False)
    ump_split_f = open("/home/flowbot3d/umpnet_data_split.json")
    data = json.load(ump_split_f)
    classes = data["test"].keys()
    results = defaultdict(list)
    file = open(
        f"/home/{os.getlogin()}/flowbot3d/umpnet_obj_splits/test_test_split.txt",
        "r",
    )
    available_envs = []
    for i in file.readlines():
        available_envs.append(i)
    for num, i in enumerate(available_envs):
        i = i[:-1]
        for cl in classes:
            idx = [m.start() for m in re.finditer("_", i)]
            env_id = i[idx[0] + 1 : idx[1]]
            if env_id in data["test"][cl]["test"]:
                temp = cl
        if temp != "Door":
            continue
        print("Executing task for env: ", i)
        print("Index: ", num)
        bd = False
        for d in bad_doors:
            if d in i:
                bd = True
                break
        if bd:
            succ = run(i, 0, True, True)
        else:
            succ = run(i, 0, True, False)

        if succ < 0.1:
            if os.path.isfile(
                f"/home/{os.getlogin()}/flowbot3d/umpmetric_exec_vid/{i}.mp4"
            ):
                os.remove(f"/home/{os.getlogin()}/flowbot3d/umpmetric_exec_vid/{i}.mp4")
            if os.path.isfile(
                f"/home/{os.getlogin()}/flowbot3d/umpmetric_exec_vid/fail/{i}.mp4"
            ):
                os.remove(
                    f"/home/{os.getlogin()}/flowbot3d/umpmetric_exec_vid/fail/{i}.mp4"
                )
        else:
            if os.path.isfile(
                f"/home/{os.getlogin()}/flowbot3d/umpmetric_exec_vid/{i}.mp4"
            ):
                shutil.move(
                    f"/home/{os.getlogin()}/flowbot3d/umpmetric_exec_vid/{i}.mp4",
                    f"/home/{os.getlogin()}/flowbot3d/umpmetric_exec_vid/fail/{i}.mp4",
                )
        for cl in classes:
            idx = [m.start() for m in re.finditer("_", i)]
            env_id = i[idx[0] + 1 : idx[1]]
            if env_id in data["test"][cl]["test"]:
                print("Class: ", cl)
                print(succ)
                results[cl].append(int(succ))
                res_file = open("umpmetric_gtflow_unseen_res.txt", "a")
                print("{}: {}".format(cl, succ), file=res_file)
                res_file.close()
