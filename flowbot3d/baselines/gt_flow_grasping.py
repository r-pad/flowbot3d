import os

import gym
import numpy as np
import pyransac3d as pyrsc
from trimesh import viewer  # noqa

import flowbot3d.grasping.env  # noqa
from flowbot3d.datasets.calc_art import compute_new_points
from flowbot3d.datasets.pm.pm_raw import parse_urdf_from_string


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

    p_world_flowedpts = compute_new_points(
        seg_pcd, np.eye(4), chain, org_config, target_config
    )

    flow = p_world_flowedpts - seg_pcd

    return p_world_flowedpts, flow, seg_pcd, seg_clr


def flow_w2a(flow_art, chain, magnitude):
    org_config = np.zeros(len(chain))
    target_config = np.ones(len(chain)) * magnitude

    p_world_flowedpts = compute_new_points(
        flow_art, np.eye(4), chain, org_config, target_config
    )

    flow = p_world_flowedpts - flow_art

    return flow


def vanilla_grasping_policy(
    env,
    max_flow_pt,
    pull_vector,
    phase,
    phase_counter,
    aux_T,
    gripper_horizontal,
    env_name,
):

    GLOBAL_PULL_VECTOR = np.array([-1, 0, 0])
    obs = env.get_obs()
    # Define primitives
    action = np.zeros(8)

    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])
    hand_rpy = env.agent.robot.get_qpos()[3:6]

    # Phase 1: Grasp
    if phase == 1:

        delta_T = (
            0.5 * (max_flow_pt - ee_center) / np.linalg.norm(max_flow_pt - ee_center)
        )
        # delta_R = 0.2 * R.from_matrix(T_robot_goal[:3, :3]).as_euler("xyz")
        action = np.zeros(8)
        print(ee_center)

        if gripper_horizontal and env.agent.robot.get_qpos()[5] < np.pi / 2:
            action[5] = 1
            print("Rotating Gripper")
        if not np.isclose(max_flow_pt, ee_center, atol=0.05).all():
            print("MOVING EE")
            action[:3] = aux_T[:3, :3] @ delta_T
            action[-2:] = [0.3, 0.3]
            print(action)
        else:
            print("EE HOLD")
            action[6] = 0.3
            action[7] = 0.3
            if phase_counter >= 50:
                phase = 2
                phase_counter = 0
            else:
                if not np.isclose(max_flow_pt, ee_center).all():
                    action[:3] = aux_T[:3, :3] @ delta_T
                phase_counter += 1
        return action, phase, phase_counter
    # Phase 2: Back Up
    else:
        action = np.zeros(8)
        action[-2:] = [-0.3, -0.3]
        if phase_counter >= 50:
            _, _, _, pull_vector = max_flow_pt_calc(env, env_name, 1)
            print(GLOBAL_PULL_VECTOR)
            if pull_vector is not None:
                GLOBAL_PULL_VECTOR = pull_vector
            else:
                pull_vector = GLOBAL_PULL_VECTOR
            action[0:3] = (
                0.15
                * (aux_T[:3, :3] @ pull_vector.reshape(3, 1)).squeeze()
                / np.linalg.norm(pull_vector)
            )
            # Adjust the angle
            if len(pull_vector.shape) > 1:
                print(pull_vector[:, 1])
                if pull_vector[:, 1] > 1e-3:
                    action[4] = 0.08
                elif pull_vector[:, 1] < -1e-3:
                    action[4] = -0.08
            else:
                if pull_vector[1] > 1e-3:
                    action[4] = 0.08
                elif pull_vector[1] < -1e-3:
                    action[4] = -0.08
        else:
            phase_counter += 1
            action[0] = 0
        print("PHASE 3 GRASPING AND BACKING UP")
        return action, phase, phase_counter


def max_flow_pt_calc(env, env_name, top_k=1):
    obs = env.get_obs()
    urdf = parse_urdf_from_string(env.cabinet.export_urdf())
    chain = urdf.get_chain(env.target_link.name)
    _, flow_test, seg_pcd, seg_clr = transform_pcd(obs, chain, 0.3)
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


def flow_grasp_weighted_sum(flow_vecs, grasp_scores):
    """Weighted sum of the flow magnitude and graspability score"""
    assert flow_vecs.shape[0] == grasp_scores.shape[0]
    flow_mags = np.array([np.linalg.norm(f) for f in flow_vecs])
    weighted_sum = 0.2 * flow_mags / flow_mags.max() + 0.8 * grasp_scores
    return weighted_sum.argmax()


def run(env_name, level, save_video):

    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="dense")
    env.reset(level=level)

    T_o_org = env.agent.robot.get_root_pose().to_transformation_matrix()
    knob_pts, max_flow_knob_pt, chain, flow_vec = max_flow_pt_calc(env, env_name, 1)
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

    for i in range(280):
        if save_video:
            env.render("human")
        action, phase, phase_counter = vanilla_grasping_policy(
            env,
            w2a_max_score_pt,
            pull_vector,
            phase,
            phase_counter,
            T_pose_back,
            gripper_horizontal,
            env_name,
        )
        if False:
            image = env.render(mode="color_image")["world"]["rgb"]
            image = image[..., ::-1]
            if video_writer is None:
                import cv2

                video_file = os.path.join(os.getcwd(), f"{env_name}.mp4")
                video_writer = cv2.VideoWriter(
                    video_file,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (image.shape[1], image.shape[0]),
                )
            video_writer.write(np.uint8(image * 255))
        obs, reward, done, info = env.step(action)
        if info["eval_info"]["success"]:
            print("SUCCESS")
            break
    env.close()
    return info["eval_info"]["success"]


if __name__ == "__main__":
    run("OpenCabinetDoorGripper_7290_link_0-v0", 0, True)
