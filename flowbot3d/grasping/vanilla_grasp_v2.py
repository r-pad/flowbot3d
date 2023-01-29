import os

import gym
import mani_skill.env  # noqa
import numpy as np
import pyransac3d as pyrsc
import trimesh
from mani_skill.utils.osc import OperationalSpaceControlInterface
from pytransform3d import transformations
from rpad.partnet_mobility_utils.articulate import articulate_points
from rpad.partnet_mobility_utils.urdf import PMTree
from scipy.spatial.transform import Rotation as R


def transform_pcd(obs, chain, magnitude):
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


def vanilla_grasping_policy(
    env, max_flow_pt, gripper_horizontal, phase, phase_counter, rpy
):
    obs = env.get_obs()
    qpos = osc_interface.get_robot_qpos_from_obs(obs)
    hand_forward = np.zeros(osc_interface.osc_dim)
    extra_dim = len(osc_interface.osc_extra_joints)
    dim = 1  # move along x direction in end effector's frame
    action = env.action_space.sample() * 0
    base_loc = env.agent.get_base_link().get_pose().to_transformation_matrix()[:-1, -1]
    hand_forward[1] = 0.5
    hand_forward[4] = 0.5
    hand_forward[5] = 0.5
    forward_action_y_p = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, hand_forward, action[: osc_interface.null_space_dim]
    )
    hand_forward[1] = -0.5
    forward_action_y_n = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, hand_forward, action[: osc_interface.null_space_dim]
    )

    hand_forward[0] = 0.5
    hand_forward[1] = 0
    hand_forward[3] = 0.2
    forward_action_x_p = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, hand_forward, action[: osc_interface.null_space_dim]
    )
    hand_forward[0] = -0.5
    forward_action_x_n = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, hand_forward, action[: osc_interface.null_space_dim]
    )

    hand_forward[3] = 0.5
    hand_forward[0] = 0
    forward_action_z_p = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, hand_forward, action[: osc_interface.null_space_dim]
    )
    hand_forward[3] = -0.5
    forward_action_z_n = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, hand_forward, action[: osc_interface.null_space_dim]
    )
    hand_forward[3] = 0
    keep_open = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, hand_forward, action[: osc_interface.null_space_dim]
    )
    keep_close = osc_interface.operational_space_and_null_space_to_joint_space(
        qpos, -hand_forward, action[: osc_interface.null_space_dim]
    )

    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])
    # hand_xyz = env.agent.hand.get_pose().p
    # hand_rpy = R.from_quat(env.agent.hand.get_pose().q).as_matrix()
    hand_xyz = env.agent.robot.get_links()[15].pose.p
    hand_rpy = R.from_quat(env.agent.robot.get_links()[15].pose.q).as_matrix()
    base_to_flow_dist = np.linalg.norm(max_flow_pt - base_loc)
    ee_to_flow_pose_dist = max_flow_pt - ee_center

    # Phase 1: approach to the anchor point, base only
    if phase == 0:
        # First align with the target point in x direction
        if base_to_flow_dist >= 1.2:
            action = forward_action_x_p
            if gripper_horizontal and phase_counter < 30:
                print("ALIGNING GRIPPER")
                # If the hand joint is not "flat"
                action[10] = 0.3
                phase_counter += 1
            print("MOVING FORWARD")
        else:
            print("Phase 1 DONE")
            action = keep_open
            phase = 1
            phase_counter = 0
            hand_rpy = R.from_quat(env.agent.robot.get_links()[15].pose.q).as_matrix()
        return action, phase, phase_counter, hand_rpy

    # Phase 2: Grasp
    elif phase == 1:
        rot = hand_rpy
        T_world_robot = np.vstack(
            [np.hstack([rot, hand_xyz.reshape(3, 1)]), np.array([[0, 0, 0, 1]])]
        )
        T_robot_world = np.linalg.inv(T_world_robot)
        T_world_goal = np.vstack(
            [np.hstack([rpy, max_flow_pt.reshape(3, 1)]), np.array([[0, 0, 0, 1]])]
        )
        T_robot_goal = T_robot_world @ T_world_goal

        twist_mat = transformations.transform_log_from_transform(T_robot_goal)
        deltaV = twist_mat[:3, -1]
        omg_wedge = twist_mat[:3, :3]
        deltaOmg = np.array([omg_wedge[-1, 1], -omg_wedge[-1, 0], omg_wedge[1, 0]])
        deltaOmg, deltaV, _ = pose2exp_coordinate(T_robot_goal)

        hand_forward = np.zeros(osc_interface.osc_dim)

        if not np.isclose(max_flow_pt, hand_xyz).all():
            print("MOVING EE")
            # planner.ik_solve(np.hstack([max_flow_pt, R.from_matrix(rpy).as_quat()]))
            breakpoint()
            hand_forward[4] = 0.5
            hand_forward[5] = 0.5
            hand_forward[6:9] = deltaV * 0.3
            hand_forward[9:12] = deltaOmg * 0.3
            print(hand_forward)
        else:
            print("EE HOLD")
            hand_forward[4] = 0.5
            hand_forward[5] = 0.5
            hand_forward = np.zeros_like(hand_forward)
            if phase_counter >= 50:
                phase = 2
                phase_counter = 0
            else:
                phase_counter += 1
        keep_open = osc_interface.operational_space_and_null_space_to_joint_space(
            qpos, hand_forward, action[: osc_interface.null_space_dim]
        )
        action = keep_open
        return action, phase, phase_counter, rpy
    # Phase 3: Back Up
    else:
        hand_forward = np.zeros(osc_interface.osc_dim)
        hand_forward[4] = -0.5
        hand_forward[5] = -0.5
        if phase_counter >= 50:
            hand_forward[0] = -0.2
        else:
            phase_counter += 1
            hand_forward[0] = 0
        print("PHASE 3 GRASPING AND BACKING UP")
        keep_close = osc_interface.operational_space_and_null_space_to_joint_space(
            qpos, hand_forward, action[: osc_interface.null_space_dim]
        )
        action = keep_close
        return action, phase, phase_counter, hand_rpy


def pose2exp_coordinate(pose):
    """
    Compute the exponential coordinate corresponding to the given SE(3) matrix
    Note: unit twist is not a unit vector
    Args:
        pose: (4, 4) transformation matrix
    Returns:
        Unit twist: (6, ) vector represent the unit twist
        Theta: scalar represent the quantity of exponential coordinate
    """

    omega, theta = rot2so3(pose[:3, :3])
    ss = skew(omega)
    inv_left_jacobian = (
        np.eye(3, dtype=np.float) / theta
        - 0.5 * ss
        + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
    )
    v = inv_left_jacobian @ pose[:3, 3]
    return omega, v, theta


def skew(vec):
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


def rot2so3(rotation):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    # if np.isclose(rotation.trace(), -1):
    #     raise RuntimeError
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = (
        1
        / 2
        / np.sin(theta)
        * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ).T
    )
    return omega, theta


def max_flow_pt_calc(env, env_name, top_k=1):

    obs = env.get_obs()
    with open("gen.urdf", "w") as f:
        f.write(env.cabinet.export_urdf())
        f.close()

    urdf = PMTree.parse_urdf("gen.urdf")
    chain = urdf.get_chain(env_name[-9:-3])
    _, flow_test, seg_pcd, seg_clr = transform_pcd(obs, chain, 0.3)
    flow_norm_allpts = np.linalg.norm(flow_test, axis=1)
    max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]
    anchor_pt = np.mean(seg_pcd[max_flow_idx, :], axis=0)
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
    # np.save('knob.npy', {'xyz': seg_pcd[seg_knob_idx], 'xyz_color': seg_clr[seg_knob_idx]})
    # np.save("handle_mani_1000.npy", seg_pcd[seg_knob_idx])
    max_flow_knob_idx_temp = np.argpartition(flow_norm_allpts[seg_knob_idx], -top_k)[
        -top_k:
    ]

    # Map the argmax back to the original seg pcd
    max_flow_knob_pt = np.mean(seg_pcd[seg_knob_idx[max_flow_knob_idx_temp], :], axis=0)
    knob_pts = seg_pcd[seg_knob_idx]
    np.save("mani_1000_world", seg_pcd)

    # Visualize the max flow point on knob
    # Also Visualizing where2act space PCD since they are in different spaces.
    # Hard-coding a transformation for now.

    # w2a = np.load("/home/knox/ManiSkill/w2a_handle.npy")
    w2a = np.load("/home/knox/discriminative_embeddings/test_w2a.npy")
    # t_mani_w2a(seg_pcd, w2a)
    T_m_w = np.load(os.path.join(os.getcwd(), "T_mani_w2a.npy"), allow_pickle=True)
    R_trans = np.array(dict(enumerate(T_m_w.flatten(), 1))[1]["R"])
    # R_trans = T_m_w.item().get("R")
    ratio = T_m_w.item().get("ratio")
    shift_1 = T_m_w.item().get("shift_1")
    shift_2 = T_m_w.item().get("shift_2")
    trans_m_w_matrix = np.array(
        dict(enumerate(T_m_w.flatten(), 1))[1]["transformation"]
    )
    trans_w_m_matrix = np.linalg.inv(trans_m_w_matrix)
    # tra_seg_pcd = (seg_pcd * ratio + shift_1) @ R_trans + shift_2
    tra_seg_pcd = seg_pcd @ trans_m_w_matrix[:3, :3] + trans_m_w_matrix[3, :3]
    tra_w2a = w2a @ trans_w_m_matrix[:3, :3] + trans_w_m_matrix[3, :3]
    print(tra_seg_pcd.shape)
    scene = trimesh.Scene(
        [
            trimesh.points.PointCloud(seg_pcd[seg_knob_idx]),
            # Max flow point of 1001 from WHERE2ACT
            trimesh.points.PointCloud(
                w2a,
                colors=(0, 255, 0),
            ),
            trimesh.points.PointCloud(
                max_flow_knob_pt.reshape(1, 3), colors=(255, 0, 0)
            ),
            trimesh.points.PointCloud(tra_w2a, colors=(0, 0, 255)),
        ]
    )
    scene.show(viewer="gl")

    return knob_pts, max_flow_knob_pt


def R_arb(theta, u):
    from math import cos, sin

    return [
        [
            cos(theta) + u[0] ** 2 * (1 - cos(theta)),
            u[0] * u[1] * (1 - cos(theta)) - u[2] * sin(theta),
            u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta),
        ],
        [
            u[0] * u[1] * (1 - cos(theta)) + u[2] * sin(theta),
            cos(theta) + u[1] ** 2 * (1 - cos(theta)),
            u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta),
        ],
        [
            u[0] * u[2] * (1 - cos(theta)) - u[1] * sin(theta),
            u[1] * u[2] * (1 - cos(theta)) + u[0] * sin(theta),
            cos(theta) + u[2] ** 2 * (1 - cos(theta)),
        ],
    ]


def t_mani_w2a(mani, w2a):
    mani_bottom_idx = np.where(mani[:, 2] < 0.12)[0]
    mani_right_idx = np.argmin(mani[mani_bottom_idx, 1])
    mani_left_idx = np.argmax(mani[mani_bottom_idx, 1])
    mani_bottom_right_idx = mani_bottom_idx[mani_right_idx]
    mani_bottom_left_idx = mani_bottom_idx[mani_left_idx]
    mani_bottom_right = mani[mani_bottom_right_idx]
    mani_bottom_left = mani[mani_bottom_left_idx]

    w2a = w2a[np.where(w2a[:, 2] > -0.68)]
    w2a_bottom_idx = np.where(w2a[:, 2] < -0.6)[0]
    w2a_right_idx = np.argmin(w2a[w2a_bottom_idx, 1])
    w2a_bottom_right_idx = w2a_bottom_idx[w2a_right_idx]
    w2a_bottom_right = w2a[w2a_bottom_right_idx]
    w2a_left_idx = np.argmax(w2a[w2a_bottom_idx, 1])
    w2a_bottom_left_idx = w2a_bottom_idx[w2a_left_idx]
    w2a_bottom_left = w2a[w2a_bottom_left_idx]

    mani_length = mani_bottom_right[1] - mani_bottom_left[1]
    w2a_length = w2a_bottom_right[1] - w2a_bottom_left[1]
    ratio = 0.5 * w2a_length / mani_length
    mani_scaled = mani * ratio
    shift_1 = w2a_bottom_right - mani_scaled[mani_bottom_right_idx]
    mani_scaled = mani_scaled + shift_1
    mani_axis = w2a[w2a_bottom_right_idx] - w2a[w2a_bottom_left_idx]
    mani_axis[2] = 0
    mani_axis /= np.linalg.norm(mani_axis)
    R_transform = np.array(R_arb(-36 / 180 * np.pi, mani_axis))
    mani_scaled = mani_scaled @ R_transform
    shift_2 = w2a_bottom_right - mani_scaled[mani_bottom_right_idx]
    mani_scaled = mani_scaled + shift_2
    meta_transformation = {
        "ratio": ratio,
        "R": R_transform,
        "shift_1": shift_1,
        "shift_2": shift_2,
        "transformation": np.hstack(
            [
                np.vstack(
                    [
                        ratio * R_transform,
                        (shift_1 @ R_transform + shift_2),
                    ]
                ),
                np.array([0, 0, 0, 1]).reshape(4, 1),
            ],
        ),
    }
    print(meta_transformation["transformation"])
    np.save(os.path.join(os.getcwd(), "T_mani_w2a"), meta_transformation)
    return mani_scaled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, default="OpenCabinetDoor_1000_link_0-v0"
    )
    args = parser.parse_args()
    env_name = args.env_name

    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="dense")
    env.reset(level=1)
    osc_interface = OperationalSpaceControlInterface(env_name)
    # planner = PlanningModule(env.agent.robot)

    knob_pts, max_flow_knob_pt = max_flow_pt_calc(env, env_name, 1)

    # Determine the alignment of the gripper (horizontal or vertical)
    knob_w = knob_pts[:, 1].max() - knob_pts[:, 1].min()
    knob_h = knob_pts[:, 2].max() - knob_pts[:, 2].min()
    gripper_horizontal = knob_w < knob_h
    phase = 0

    # Stepping in gym
    phase_counter = 0
    rpy = None
    for i in range(10_000):
        # env.render("human")
        action, phase, phase_counter, rpy = vanilla_grasping_policy(
            env, max_flow_knob_pt, gripper_horizontal, phase, phase_counter, rpy
        )
        obs, reward, done, info = env.step(action)
    env.close()
