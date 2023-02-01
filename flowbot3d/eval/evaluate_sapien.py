import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from typing import Any, Dict

import cv2
import gym
import numpy as np
import torch

# from mani_skill_learn.env.observation_process import process_mani_skill_base
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

import flowbot3d.grasping.env  # noqa
from flowbot3d.eval.utils import distributed_eval
from flowbot3d.models.flowbot3d import ArtFlowNet
from flowbot3d.visualizations import FlowNetAnimation

GLOBAL_PULL_VECTOR = np.array([[-1, 0, 0]])

# These are doors which have weird convex hull issues, so grasping doesn't work.
BAD_DOORS = {
    "8930",
    "9003",
    "9016",
    "9107",
    "9164",
    "9168",
    "9386",
    "9388",
    "9410",
}


def __load_obj_id_to_category():
    # Extract existing classes.
    with open(os.path.join(os.getcwd(), "umpnet_data_split.json"), "r") as f:
        data = json.load(f)

    id_to_cat = {}
    for _, category_dict in data.items():
        for category, split_dict in category_dict.items():
            for _, id_list in split_dict.items():
                for id in id_list:
                    id_to_cat[id] = category
    return id_to_cat


OBJ_ID_TO_CATEGORY = __load_obj_id_to_category()


def vanilla_grasping_policy(
    env,
    obs,
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
    device,
    animate,
):

    # obs = env.get_obs()
    # Define primitives
    action = np.zeros(8)
    ee_center = 0.5 * (env.agent.get_ee_coords()[0] + env.agent.get_ee_coords()[1])
    global GLOBAL_PULL_VECTOR
    # Phase 1: Grasp
    if phase == 1:

        delta_T = (max_flow_pt - ee_center) / np.linalg.norm(max_flow_pt - ee_center)
        # delta_R = 0.2 * R.from_matrix(T_robot_goal[:3, :3]).as_euler("xyz")
        action = np.zeros(8)
        # print(ee_center)

        # if gripper_horizontal and env.agent.robot.get_qpos()[5] < np.pi / 2:
        if gripper_vertical and env.agent.robot.get_qpos()[3] > -np.pi / 2:
            action[3] = -1
            # print("Rotating Gripper")
        if env.agent.robot.get_qpos()[5] < np.pi / 2:
            action[5] = 1
            # print("Rotating Gripper")
        if not np.isclose(max_flow_pt, ee_center, atol=0.1).all():
            # print("MOVING EE")
            action[:3] = aux_T[:3, :3] @ delta_T
            # print(action)
            vel = np.linalg.norm(env.agent.get_ee_vels().mean(axis=0))
            if vel < 1e-2:
                # print(vel)
                # print(phase_counter)
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
            # print("EE HOLD")
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
                env,
                obs,
                flow_model,
                1,
                phase_counter,
                animation,
                cam_frame,
                device,
                animate,
            )
            pull_vector = pull_vector.reshape(1, 3)
            # print("pull vec: ", pull_vector)
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
        # print("PHASE 3 GRASPING AND BACKING UP")
        return action, phase, phase_counter, dr


def max_flow_pt_calc(
    env,
    obs,
    model,
    top_k,
    id,
    animation: FlowNetAnimation,
    cam_frame=False,
    device="cuda",
    animate=True,
):
    """For the initial grasping point selection"""
    # obs = env.get_obs()
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
            torch.from_numpy(xyz).to(device),
            torch.from_numpy(mask_1[mask_meta]).float(),
        )
        pred_flow = pred_flow.cpu().numpy()

    else:
        cam_mat = env.cameras[1].sub_cameras[0].get_pose().to_transformation_matrix()
        pred_flow = model.predict(
            torch.from_numpy(pcd @ cam_mat[:3, :3] + cam_mat[:3, -1]).to(device),
            torch.from_numpy(mask_1[mask_meta]).float(),
        )
        pred_flow = pred_flow.cpu().numpy()
        pred_flow = pred_flow @ np.linalg.inv(cam_mat)[:3, :3]

    flow_norm_allpts = np.linalg.norm(pred_flow, axis=1)
    flow_norm_allpts = np.divide(flow_norm_allpts, ee_to_pt_dist)
    max_flow_idx = np.argpartition(flow_norm_allpts, -top_k)[-top_k:]
    max_flow_pt = pcd[max_flow_idx]
    max_flow_vector = pred_flow[max_flow_idx]
    # print("max_flow: ", max_flow_vector)
    if animate:
        if animation:
            temp = animation.add_trace(
                torch.as_tensor(pcd),
                torch.as_tensor([pcd[mask_1[mask_meta]]]),
                torch.as_tensor(
                    [pred_flow[mask_1[mask_meta]] / np.linalg.norm(max_flow_vector)]
                ),
                "red",
            )
            animation.append_gif_frame(temp)

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


@torch.no_grad()
def run_trial(
    env_name,
    level,
    animate,
    render_rgb_video,
    bad_convex,
    flow_model,
    animation,
    ajar,
    cam_frame,
    device,
    success_threshold=0.1,
    max_episode_length=150,
) -> Dict[str, Any]:
    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="sparse")
    obs = env.reset(level=level)

    # Special handling of objects with convex hull issues.
    if bad_convex:
        r = R.from_euler("x", 360, degrees=True)
        rot_mat = r.as_matrix()
        r = r.as_quat()
        env.cabinet.set_root_pose(Pose(env.cabinet.get_root_pose().p, r))

    # If the joint should be ajar, open it a little bit.
    if ajar:
        qpos = []
        for joint in env.cabinet.get_active_joints():
            [[lmin, lmax]] = joint.get_limits()
            if joint.name == env.target_joint.name:
                qpos.append(lmin + 0.2 * (lmax - lmin))
            else:
                qpos.append(lmin)
        env.cabinet.set_qpos(np.array(qpos))

    # Get the range, for computing normalized distance.
    for i, j in enumerate(env.cabinet.get_active_joints()):
        if j.get_child_link().name == env.target_link.name:
            q_limit = env.cabinet.get_qlimits()[i][1]
            q_idx = i
            qpos_init = env.cabinet.get_qpos()[i]
            total_qdist = abs(q_limit - qpos_init)
            break

    obs = env.get_obs()

    # Detect whether the gripper is vertical?
    max_flow_knob_pt, flow_vec = max_flow_pt_calc(
        env, obs, flow_model, 1, 0, animation, cam_frame, device, animate
    )
    if flow_vec is None:
        flow_vec = np.array([[0, 0, 0]])
    if abs(flow_vec[0, 2]) > abs(flow_vec[0, 1]) and abs(flow_vec[0, 2]) > abs(
        flow_vec[0, 0]
    ):
        gripper_vertical = True
    else:
        gripper_vertical = False

    # pcd = env.get_obs()["pointcloud"]["xyz"]
    pcd = obs["pointcloud"]["xyz"]
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

    if render_rgb_video:
        rgb_imgs = []

    for i in range(max_episode_length):
        if render_rgb_video:
            image = env.render(mode="color_image")["world"]["rgb"]
            image = image[..., ::-1]
            rgb_imgs.append(image)

        action, phase, phase_counter, dr = vanilla_grasping_policy(
            env,
            obs,
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
            device,
            animate,
        )
        obs, reward, done, info = env.step(action)
        if dr:
            env._scene.remove_drive(dr)

        # UMPNet Metrics
        if not math.isnan(env.cabinet.get_qpos()[0]):
            # print("curr qpos:", env.cabinet.get_qpos()[q_idx])
            # print("qlimit: ", q_limit)
            norm_dist = abs(q_limit - env.cabinet.get_qpos()[q_idx]) / (
                1e-6 + total_qdist
            )
            # print("DIST: ", norm_dist)
            if np.isclose(norm_dist, 0.0, atol=1e-5):
                # print("SUCCESS")
                break
    env.close()
    if not math.isnan(env.cabinet.get_qpos()[0]):
        dist = norm_dist
    else:
        dist = 1 - int(info["eval_info"]["success"])

    result_dict = {
        "normalized_distance": dist,
        "success": dist < success_threshold,
    }

    if render_rgb_video:
        result_dict["rgb"] = rgb_imgs

    return result_dict


def set_up_and_run_trial(
    model_name,
    ckpt_path,
    mode,
    result_dir,
    cam_frame,
    succ_res_dir,
    fail_res_dir,
    env_name,
    ajar,
    device,
    save_video=True,
    save_animation=True,
):

    # Load the model.
    model = load_model(model_name, ckpt_path)
    model = model.to(device)  # type: ignore
    model.eval()

    # Get the object id and category.
    obj_id = env_name.split("_")[1]
    obj_cat = OBJ_ID_TO_CATEGORY[obj_id]

    # For certain objects, we want to force the object to be ajar - this is because
    # the collision geometries at 0-articulation are such that the simulator won't
    # allow the agent to open things even if it's pulling in the right direction.
    if obj_cat in {"Door", "Box", "Table", "Phone", "Bucket"}:
        ajar = True

    # Next, we need to detect if it's a "bad door", which has nasty collision geometries.
    bad_door = obj_id in BAD_DOORS

    animation_module = FlowNetAnimation()
    results = run_trial(
        env_name=env_name,
        level=0,
        animate=save_animation,
        render_rgb_video=save_video,
        bad_convex=bad_door,
        flow_model=model,
        animation=animation_module,
        ajar=ajar,
        cam_frame=cam_frame,
        device=device,
    )
    n_dist = results["normalized_distance"]
    success = results["success"]
    outcome_dir = succ_res_dir if success else fail_res_dir

    # Video.
    if save_video:
        rgb = results["rgb"]

        # Configure video writing.
        video_file = os.path.join(outcome_dir, f"{env_name}.mp4")
        video_writer = cv2.VideoWriter(
            video_file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (rgb[0].shape[1], rgb[0].shape[0]),
        )
        for img in rgb:
            video_writer.write(np.uint8(img * 255))

    # Animation.
    animated_fig = animation_module.animate()
    if animated_fig:
        animation_fn = os.path.join(outcome_dir, f"{env_name}.html")
        animated_fig.write_html(animation_fn)

    # Append results.
    m = "test" if mode == "test" else "train"
    results_fn = os.path.join(result_dir, f"{m}_test_res.txt")
    with open(results_fn, "a") as f:
        print("{}: {}".format(obj_cat, n_dist), file=f)


def load_model(model_name, ckpt_path):
    if "flowbot" in model_name:
        model = ArtFlowNet.load_from_checkpoint(ckpt_path)
    else:
        raise NotImplementedError

    return model


def debug_single(model_name, ckpt_path, device, result_dir, ajar, cam_frame):
    # Load a model.
    model = load_model(model_name, ckpt_path)
    model = model.to(device)  # type: ignore
    model.eval()

    animation_module = FlowNetAnimation()

    env_name = "OpenCabinetDoorGripper_{}_link_0-v0".format("101377")

    run_trial(
        env_name=env_name,
        level=0,
        animate=True,
        render_rgb_video=True,
        bad_convex=False,
        flow_model=model,
        animation=animation_module,
        ajar=ajar,
        cam_frame=cam_frame,
        device=device,
    )

    save_html = animation_module.animate()
    if save_html:
        save_html.write_html("debug.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="4_to_4_umpnet")
    parser.add_argument("--ind_start", type=int, default="0")
    parser.add_argument("--ind_end", type=int, default="0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ajar", action="store_true")
    parser.add_argument("--cf", action="store_true")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--proc_start", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=30)
    parser.add_argument("--n_proc_per_worker", type=int, default=1)
    parser.add_argument("--no_video", action="store_false")
    parser.add_argument("--no_animation", action="store_false")

    args = parser.parse_args()
    mode = args.mode
    ajar = args.ajar
    cam_frame = args.cf
    ckpt_path = args.ckpt_path
    proc_start = args.proc_start
    n_workers = args.n_workers
    n_proc_per_worker = args.n_proc_per_worker
    result_dir = os.path.join(os.getcwd(), "umpmetric_results_master", args.model_name)
    start_ind = args.ind_start
    end_ind = args.ind_end
    model_name = args.model_name
    debug = args.debug

    seed = 12345
    save_video = args.no_video
    save_animation = args.no_animation

    device = "cuda:1"

    # If we're just debugging a single example, short-circuit the rest..
    if debug:
        debug_single(model_name, ckpt_path, device, result_dir, ajar, cam_frame)
        sys.exit(0)

    # Create results directories.
    if mode == "train":
        succ_res_dir = os.path.join(result_dir, "succ")
        fail_res_dir = os.path.join(result_dir, "fail")
    else:
        succ_res_dir = os.path.join(result_dir, "succ_unseen")
        fail_res_dir = os.path.join(result_dir, "fail_unseen")
    if not os.path.exists(result_dir):
        print("Creating result directory")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(succ_res_dir, exist_ok=True)
        os.makedirs(fail_res_dir, exist_ok=True)
        print("Directory created")

    # Get the envs.
    umpnet_split_fn = os.path.join(
        os.getcwd(), "umpnet_obj_splits", f"{mode}_test_split.txt"
    )
    with open(umpnet_split_fn, "r") as file:
        # Get all the environments, and lop off the last newline.
        available_envs = file.read().split("\n")[:-1]

    start = time.perf_counter()

    # Construct inputs.
    kwargs_list = [
        dict(
            model_name=model_name,
            ckpt_path=ckpt_path,
            mode=mode,
            result_dir=result_dir,
            cam_frame=cam_frame,
            succ_res_dir=succ_res_dir,
            fail_res_dir=fail_res_dir,
            env_name=env_name,
            ajar=ajar,
            device=device,
            save_video=save_video,
            save_animation=save_animation,
        )
        for env_name in available_envs[start_ind:end_ind]
    ]

    # Will either distribute the evaluation across multiple processes,
    # or just run in a for-loop in-process (if n_workers == 0).
    results, completeds = distributed_eval(
        set_up_and_run_trial,
        kwargs_list,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        proc_start=proc_start,
        seed=seed,
    )

    print(f"number of non-errored trials: {sum(completeds)}")

    end = time.perf_counter()
    print(f"time to evaluate all episodes: {end - start:.2f}s")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
