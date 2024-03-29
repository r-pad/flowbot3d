import abc
import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, Optional, Protocol, Tuple

import cv2
import gym
import numpy as np
import plotly.graph_objects as go
import torch

# from mani_skill_learn.env.observation_process import process_mani_skill_base
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

import flowbot3d.grasping.env  # noqa
from flowbot3d.distributed_eval import distributed_eval
from flowbot3d.grasping.agents.flowbot3d import FlowBot3DDetector, FlowNetAnimation
from flowbot3d.grasping.agents.grasp_pull import GraspPullAgent
from flowbot3d.grasping.agents.gt_flow import GTFlowDetector
from flowbot3d.grasping.agents.normal import NormalPullDirectionDetector
from flowbot3d.grasping.agents.screwnet import ScrewNetDetector
from flowbot3d.grasping.agents.umpnet_di import (
    UMPAnimation,
    UMPNetPullDirectionDetector,
)
from flowbot3d.grasping.env.wrappers import FlowBot3DWrapper

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


class PCAgent(Protocol):
    @abc.abstractmethod
    def get_action(self, obs) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def reset(self, obs):
        pass


class EpisodeAnimator(Protocol):
    @abc.abstractmethod
    def animate(self) -> go.Figure:
        pass


@torch.no_grad()
def run_trial(
    env_name,
    level,
    agent: PCAgent,
    render_rgb_video,
    bad_convex,
    ajar,
    success_threshold=0.1,
    max_episode_length=150,
) -> Dict[str, Any]:
    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="sparse")
    env = FlowBot3DWrapper(env)
    _ = env.reset(level=level)

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

    # Get the initial observation, and reset the agent.
    obs = env.observation(env.get_obs())
    agent.reset(obs)

    dr = None

    if render_rgb_video:
        rgb_imgs = []

    for i in range(max_episode_length):
        if render_rgb_video:
            image = env.render(mode="color_image")["world"]["rgb"]
            image = image[..., ::-1]
            rgb_imgs.append(image)

        # Get an action.
        action, extras = agent.get_action(obs)

        # If the policy asks, we can add a drive to the scene.
        if "drive" in extras:
            dr = env.env._scene.create_drive(
                env.agent.robot.get_links()[-1],
                env.agent.robot.get_links()[-1].get_cmass_local_pose(),
                env.target_link,
                env.target_link.get_cmass_local_pose(),
            )
            dr.set_x_properties(stiffness=45000, damping=0)
            dr.set_y_properties(stiffness=45000, damping=0)
            dr.set_z_properties(stiffness=45000, damping=0)

        # Step the action.
        obs, _, _, info = env.step(action)

        # Remove when we're done, at every time step.
        if "drive" in extras:
            env.env._scene.remove_drive(dr)
            dr = None

        # UMPNet Metrics
        if not math.isnan(env.cabinet.get_qpos()[0]):
            norm_dist = abs(q_limit - env.cabinet.get_qpos()[q_idx]) / (
                1e-6 + total_qdist
            )
            if np.isclose(norm_dist, 0.0, atol=1e-5):
                break
    env.close()

    # Extract distance metric.
    if not math.isnan(env.cabinet.get_qpos()[0]):
        dist = norm_dist
    else:
        dist = 1 - int(info["eval_info"]["success"])

    # Put the results together.
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

    # Create an agent.
    agent, animation_module = create_agent(
        model_name, ckpt_path, device, save_animation, cam_frame, bad_door=bad_door
    )

    results = run_trial(
        env_name=env_name,
        level=0,
        agent=agent,
        render_rgb_video=save_video,
        bad_convex=bad_door,
        ajar=ajar,
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
    if animation_module:
        animated_fig = animation_module.animate()
        animation_fn = os.path.join(outcome_dir, f"{env_name}.html")
        animated_fig.write_html(animation_fn)

    # Append results.
    results_fn = os.path.join(result_dir, "results.txt")
    with open(results_fn, "a") as f:
        print("{}: {}".format(obj_cat, n_dist), file=f)


def create_agent(
    model_name, ckpt_path, device, animate, cam_frame, bad_door
) -> Tuple[PCAgent, Optional[EpisodeAnimator]]:
    animation: Optional[EpisodeAnimator]
    if "flowbot" in model_name:
        if animate:
            animation = FlowNetAnimation()
        else:
            animation = None

        # We use ArtFlowNet predictions to detect both contacts and pull direction.
        flowbot3d_detector = FlowBot3DDetector(ckpt_path, device, cam_frame, animation)
        agent = GraspPullAgent(
            contact_detector=flowbot3d_detector,
            pull_dir_detector=flowbot3d_detector,
            device=device,
        )
        return agent, animation
    elif "normal" in model_name:
        if animate:
            animation = UMPAnimation()
        else:
            animation = None
        contact_detector = GTFlowDetector(bad_door)
        pull_dir_detector = NormalPullDirectionDetector(animation, cam_frame)
        agent = GraspPullAgent(
            contact_detector=contact_detector,
            pull_dir_detector=pull_dir_detector,
            device=device,
        )
        return agent, animation
    elif "umpnet" in model_name:
        if animate:
            animation = UMPAnimation()
        else:
            animation = None

        agent = GraspPullAgent(
            contact_detector=GTFlowDetector(bad_door),
            pull_dir_detector=UMPNetPullDirectionDetector(ckpt_path, device, animation),
            device=device,
        )
        return agent, animation
    elif "screw" in model_name:
        if animate:
            animation = FlowNetAnimation()
        else:
            animation = None

        # We use ArtFlowNet predictions to detect both contacts and pull direction.
        screwnet_detector = ScrewNetDetector(ckpt_path, device, animation, cam_frame)
        agent = GraspPullAgent(
            contact_detector=screwnet_detector,
            pull_dir_detector=screwnet_detector,
            device=device,
        )
        return agent, animation
    elif "bc" in model_name or "dagger_e2e" in model_name:
        raise NotImplementedError("bc not implemented")
    elif "dagger_oracle" in model_name:
        raise NotImplementedError("dagger not implemented")
    elif "gt_flow" in model_name:
        gt_flow_detector = GTFlowDetector(bad_door)
        agent = GraspPullAgent(
            contact_detector=gt_flow_detector,
            pull_dir_detector=gt_flow_detector,
            device=device,
        )
        return agent, None
    else:
        raise ValueError("no agent with that name")


def debug_single(model_name, ckpt_path, device, result_dir, ajar, cam_frame):
    # Load a model.
    agent, animation = create_agent(
        model_name, ckpt_path, device, animate=True, cam_frame=cam_frame, bad_door=False
    )

    env_name = "OpenCabinetDoorGripper_{}_link_0-v0".format("101377")

    run_trial(
        env_name=env_name,
        level=0,
        agent=agent,
        render_rgb_video=True,
        bad_convex=False,
        ajar=ajar,
    )

    if animation:
        save_html = animation.animate()
        if save_html:
            save_html.write_html(os.path.join(result_dir, "debug.html"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
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
    split = args.split
    ajar = args.ajar
    cam_frame = args.cf
    ckpt_path = args.ckpt_path
    proc_start = args.proc_start
    n_workers = args.n_workers
    n_proc_per_worker = args.n_proc_per_worker
    result_dir = os.path.join(os.getcwd(), "results", "sapien_evals", args.model_name)
    start_ind = args.ind_start
    end_ind = args.ind_end
    model_name = args.model_name
    debug = args.debug

    seed = 12345
    save_video = args.no_video
    save_animation = args.no_animation

    # For this version of SAPIEN, GPU is not respected. So put all the torch models on a different device.
    device = "cuda:1"

    # If we're just debugging a single example, short-circuit the rest..
    if debug:
        os.makedirs(result_dir, exist_ok=True)
        debug_single(model_name, ckpt_path, device, result_dir, ajar, cam_frame)
        sys.exit(0)

    if split not in {"train_train", "train_test", "unseen"}:
        raise ValueError("invalid split")

    # Create results directories.
    if split == "train_train" or split == "train_test":
        # Get the envs.
        umpnet_split_fn = os.path.join(
            os.getcwd(), "umpnet_obj_splits", f"{split}_split.txt"
        )
    else:
        umpnet_split_fn = os.path.join(
            os.getcwd(), "umpnet_obj_splits", "test_test_split.txt"
        )

    result_dir = os.path.join(result_dir, split)
    succ_res_dir = os.path.join(result_dir, "succ")
    fail_res_dir = os.path.join(result_dir, "fail")

    if not os.path.exists(result_dir):
        print("Creating result directory")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(succ_res_dir, exist_ok=True)
        os.makedirs(fail_res_dir, exist_ok=True)
        print("Directory created")
    else:
        raise ValueError(f"results already exist in this directory: {result_dir}")

    with open(umpnet_split_fn, "r") as file:
        # Get all the environments, and lop off the last newline.
        available_envs = file.read().split("\n")[:-1]

    start = time.perf_counter()

    # Construct inputs.
    kwargs_list = [
        dict(
            model_name=model_name,
            ckpt_path=ckpt_path,
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
