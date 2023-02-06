# MODIFIED FROM:https://github.com/haosulab/ManiSkill-Learn/blob/main/mani_skill_learn/env/observation_process.py

import numpy as np


def process_mani_skill_base(obs, env=None, even_sampling: bool = False):
    # obs: iterated dict,
    # e.g. {'agent': [], 'additional_task_info': [], 'rgbd': {'rgb': [], 'depth': [], 'seg': []}}
    # e.g. {'agent': [], 'additional_task_info': [], 'pointcloud': {'rgb': [], 'xyz': [], 'seg': []}}

    if not isinstance(obs, dict):
        # e.g. [N,d] numpy array, for state observation
        return obs
    obs_mode = env.obs_mode
    if obs_mode in ["state", "rgbd"]:
        return obs
    elif obs_mode == "pointcloud":
        rgb = obs[obs_mode]["rgb"]
        xyz = obs[obs_mode]["xyz"]
        seg = obs[obs_mode]["seg"]
        flow = obs[obs_mode]["flow"]
        flow_all = obs[obs_mode]["flow_all"]

        # import pdb
        #
        # pdb.set_trace()
        # Given that xyz are already in world-frame, then filter the point clouds that belong to ground.
        mask = xyz[:, 2] > 1e-3
        rgb = rgb[mask]
        xyz = xyz[mask]
        seg = seg[mask]
        flow = flow[mask]
        flow_all = {k: v[mask] for k, v in flow_all.items()}

        tot_pts = 1200

        if even_sampling:
            if len(rgb) <= tot_pts:
                chosen_seg = seg
                chosen_xyz = xyz
                chosen_rgb = rgb
                chosen_flow = flow
                chosen_flow_all = flow_all
            else:
                chosen_ixs = np.random.permutation(len(rgb))[:tot_pts]
                chosen_seg = seg[chosen_ixs]
                chosen_xyz = xyz[chosen_ixs]
                chosen_rgb = rgb[chosen_ixs]
                chosen_flow = flow[chosen_ixs]
                chosen_flow_all = {k: f[chosen_ixs] for k, f in flow_all.items()}
        else:
            target_mask_pts = 800
            min_pts = 50
            num_pts = np.sum(seg, axis=0)
            tgt_pts = np.array(num_pts)
            # if there are fewer than min_pts points, keep all points
            surplus = np.sum(np.maximum(num_pts - min_pts, 0)) + 1e-6

            # randomly sample from the rest
            sample_pts = target_mask_pts - np.sum(np.minimum(num_pts, min_pts))
            for i in range(seg.shape[1]):
                if num_pts[i] <= min_pts:
                    tgt_pts[i] = num_pts[i]
                else:
                    tgt_pts[i] = min_pts + int(
                        (num_pts[i] - min_pts) / surplus * sample_pts
                    )

            chosen_seg = []
            chosen_rgb = []
            chosen_xyz = []
            chosen_flow = []
            chosen_flow_all = {k: [] for k in flow_all.keys()}
            chosen_mask_pts = 0
            for i in range(seg.shape[1]):
                if num_pts[i] == 0:
                    continue
                cur_seg = np.where(seg[:, i])[0]
                shuffle_indices = np.random.permutation(cur_seg)[: tgt_pts[i]]
                chosen_mask_pts += shuffle_indices.shape[0]
                chosen_seg.append(seg[shuffle_indices])
                chosen_rgb.append(rgb[shuffle_indices])
                chosen_xyz.append(xyz[shuffle_indices])
                chosen_flow.append(flow[shuffle_indices])
                for k in chosen_flow_all.keys():
                    chosen_flow_all[k].append(flow_all[k][shuffle_indices])
            sample_background_pts = tot_pts - chosen_mask_pts

            if seg.shape[1] == 1:
                bk_seg = np.logical_not(seg[:, 0])
            else:
                bk_seg = np.logical_not(
                    np.logical_or(*([seg[:, i] for i in range(seg.shape[1])]))
                )
            bk_seg = np.where(bk_seg)[0]
            shuffle_indices = np.random.permutation(bk_seg)[:sample_background_pts]

            chosen_seg.append(seg[shuffle_indices])
            chosen_rgb.append(rgb[shuffle_indices])
            chosen_xyz.append(xyz[shuffle_indices])
            chosen_flow.append(flow[shuffle_indices])
            for k in chosen_flow_all.keys():
                chosen_flow_all[k].append(flow_all[k][shuffle_indices])

            chosen_seg = np.concatenate(chosen_seg, axis=0)
            chosen_rgb = np.concatenate(chosen_rgb, axis=0)
            chosen_xyz = np.concatenate(chosen_xyz, axis=0)
            chosen_flow = np.concatenate(chosen_flow, axis=0)
            for k in chosen_flow_all.keys():
                chosen_flow_all[k] = np.concatenate(chosen_flow_all[k], axis=0)

        if chosen_seg.shape[0] < tot_pts:
            pad_pts = tot_pts - chosen_seg.shape[0]
            chosen_seg = np.concatenate(
                [
                    chosen_seg,
                    np.zeros([pad_pts, chosen_seg.shape[1]]).astype(chosen_seg.dtype),
                ],
                axis=0,
            )
            chosen_rgb = np.concatenate(
                [
                    chosen_rgb,
                    np.zeros([pad_pts, chosen_rgb.shape[1]]).astype(chosen_rgb.dtype),
                ],
                axis=0,
            )
            chosen_xyz = np.concatenate(
                [
                    chosen_xyz,
                    np.zeros([pad_pts, chosen_xyz.shape[1]]).astype(chosen_xyz.dtype),
                ],
                axis=0,
            )
            chosen_flow = np.concatenate(
                [
                    chosen_flow,
                    np.zeros([pad_pts, chosen_flow.shape[1]]).astype(chosen_flow.dtype),
                ],
                axis=0,
            )
            chosen_flow_all = {
                k: np.concatenate(
                    [
                        chosen_flow_all[k],
                        np.zeros([pad_pts, chosen_flow_all[k].shape[1]]).astype(
                            chosen_flow_all[k].dtype
                        ),
                    ],
                    axis=0,
                )
                for k in chosen_flow_all.keys()
            }
        obs[obs_mode]["seg"] = chosen_seg
        obs[obs_mode]["xyz"] = chosen_xyz
        obs[obs_mode]["rgb"] = chosen_rgb
        obs[obs_mode]["flow"] = chosen_flow
        obs[obs_mode]["flow_all"] = chosen_flow_all
        return obs
    else:
        print(f"Unknown observation mode {obs_mode}")
        exit(0)
