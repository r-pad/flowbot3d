from typing import List, Optional, Protocol, Union, cast

import numpy as np
import rpad.partnet_mobility_utils.articulate as rpa
import rpad.partnet_mobility_utils.dataset as pmd
import torch
import torch_geometric.data as tgd


class ScrewPCData(Protocol):
    id: str  # Object ID.
    category: str  # Object category.
    parent_link: str
    child_link: str
    joint_type: str
    joint_name: str
    joint_angle: float

    pos: torch.Tensor  # Points in the point cloud.
    mask: torch.Tensor  # Mask of the part of interest.
    origin: torch.Tensor  # Origin. If not revolute, is a 0.
    axis: torch.Tensor  # Axis.


class ScrewDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        n_points: Optional[int] = 1200,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.n_points = n_points
        self.normalize = normalize

        # Store a list of all joints.
        self.joint_list, self._dataset = ScrewDataset.get_joint_list(root, split)

        super().__init__(root)

    @staticmethod
    def get_joint_list(root, split):
        dset = pmd.PCDataset(root=root, split=split, renderer="pybullet")
        joint_list = []
        for id, data in dset.pm_objs.items():
            for joint_sem in data.semantics.by_type(["slider", "hinge"]):
                joint_name = data.obj.get_joint_by_child(joint_sem.name).name
                joint_list.append((id, joint_name))
        return joint_list, dset

    @staticmethod
    def get_processed_dir(normalize):
        suffix = "_norm" if normalize else ""
        return f"screw_processed{suffix}"

    def len(self) -> int:
        return len(self.joint_list)

    def get(self, index) -> tgd.Data:
        obj_id, joint_name = self.joint_list[index]
        return self.get_data(obj_id, joint_name, seed=None)

    def get_data(
        self,
        obj_id: str,
        joint_name: str,
        randomize_joints=True,
        randomize_camera=True,
        ja=None,
        seed=None,
    ) -> ScrewPCData:
        raw_data = self._dataset.pm_objs[obj_id]
        joint = raw_data.obj.get_joint(joint_name)
        assert joint.origin is not None and joint.axis is not None

        pc_dict = self._dataset.get(
            obj_id,
            "random" if randomize_joints else ja,
            "random" if randomize_camera else None,
            seed,
        )

        # The origin and axis are in the link frame, so we want to get the transform from
        # the base frame to the link frame. This step gets that transform.
        chain = raw_data.obj.get_chain(joint.child)
        angles = pc_dict["angles"]
        jas = [angles[joint.name] for joint in chain[:-1]]
        T_base_link = rpa.fk(chain[:-1], jas)
        T_world_base = pc_dict["T_world_base"]
        T_world_link = T_world_base @ T_base_link

        # Take the origin and axis in the link frame.
        o_link, _ = joint.origin
        a_link = joint.axis
        p = np.array([o_link[0], o_link[1], o_link[2], 1.0]).reshape((4, 1))
        v = np.array([a_link[0], a_link[1], a_link[2]]).reshape((3, 1))

        # Put them in the world frame. The axis is a unit vector, and should only be rotated.
        origin_base = (T_world_link @ p)[:3, 0]
        axis_base = (T_world_link[:3, :3] @ v)[:3, 0]

        # Grab the pointcloud.
        pos = pc_dict["pos"]

        # The mask is just the link in question.
        mask = np.zeros(len(pos), dtype=bool)
        mask[pc_dict["seg"] == pc_dict["labelmap"][joint.child]] = True

        # Downsample.
        if self.n_points:
            ixs = np.random.permutation(range(len(pos)))[: self.n_points]
            pos = pos[ixs]
            mask = mask[ixs]

        if self.normalize:
            # Normalize XYZ.
            centroid = pos.mean(axis=-2)
            pos = pos - centroid
            scale = (1 / np.abs(pos).max()) * 0.999999
            pos = pos * scale

            # Normalize origin by the same factors.
            origin_base = origin_base - centroid
            origin_base = origin_base * scale

        data = tgd.Data(
            id=raw_data.obj_id,
            category=raw_data.category,
            parent_link=joint.parent,
            child_link=joint.child,
            joint_name=joint.name,
            joint_type=joint.type,
            joint_angle=angles[joint.name],
            pos=torch.from_numpy(pos).float(),
            mask=torch.from_numpy(mask).reshape(-1, 1).float(),
            origin=torch.from_numpy(origin_base).reshape(-1, 3).float(),
            axis=torch.from_numpy(axis_base).reshape(-1, 3).float(),
        )
        return cast(ScrewPCData, data)


class ScrewUniformDataset(tgd.Dataset):
    def __init__(self, root, split, n_points, n_views):
        self.n_views = n_views
        self.dataset = ScrewDataset(root, split, n_points)

        new_joint_list = []
        for oid, jn in self.dataset.joint_list:
            new_joint_list.extend([(oid, jn, vn) for vn in range(n_views)])
        self.joint_views = new_joint_list

    def get(self, index) -> tgd.Data:
        obj_id, joint_name, view_num = self.joint_views[index]
        return self.get_data(obj_id, joint_name, view_num=view_num, seed=None)

    def get_data(
        self,
        obj_id: str,
        joint_name: str,
        view_num: int = 0,
        randomize_joints=True,
        randomize_camera=True,
        ja=None,
        seed=None,
    ) -> ScrewPCData:
        assert not randomize_joints and ja is None

        ja = view_num / self.n_views

        raise NotImplementedError
        return self.dataset.get_data(
            obj_id,
            joint_name,
            randomize_joints=False,
            randomize_camera=randomize_camera,
            ja=ja,
            seed=seed,
        )
