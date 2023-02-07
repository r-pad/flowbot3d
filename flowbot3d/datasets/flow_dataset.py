from typing import Dict, List, Literal, Optional, Sequence, TypedDict, Union

import numpy as np
import numpy.typing as npt
import rpad.partnet_mobility_utils.articulate as pma
import rpad.partnet_mobility_utils.dataset as pmd
from rpad.partnet_mobility_utils.data import PMObject


class Flowbot3DData(TypedDict):
    id: str
    pos: npt.NDArray[np.float32]  # (N, 3): Point cloud observation.
    flow: npt.NDArray[np.float32]  # (N, 3): Ground-truth flow.
    mask: npt.NDArray[np.bool_]  #  (N,): Mask the point of interest.


def compute_normalized_flow(
    P_world: npt.NDArray[np.float32],
    T_world_base: npt.NDArray[np.float32],
    current_jas: Dict[str, float],
    pc_seg: npt.NDArray[np.uint8],
    labelmap: Dict[str, int],
    pm_raw_data: PMObject,
    linknames: Union[Literal["all"], Sequence[str]] = "all",
) -> npt.NDArray[np.float32]:
    """Compute normalized flow for an object, based on its kinematics.

    Args:
        P_world (npt.NDArray[np.float32]): Point cloud render of the object in the world frame.
        T_world_base (npt.NDArray[np.float32]): The pose of the base link in the world frame.
        current_jas (Dict[str, float]): The current joint angles (easy to acquire from the render that created the points.)
        pc_seg (npt.NDArray[np.uint8]): The segmentation labels of each point.
        labelmap (Dict[str, int]): Map from the link name to segmentation name.
        pm_raw_data (PMObject): The object description, essentially providing the kinematic structure of the object.
        linknames (Union[Literal['all'], Sequence[str]], optional): The names of the links for which to
            compute flow. Defaults to "all", which will articulate all of them.

    Returns:
        npt.NDArray[np.float32]: _description_
    """

    # We actuate all links.
    if linknames == "all":
        joints = pm_raw_data.semantics.by_type("slider")
        joints += pm_raw_data.semantics.by_type("hinge")
        linknames = [joint.name for joint in joints]

    flow = np.zeros_like(P_world)

    for linkname in linknames:
        P_world_new = pma.articulate_joint(
            pm_raw_data,
            current_jas,
            linkname,
            0.01,  # Articulate by only a little bit.
            P_world,
            pc_seg,
            labelmap,
            T_world_base,
        )
        link_flow = P_world_new - P_world
        flow += link_flow

    largest_mag: float = np.linalg.norm(flow, axis=-1).max()

    normalized_flow = flow / (largest_mag + 1e-6)

    return normalized_flow


class Flowbot3DDataset:
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        n_points: Optional[int] = None,
    ) -> None:
        """The FlowBot3D dataset. Set n_points depending if you can handle ragged batches or not.

        Args:
            root (str): The root directory of the downloaded partnet-mobility dataset.
            split (Union[pmd.AVAILABLE_DATASET, List[str]]): Either an available split like "umpnet-train-train" or a list of object IDs from the PM dataset.
            randomize_joints (bool): Whether or not to randomize the joints.
            randomize_camera (bool): Whether or not to randomize the camera location (in a fixed range, see the underlying renderer...)
            n_points (Optional[int], optional): Whether or not to downsample the number of points returned for each example. If
                you want to use this datasets as a standard PyTorch dataset, you should set this to a non-None value (otherwise passing it into
                a dataloader won't really work, since you'll have ragged batches. If you're using PyTorch-Geometric to handle batches, do whatever you want.
                Defaults to None.
        """
        self._dataset = pmd.PCDataset(root=root, split=split, renderer="pybullet")
        self.randomize_joints = randomize_joints
        self.randomize_camera = randomize_camera
        self.n_points = n_points

    def get_data(self, obj_id: str, seed=None) -> Flowbot3DData:
        # Select the camera.
        joints = "random" if self.randomize_joints else None
        camera_xyz = "random" if self.randomize_camera else None

        rng = np.random.default_rng(seed)
        seed1, seed2 = rng.bit_generator._seed_seq.spawn(2)  # type: ignore

        data = self._dataset.get(
            obj_id=obj_id, joints=joints, camera_xyz=camera_xyz, seed=seed1  # type: ignore
        )
        pos = data["pos"]

        # Compute the flow.
        flow = compute_normalized_flow(
            P_world=pos,
            T_world_base=data["T_world_base"],
            current_jas=data["angles"],
            pc_seg=data["seg"],
            labelmap=data["labelmap"],
            pm_raw_data=self._dataset.pm_objs[obj_id],
            linknames="all",
        )

        # Compute the mask of any part which has flow.
        mask = (~(np.isclose(flow, 0.0)).all(axis=-1)).astype(np.bool_)

        if self.n_points:
            rng = np.random.default_rng(seed2)
            ixs = rng.permutation(range(len(pos)))[: self.n_points]
            pos = pos[ixs]
            flow = flow[ixs]
            mask = mask[ixs]

        return {
            "id": data["id"],
            "pos": pos,
            "flow": flow,
            "mask": mask,
        }

    def __getitem__(self, item: int) -> Flowbot3DData:
        obj_id = self._dataset._ids[item]
        return self.get_data(obj_id)

    def __len__(self):
        return len(self._dataset)
