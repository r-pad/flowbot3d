import os
from typing import List, Optional, Protocol, Union, cast

import partnet_mobility_utils.dataset as pmd
import torch
import torch.utils.data as td
import torch_geometric.data as tgd

from flowbot3d.dataset import Flowbot3DDataset
from flowbot3d.tg_utils import SingleObjDataset, parallel_sample


class Flowbot3DTGData(Protocol):
    id: str  # Object ID.


#     pos: torch.Tensor  # Points in the point cloud.
#     color: torch.Tensor  # Color of the points.
#     flow: torch.Tensor  # instantaneous positive 3D flow.
#     mask: torch.Tensor  # Mask of the part of interest.

#     x: Optional[torch.Tensor] = None


class Flowbot3DTGDataset(tgd.Dataset):
    """This dataset exists to wrap the Flowbot3D dataset so that it can be:
    1. consumed by Torch Geometric, and
    2. processed in parallel to sample an offline dataset.
    """

    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        n_repeat: int = 100,
        n_points: Optional[int] = 1200,
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        use_processed: bool = True,
        n_proc: int = -1,
    ):
        self.use_processed = use_processed
        self.split = split
        self.n_repeat = n_repeat
        self.n_points = n_points
        self.randomize_joints = randomize_joints
        self.randomize_camera = randomize_camera
        self.n_proc = n_proc

        self._dataset = Flowbot3DDataset(
            os.path.join(root, "raw"),
            split,
            randomize_joints,
            randomize_camera,
            n_points,
        )

        super().__init__(root)

        if self.use_processed:
            self.inmem: td.ConcatDataset = td.ConcatDataset(
                [SingleObjDataset(data_path) for data_path in self.processed_paths]
            )

    @property
    def processed_dir(self) -> str:
        joint_chunk = "rj" if self.randomize_joints else "sj"
        camera_chunk = "rc" if self.randomize_camera else "sc"
        return os.path.join(self.root, f"processed_{joint_chunk}_{camera_chunk}")

    @property
    def processed_file_names(self) -> List[str]:
        obj_ids = self._dataset._dataset._ids
        return [f"{obj_id}_{self.n_repeat}.pt" for obj_id in obj_ids]

    def process(self):
        if not self.use_processed:
            return
        else:
            obj_ids = self._dataset._dataset._ids

            n_proc = os.cpu_count() if self.n_proc == -1 else self.n_proc

            parallel_sample(
                dset_cls=Flowbot3DTGDataset,
                dset_kwargs=dict(
                    root=self.root,
                    split=self.split,
                    n_repeat=self.n_repeat,
                    n_points=self.n_points,
                    randomize_joints=self.randomize_joints,
                    randomize_camera=self.randomize_camera,
                    use_processed=False,
                ),
                get_data_args=[(obj_id,) for obj_id in obj_ids],
                n_repeat=self.n_repeat,
                n_proc=n_proc,
            )

    def len(self) -> int:
        return len(self._dataset) * self.n_repeat

    def get(self, idx: int) -> Flowbot3DTGData:
        if self.use_processed:
            data = self.inmem[idx]
            return cast(Flowbot3DTGData, data)
        else:
            idx = idx // self.n_repeat
            obj_ids = self._dataset._dataset._ids
            obj_id = obj_ids[idx]
            return self.get_data(obj_id)

    def get_data(self, obj_id: str) -> Flowbot3DTGData:

        data_dict = self._dataset[obj_id]

        return tgd.Data(
            id=data_dict["id"],
            pos=torch.from_numpy(data_dict["pos"]).float(),
            flow=torch.from_numpy(data_dict["flow"]).float(),
            mask=torch.from_numpy(data_dict["mask"]).float(),
        )
