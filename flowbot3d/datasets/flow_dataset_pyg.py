from typing import List, Optional, Protocol, Union, cast

import rpad.partnet_mobility_utils.dataset as pmd
import torch
import torch_geometric.data as tgd

from flowbot3d.datasets.flow_dataset import Flowbot3DDataset


class Flowbot3DTGData(Protocol):
    id: str  # Object ID.

    pos: torch.Tensor  # Points in the point cloud.
    flow: torch.Tensor  # instantaneous positive 3D flow.
    mask: torch.Tensor  # Mask of the part of interest.


class Flowbot3DPyGDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        n_points: Optional[int] = 1200,
        seed: int = 42
    ) -> None:
        super().__init__()

        self.dataset = Flowbot3DDataset(
            root,
            split,
            randomize_joints,
            randomize_camera,
            n_points,
        )
        self.seed = seed

    def len(self) -> int:
        return len(self.dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(self.dataset._dataset._ids[index])

    @staticmethod
    def get_processed_dir(randomize_joints, randomize_camera):
        joint_chunk = "rj" if randomize_joints else "sj"
        camera_chunk = "rc" if randomize_camera else "sc"
        return f"processed_{joint_chunk}_{camera_chunk}"

    def get_data(self, obj_id: str) -> Flowbot3DTGData:
        data_dict = self.dataset.get_data(obj_id, self.seed)

        data = tgd.Data(
            id=data_dict["id"],
            pos=torch.from_numpy(data_dict["pos"]).float(),
            flow=torch.from_numpy(data_dict["flow"]).float(),
            mask=torch.from_numpy(data_dict["mask"]).float(),
        )
        return cast(Flowbot3DTGData, data)
