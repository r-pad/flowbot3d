import pathlib

from gym import register
from mani_skill.utils.misc import get_raw_yaml

_this_file = pathlib.Path(__file__).resolve()

################################################################
# OpenCabinetDoor
################################################################
cabinet_door_model_file = _this_file.parent.joinpath(
    "../../../third_party/ManiSkill/mani_skill/assets/config_files/cabinet_models_door.yml"
)
# cabinet_door_ids = get_model_ids_from_yaml(cabinet_door_model_file)
cabinet_door_infos = get_raw_yaml(cabinet_door_model_file)

for cabinet_id in cabinet_door_infos:
    register(
        id="OpenCabinetDoorGripper_{:s}-v0".format(cabinet_id),
        entry_point="flowbot3d.grasping.env.open_cabinet_door_drawer_gripper:OpenCabinetDoorGripperEnv",
        kwargs={
            "variant_config": {"partnet_mobility_id": cabinet_id},
        },
    )

    for fixed_target_link_id in range(
        cabinet_door_infos[cabinet_id]["num_target_links"]
    ):
        register(
            id="OpenCabinetDoorGripper_{:s}_link_{:d}-v0".format(
                cabinet_id, fixed_target_link_id
            ),
            entry_point="flowbot3d.grasping.env.open_cabinet_door_drawer_gripper:OpenCabinetDoorGripperEnv",
            kwargs={
                "variant_config": {"partnet_mobility_id": cabinet_id},
                "fixed_target_link_id": fixed_target_link_id,
            },
        )

################################################################
# OpenCabinetDrawer
################################################################

cabinet_drawer_model_file = _this_file.parent.joinpath(
    "../../../third_party/ManiSkill/mani_skill/assets/config_files/cabinet_models_drawer.yml"
)
# cabinet_drawer_ids = get_model_ids_from_yaml(cabinet_drawer_model_file)
cabinet_drawer_infos = get_raw_yaml(cabinet_drawer_model_file)

for cabinet_id in cabinet_drawer_infos:
    register(
        id="OpenCabinetDrawerGripper_{:s}-v0".format(cabinet_id),
        entry_point="flowbot3d.grasping.env.open_cabinet_door_drawer_gripper:OpenCabinetDrawerGripperEnv",
        kwargs={
            "variant_config": {"partnet_mobility_id": cabinet_id},
        },
    )

    for fixed_target_link_id in range(
        cabinet_drawer_infos[cabinet_id]["num_target_links"]
    ):
        register(
            id="OpenCabinetDrawerGripper_{:s}_link_{:d}-v0".format(
                cabinet_id, fixed_target_link_id
            ),
            entry_point="flowbot3d.grasping.env.open_cabinet_door_drawer_gripper:OpenCabinetDrawerGripperEnv",
            kwargs={
                "variant_config": {"partnet_mobility_id": cabinet_id},
                "fixed_target_link_id": fixed_target_link_id,
            },
        )
