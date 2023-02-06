import os

import gym
import pytest

import flowbot3d.grasping.env  # noqa


@pytest.mark.skipif(
    not "PARTNET_MOBILITY_DATASET" in os.environ, reason="no dataset given"
)
def test_env():
    env_name = "OpenCabinetDrawerGripper_33930_link_0-v0"
    env = gym.make(env_name)
    env.set_env_mode(obs_mode="pointcloud", reward_type="sparse")
    obs = env.reset(level=0)
