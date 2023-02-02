# flowbot3d

## Installation

```
pip install torch

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

pip install -e "git+https://github.com/harryzhangOG/ManiSkill.git#egg=mani-skill"

pip install -e ./third_party/ManiSkill

pip install -e "git+https://github.com/haosulab/ManiSkill-Learn.git#egg=mani-skill-learn"

pip install -e .
```


# Reproduce the paper.

## Training models.

### FlowBot3D

FlowBot3D

FlowBot3D w/o Mask

FlowBot3D w/o Mask (+VPA)

### UMP-DI

### Screw Parameters

### Behavioral Cloning
Behavioral cloning baselines rely on infrustructure in our custom version of ManiSkill-Learn. To try out the BC baselines, first clone the following repository to your machine:
```
git clone https://github.com/harryzhangOG/ManiSkill-Learn
```
Note that this is based on a very early version of ManiSkill-Learn, so it might be very different from the official code base now.

When the custom ManiSkill-Learn repo is installed correctly, we can then collect expert data for our behavioral cloning agents. Specifically, as indicated in the paper, we use an oracle ground-truth-flow-based policy to collect expert trajectories for BC. We run the following script to collect trajectories:
```
python -m flowbot.baselines.bc_datagen_gt_flow_grasping
```
This will sequentially generate demonstration trajectories based on the oracle policy for all the available environments being passed in. The generated data will then be used to train the BC agent in ManiSkill-Learn.

In ManiSkill-Learn, first convert the collected default trajectories to point clouds using
```
python convert_pm_obj_demo_data.py
```
This would automatically generate point clouds data that the PointNet/Transformer BC agent understands. Then we would run the training-evaluation-logging meta script to train on all environments:
```
python run_bc_from_demo.py
```
This script loads the demo data, trains the agent using the data, evaluates the agent and logs the numbers to a text file. 
