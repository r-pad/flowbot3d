# FlowBot3D Code Release

This is the official code release for our RSS 2022 paper:

**FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects**  
*Ben Eisner\*, Harry Zhang\*, David Held*  
Best Paper Finalist, RSS 2022  
[arXiv](https://arxiv.org/abs/2205.04382) | [RSS 2022 Proceedings](http://www.roboticsproceedings.org/rss18/p018.html) | [Website](https://sites.google.com/view/articulated-flowbot-3d/home)  

```
@INPROCEEDINGS{Eisner-RSS-22, 
    AUTHOR    = {Ben Eisner AND Harry Zhang AND David Held}, 
    TITLE     = {{FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2022}, 
    ADDRESS   = {New York City, NY, USA}, 
    MONTH     = {June}, 
    DOI       = {10.15607/RSS.2022.XVIII.018} 
} 
```

Questions? Open an Issue, or send an email to:  
`ben [dot] a [dot] eisner [at] gmail [dot] com`


## A note on reproducibility
In the codebase's current form there is some nondeterminism in the ManiSkill environment evals, which we have not yet tracked down yet. Therefore, the results that this codebase will generate may vary slightly from the results presented in Tables I and II in the paper (we're currently observing +/- 4% success rate across runs). However, the relative ranking of each method is still consistent with the results presented in the original manuscript. Once we find the source, we will update this repository accordingly.

## Versions

* 2023/02/07: Intial code release.

# Installation

This codebase is structured as a **standard Python package**, which means that we've designed it to be installed in whatever Python environment you want with minimal hassle - and definitely no path-mangling.

```

# Clone the repository
git clone --recurse-submodules https://github.com/r-pad/flowbot3d

# Enter the flowbot directory.
cd flowbot3d

# Install torch and pytorch-geometric, must be done separately (for now).
pip install torch==1.13.1
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Install this repository. Optionally, omit the "-e" if you don't need it to be editable, i.e. if you're just running our models as-is.
pip install -e .

# Install our custom fork of ManiSkill
pip install -e ./third_party/ManiSkill

# [Optional] Install ManiSkill Learn, if you want to reproduce our Behavior Cloning baseline.
pip install -e "git+https://github.com/haosulab/ManiSkill-Learn.git#egg=mani-skill-learn"
```

## Major external dependencies

* PyTorch - all neural network training is PyTorch under the hood.
* PyTorch Lightning - for training loops, WandB integration, etc.
* PyTorch Geometric - for implementing Point Cloud neural networks (i.e. PointNet++) which can handle "ragged" batches - i.e. batches of point clouds where there may be a variable number of points per sample.
* PartNet-Mobility / SAPIEN / ManiSkill - the core object dataset and simulation environment.
* Pybullet - for rendering PartNet-Mobility objects.
* Where2Act - for a Panda gripper model.
* UMPNet - for evaluation environments.

## Our custom dependencies

FlowBot3D depends on a couple custom Python packages, some of which are home-grown from our lab ([R-PAD](https://r-pad.github.io/)), and some of which are modified

* [rpad-partnet-mobility-utils](https://github.com/r-pad/partnet_mobility_utils) - a nice Python library which wraps the PartNet-Mobility dataset, and allows access to various object properties without going through SAPIEN. Also has a custom PyBullet rendering environment.
* [rpad-pyg](https://github.com/r-pad/pyg_libs) - Some Pytorch Geometric models and utilities (i.e. dataset caching) we've implemented. Notably our impelmentation of PointNet++ for dense prediction.
* [rpad-visualize-3d](https://github.com/r-pad/visualize_3d) - Some utilities for visualizing point clouds and flows in Plotly.
* [Fork of ManiSkill](https://github.com/harryzhangOG/ManiSkill/) - Custom modifications of ManiSkill for this paper.
* [Fork of UMPNet](https://github.com/harryzhangOG/umpnet) - Custom fork of UMPNet to run our models in their environments.

# Code Structure

* `docs/`: Documentation for the (eventual) comprehensive documentation for the library.
* `flowbot3d/`: The library
  * `datasets/`
    * `flow_dataset.py`: Simple numpy flow dataset.
    * `flow_dataset_pyg.py`: Pytorch Geometric wrapper of flow dataset.
    * `screw_dataset_pyg.py`: Screw parameter dataset for the ScrewNet baseline. Also in Pytorch Geometric.
  * `grasping/`: Everything for the ManiSkill baseline
    * `agents/`: Agents which can operate in the environment.
    * `env/`: The custom environment on top of ManiSkill.
  * `models/`: The different flow models we can train.
  * `evaluate_flow_models.py`: Generate some tables for evaluating how good ArtFlowNet is vs. ground truth.
  * `evaluate_maniskill.py`: The main script for running ManiSkill evaluations.
  * `train.py`: The main script for training ArtFlowNet models (and baselines)
* `notebooks/`:
  * `explore_dataset.ipynb`: Visualize the Flow dataset.
  * `flowbot_predictions.ipynb`: Visualize and quantify predictions of ArtFlowNet and its ablations on the raw dataset.
  * `maniskill_results.ipynb`: Aggregate the results of ManiSkill evaluations into a few tables. Ideally should exactly be Tables I and II in our paper.
* `results/`: Not in the repository by default, but will be generated if you run any evaluations.
* `scripts/`:
  * `generate_dataset.py`: Simply generates the cached version of the Flow dataset we used for fast training.
* `tests/`: Some unit tests.
* `third_party/`: Submodules we've checked in.
* `umpnet_obj_splits/`: Files describing which ManiSkill environments constitute each evaluation group. Based on the UMPNet paper's original splits.
* `umpnet_data_split.json`: Object splits for training, also from UMPNet.

# I Want To...

## ...Use the Flow dataset you used.

First, you have to [download the PartNet-Mobility dataset](https://sapien.ucsd.edu/downloads), and extract it somewhere. Then:

1. If you only want to access the PartNet-Mobility dataset with 3D Articulation Flow (3DAF) annotations, and don't necessarily care about which exact random states were used during training:

```
from flowbot3d.datasets.flow_dataset import Flowbot3DDataset

dset = Flowbot3DDataset(
    root=PATH_TO_PARTNET_MOBILITY,
    split="umpnet-train-train",  # multiple split options.
    n_points=1200,  # how many points to sample. 
)

data = dset.get_data(OBJ_ID)
```

2. If you want the exact dataset we used to train on, you can generate it yourself by running:

```
python scripts/generate_dataset.py --split "umpnet-train-train"
python scripts/generate_dataset.py --split "umpnet-train-test"
python scripts/generate_dataset.py --split "umpnet-test"
```

This will launch a parallel sampling job on your machine - you can control the number of cores used with ``--n-proc``. It samples each object 100 times in random configuraitons, and saves the output renders as Pytorch Geometric objects. The generation process is fully deterministic based on the provided seed.

3. Download it (TBD). 

## ...Use a pre-trained ArtFlowNet model to predict 3DAF.

Right now the easiest way to use the model is to load a Pytorch Lightning checkpoint file (pre-trained version TBD).

```
from flowbot3d.models.artflownet import ArtFlowNet

ckpt_file = PATH_TO_CKPT_FILE
model = ArtFlowNet.load_from_checkpoint(ckpt_file)
flow = model.predict(xyz, mask)
```

## ...Train ArtFlowNet from scratch.

You can use `train.py` to train. Lots of options to set.

```python flowbot3d/train.py --pm-root ${DATASET_PATH} --dataset umpnet --epochs=100 --model-type flowbot```

## ...Use the ManiSkill environments we created in my own codebase.

The ManiSkill environmetnts are just Gym environments.

```
import gym
import flowbot3d.grasping.env

env = gym.make("OpenCabinetDrawerGripper_33930_link_0-v0")
obs = env.reset(level=0)
```

## ...Evaluate my own agent on the environment.

1. Implement a custom [`PCAgent`](https://github.com/r-pad/flowbot3d/blob/main/flowbot3d/evaluate_maniskill.py#L65) that takes point cloud observations and outputs an action (and an optional dictionary of extras).
2. Modify [`create_agent`](https://github.com/r-pad/flowbot3d/blob/main/flowbot3d/evaluate_maniskill.py#L260) to instantiate your agent when you want it to.
3. Run `python flowbot3d/evaluate_maniskill.py --model_name YOUR_MODEL`, probably with some extra flags.

# How to reproduce all the tables from our paper

There are many commands to run. For each command, assume that:

```PARTNET_MOBILITY_PATH=/path/to/your/download```

## Train Flow models

### FlowBot3D

### FlowBot3D w/o Mask

### FlowBot3D w/o Mask + VPA

### UMPNet-DI

### ScrewNet

## Train Behavior Cloning Agents

Behavioral cloning baselines rely on infrustructure in our custom version of ManiSkill-Learn. Note that this is based on a very early version of ManiSkill-Learn, so it might be very different from the official code base now.

When the custom ManiSkill-Learn repo is installed correctly, we can then collect expert data for our behavioral cloning agents. Specifically, as indicated in the paper, we use an oracle ground-truth-flow-based policy to collect expert trajectories for BC. We run the following script to collect trajectories:

```
python flowbot3d/grasping/agents/bc/bc_datagen_gt_flow_grasping.py
```
This will sequentially generate demonstration trajectories based on the oracle policy for all the available environments being passed in. The generated data will then be used to train the BC agent in ManiSkill-Learn.

In ManiSkill-Learn, first convert the collected default trajectories to point clouds using

```python flowbot3d/grasping/agents/bc/convert_pm_obj_demo_data.py```

This would automatically generate point clouds data that the PointNet/Transformer BC agent understands. Then we would run the training-evaluation-logging meta script to train on all environments:

```python flowbot3d/grasping/agents/bc/run_bc_from_demo.py```

This script loads the demo data, trains the agent using the data, evaluates the agent and logs the numbers to a text file.


### BC

TODO

### DAgger E2E

TODO

### DAgger Oracle

TODO

### BC + F

TODO

### DAgger E2E + F

TODO

### DAgger Oracle + F

TODO

## Run ManiSkill Evals

### FlowBot3D

### FlowBot3D w/o Mask

### FlowBot3D w/o Mask + VPA

### UMPNet-DI

### ScrewNet

### Normal Direction

### Oracle w/ GT 3DAF

### BC

TODO

### DAgger E2E

TODO

### DAgger Oracle

TODO

### BC + F

TODO

### DAgger E2E + F

TODO

### DAgger Oracle + F

TODO

## Run UMPNet Evals

Follow instructions in [this README](https://github.com/harryzhangOG/umpnet/tree/main/umpnet#readme).


# Some things left TODO...

While we've release a (mostly) functional codebase, there are a couple of pieces which aren't quite as polished + final as we'd like them to be. They're on our (admittedly quite full) plates to get done eventually, or as the community requests.

- [] Fill in the `maniskill_results.ipynb` tables with all of our evals.
- [] Make the ManiSkill evaluation deterministic.
- [] Merge the BC and DAgger baseline loop into the `evaluate_maniskill.py`.
- [] Modify the UMPNet baseline code so that it's easier to run.
- [] Clean up our fork of ManiSkill.
- [] Clean up documentation + website
- [] Release pre-trained models.
- [] Release a zip of the specific datasets we used (although dataset generation is deterministic).
- [] Release the custom ROS package we created to run real-world experiments.
- [] Make this codebase installation process easier.
