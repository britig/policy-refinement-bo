# Policy Refinement Using Bayesian Optimization

# System Requirements

- Ubuntu 20.04.2 LTS

## Installation

1. Setup conda environment

```
$ conda create -n env_name python=3.8.5
$ conda activate env_name
```
2. Clone the repository to an appropriate folder
3. Install requirements

```
$ pip install -r requirements.txt
$ pip install -e .
```

## Usage

### Training

The main program takes the following arguments

1) env : environment name (default is LunarLanderContinuous-v2
2) actor : filepath to the actor network (default is ppo_actorLunarLanderContinuous-v2.pth)
3) critic : filepath to the critic network (default is ppo_criticLunarLanderContinuous-v2.pth)
4) failuretraj : The filepath to the failure trajectory path

Note : I generally change the default argument in the main.py file otherwise the command line becomes too long

To train a model run:

```
$ python main.py --train
```
The hyperparameters can be changed in the hyperparameters.yml file

### Testing

To test a trained model run:

```
$ python main.py --test
```

Press ctr+c to end testing

# G
