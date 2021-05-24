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

All the trained policies, sub-policies and updated policies are avialable in the policies folder

### Important step

Navigate to the installed gym folder and replace the box2d and mujoco folders with the one in this repository. We have changed some private variables to class variables to acceess them from outside.

### Training

The trained policies are available in the policies folder and need not be trained again

The main program takes the following arguments

1) env : environment name (default is LunarLanderContinuous-v2)
2) actor : filepath to the actor network (default is Policies/ppo_actorLunarLanderContinuous-v2.pth)
3) critic : filepath to the critic network (default is Policies/ppo_criticLunarLanderContinuous-v2.pth)
4) failuretraj : The filepath to the failure trajectory path (default is Failure_Trajectories/failure_trajectory_lunar_implication.data)

Note : Change the default arguments in the main.py file otherwise the command line may become too long

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

### Generating Failure trajectories for a specific environment

Each environment has a seperate Bayesian Optimization file. Run the Bayesian Optimization correspondig to the environment
We use GpyOpt Library for Bayesian Optimization. As per (https://github.com/SheffieldML/GPyOpt/issues/337) GpyOpt has stochastic evaluations
This may lead to identification of a different number failure trajectories (higher or lower) than what is reported in the paper
For example to generate failure trajectories for the Lunar Lander environment run:

```
$ python LunarLanderBO.py
```

The failure trajectories will be written in the corresponding data files

### Displaying Failure trajectories

To display failure trajectories:

```
$ python main.py --display
```
Mention the policy and the failure trajectory file in arguments or in the main.py file

Change the actor_model argument for observing the behaviour of sub-policy and updated policy on the failure trajectories


### Training the sub-policy

```
$ python main.py --subtrain
```

Mention the failure trajectory file in arguments or in the main.py file

### Update the original Policy to new Policy via gradient based updates

```
$ python main.py --correct
```
The correct method takes the actor and critic networks of the old policy and the subpolicy as an argument

default function parameters are 
correct_policy(env,'Policies/ppo_actorCartPole-v0.pth','Policies/ppo_criticCartPole-v0.pth','ppo_actor_subpolicyCartPole-v0.pth','ppo_critic_subpolicyCartPole-v0.pth',is_discrete,failure_trajectory)


### Calculate the distance between the original policy and the updated policy

```
$ python main.py --distance
```
default function parameters are:

compute_distance('Policies/ppo_actorLunarLanderContinuous-v2.pth','Policies/ppo_actor_updatedLunarLanderContinuous-v2.pth',env,is_discrete)
