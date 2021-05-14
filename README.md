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

### Training

The main program takes the following arguments

1) env : environment name (default is LunarLanderContinuous-v2)
2) actor : filepath to the actor network (default is ppo_actorLunarLanderContinuous-v2.pth)
3) critic : filepath to the critic network (default is ppo_criticLunarLanderContinuous-v2.pth)
4) failuretraj : The filepath to the failure trajectory path (default is failure_trajectory_lunar.data)

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

### Generating Failure trajectories for a specific environment

Each environment has a seperate Bayesian Optimization file. Run the Bayesian Optimization correspondig to the environment
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
Mention the policy and the failure trajectory file

### Training the sub-policy

```
$ python main.py --subtrain
```

### Update the original Policy to new Policy via gradient based updates

```
$ python main.py --correct
```
The correct method takes the actor and critic networks of the old policy and the subpolicy as an argument


### Calculate the distance between the original policy and the updated policy

```
$ python main.py --distance
```

The correct method takes the actor network of the old policy and the updated policy as an argument
