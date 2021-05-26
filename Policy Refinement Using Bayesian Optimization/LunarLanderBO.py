"""
    Code for collecting failure trajectories using Bayesian Optimization for environment Lunar Lander
    Project : Policy correction using Bayesian Optimization
    Description : The file contains functions for computing failure trajectories given RL policy and
    safety specifications
"""

import numpy as np
import gym
import GPyOpt
from numpy.random import seed
from eval_policy import display
import gym
from network import FeedForwardActorNN
import torch
import pickle

'''
    Bayesian Optimization module for uncovering failure trajectories

    Safety Requirement
    # Requirement 1: The lander should not fall down in any trajectory
'''

#=============================================Global Variables =================================#
policy = None
env = None
traj_spec_dic = {}
traj_count = 0
index_count = 0


'''
    The function called from within the bayesian optimization module
    parameters : bounds containing the sampled variables of the state vector
    return : calls specification function and computes and returns the minimum value
'''
def sample_trajectory(bounds):
    global policy, env, traj_spec_dic,traj_count, index_count
    selected_seed = env.seed(None)
    x1 = bounds[0][0]
    x2 = bounds[0][1]
    x3 = bounds[0][2]
    obs = env.reset()
    env.env.lander.position[0] = env.lander.position[0]+x1
    env.env.lander.position[1] = env.lander.position[1]+x2
    env.env.lander.linearVelocity[0] = env.lander.linearVelocity[0]+3
    obs = torch.Tensor(env.env.state)
    #print(f'env.env.state =========== {env.env.state}')
    ep_ret = 0
    ep_ret, traj, iter = display(obs,policy,env,False)
    additional_data = {'reward':ep_ret}
    #Create trajectory to be sent to safety specification
    traj = (traj, additional_data)
    #print(f'trajectory ========== {traj}')
    specification_evaluation = safet_spec_1(traj)
    #Store the set of trajectories with negative evaluation
    if specification_evaluation<0:
        #remove this trajectory from sample space
        #Here we are using a random seed for each environment
        traj_spec_dic[traj_count] = (traj[0],specification_evaluation,selected_seed,(x1,x2,x3))
        traj_count = traj_count + 1
        #print(f'x1=========={x1}=========x2==={x2}======x3======={x3}')
        #print(f'traj=============={traj[0][0]}========selected_seed=========={selected_seed}=======specification_evaluation ========== {specification_evaluation}')
    return specification_evaluation


def run_BO():
    np.random.seed(123456)
    bounds = [{'name': 'x1', 'type': 'continuous', 'domain': (0, 10)}, # Position x
              {'name': 'x2', 'type': 'continuous', 'domain': (0, 20)}, # Position y
              {'name': 'x3', 'type': 'continuous', 'domain': (0, 3)},
              {'name': 'x3', 'type': 'continuous', 'domain': (0, 3)}] # velocity_x # velocity_y
    max_iter = 200
    myProblem = GPyOpt.methods.BayesianOptimization(sample_trajectory, bounds, acquisition_type='EI', exact_feval=True, de_duplication = True)
    myProblem.run_optimization(max_iter, max_time=None, eps=1e-6, verbosity=True)
    print(myProblem.fx_opt)



# 1. This is an implication example where two predicates get jointly optimized
def safet_spec_1(traj):
    reward = traj[1]['reward']
    traj = traj[0]
    if(reward<0):
        for state in traj:
            if(state[1] < 0.1 and (state[4] > 1 or state[4] < -1)):
                #print(f'position y ========= {state[1]} ======= angle ========= {state[4]} ==== vertical speed ====== {state[3]}')
                reward = reward - (0.1-state[1] + (state[4]-1))
    return reward


# 2. The lander should not go beyond the flag -0.4<=x_pos<=0.4
def safet_spec_2(traj):
    reward = traj[1]['reward']
    traj = traj[0]
    for state in traj:
        last_state = state[0]
    #print(f'last_state ========= {last_state}')
    return 0.4-last_state



if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    seed = 0
    env.seed(seed)
    actor_model = 'Policies/ppo_actorLunarLanderContinuous-v2.pth'
    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardActorNN(obs_dim, act_dim,False)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))
    run_BO()
    print(f'Number of failure trajectories ========== {len(traj_spec_dic)}')
    with open('failure_trajectory_lunar_implication.data', 'wb') as filehandle1:
        # store the observation data as binary data stream
        pickle.dump(traj_spec_dic, filehandle1)
