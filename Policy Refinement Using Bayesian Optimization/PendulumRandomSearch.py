"""
    Code for collecting failure trajectories using Random Search
    Project : Policy correction using Bayesian Optimization
    Description : The file contains functions for computing failure trajectories given RL policy and
    safety specifications
"""

from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

import numpy as np

import gym
import GPyOpt
from numpy.random import seed
from eval_policy import choose_best_action
import gym
from network import FeedForwardActorNN, FeedForwardCriticNN
import torch
import pickle
from numpy import arange
from numpy.random import rand

'''
    Bayesian Optimization module for uncovering failure trajectories

    Safety Requirement
    # Requirement 1: We would like the cartpole to not travel more than a certain
    # distance from its original location(2.4) 
    # Always stay within the region (-2.4, 2.4)
'''

#=============================================Global Variables =================================#
policy = None
env = None
traj_spec_dic = {}
traj_count = 0


'''
    The function called from within the bayesian optimization module
    parameters : bounds containing the sampled variables of the state vector
    return : calls specification function and computes and returns the minimum value
'''
def sample_trajectory(sample_1,sample_2,sample_3):
    global policy, env, traj_spec_dic,traj_count
    x1 = sample_1
    x2 = sample_2
    x3 = sample_3
    obs = np.array([x1,x2,x3])
    max_steps = 200
    env.reset()
    env.env.state = obs[0:2]
    print(f'env.env.state =========== {env.env.state}')
    traj = [obs]
    actions = []
    iters= 0
    ep_ret = 0
    done = False
    for _ in range(max_steps):
        iters+=1
        action = policy(obs).detach().numpy()
        actions.append(action)
        obs, rew, done, _ = env.step(action)
        #add the observation state to the current trajectory
        traj.append(obs)
        ep_ret += rew
        if done:
            break
    additional_data = {'reward':ep_ret}
    #Create trajectory to be sent to safety specification
    traj = (traj, additional_data)
    #print(f'trajectory ========== {traj}')
    specification_evaluation = safet_spec_2(traj)
    #Store the set of trajectories with negative evaluation
    if specification_evaluation<0:
        traj_spec_dic[traj_count] = (traj[0],specification_evaluation)
        traj_count = traj_count + 1
    print(f'specification_evaluation ========== {specification_evaluation}')
    return specification_evaluation


def run_Random():
    #np.random.seed(123456)
    x1_max = 1
    x1_min = -1
    x2_max = 1
    x2_min = 0
    x3_max = 1
    x3_min = -1
    # generate a random sample from the domain
    sample_1 = x1_min + rand(1000) * (x1_max - x1_min)
    sample_2 = x2_min + rand(1000) * (x2_max - x2_min)
    sample_3 = x3_min + rand(1000) * (x3_max - x3_min)
    print(f'sample length ========== {len(sample_1)}')
    for i in range(len(sample_1)):
        val = sample_trajectory(sample_1[i],sample_2[i],sample_3[i])
        print(f'sample1 =========== {sample_1[i]} ======== sample2 ==== {sample_2[i]} ==== sample3 ===== {sample_3[i]}')
    '''sample = list()
    step = 0.2
    for sample_1 in arange(x1_min, x1_max+step, step):
        for sample_2 in arange(x2_min, x2_max+step, step):
            for sample_3 in arange(x3_min, x3_max+step, step):
                sample.append([sample_1,sample_2,sample_3])
    print(f'sample length ========== {len(sample)}')
    for i in range(len(sample)):
        val = sample_trajectory(sample[i][0],sample[i][1],sample[i][2])
        print(f'sample1 =========== {sample[i][0]} ======== sample2 ==== {sample[i][1]} ==== sample3 ===== {sample[i][2]}')
        #print(f'sample1 =========== {sample_1[i]} ======== sample2 ==== {sample_2[i]} ==== sample3 ===== {sample_3[i]}')'''




# 1. Find the initial condition such that the pendulum stabilizes to 0
def safet_spec_1(traj, gamma=0.25):
    traj = traj[0]
    cos_thetas = np.array(traj).T[0]
    theta_dots = np.array(traj).T[2]
    stab_vals = 0
    for ct, td in zip(cos_thetas, theta_dots):
        stab_vals = np.abs(np.arccos(ct))**2 + np.abs(td)**2 + stab_vals*gamma
    return -stab_vals


# 1. Find the initial condition such that the pendulum stabilizes to 0
def safet_spec_2(traj):
    traj = traj[1]
    reward = traj['reward']
    return 300+reward



if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    seed = 0
    env.seed(seed)
    actor_model = 'Policies/ppo_actor_updatedPendulum-v0.pth'
    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardActorNN(obs_dim, act_dim, False)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))
    run_Random()
    print(f'Length trajectory ========== {len(traj_spec_dic)}')
    with open('failure_trajectory_pendulum.data', 'wb') as filehandle1:
        # store the observation data as binary data stream
        pickle.dump(traj_spec_dic, filehandle1)

