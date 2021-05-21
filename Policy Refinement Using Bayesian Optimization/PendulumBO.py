"""
    Code for collecting failure trajectories using Bayesian Optimization
    Author : Briti Gangopdahyay
    Project : Policy correction using Bayesian Optimization
    Description : The file contains functions for computing failure trajectories given RL policy and
    safety specifications
    Formal Methods Lab, IIT Kharagpur
"""

import sys
import numpy as np
import gym
import GPy
import GPyOpt
from numpy.random import seed
import matplotlib
from eval_policy import choose_best_action
import gym
from network import FeedForwardActorNN, FeedForwardCriticNN
import torch
import pickle

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
def sample_trajectory(bounds):
    global policy, env, traj_spec_dic,traj_count
    x1 = bounds[0][0]
    x2 = bounds[0][1]
    x3 = bounds[0][2]
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


def run_BO():
    bounds = [{'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)}, # Bounds on theta
              {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}, # Bounds on theta dot
              {'name': 'x3', 'type': 'continuous', 'domain': (-1, 1)}] # Bounds on speed
    max_iter = 200
    myProblem = GPyOpt.methods.BayesianOptimization(sample_trajectory, bounds, acquisition_type='EI', exact_feval=True)
    myProblem.run_optimization(max_iter)
    print(myProblem.fx_opt)




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
    actor_model = 'ppo_actorPendulum-v0.pth'
    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardActorNN(obs_dim, act_dim,False)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))
    run_BO()
    #print(f'traj_spec_dic ========== {traj_spec_dic}')
    with open('failure_trajectory_pendulum.data', 'wb') as filehandle1:
        # store the observation data as binary data stream
        pickle.dump(traj_spec_dic, filehandle1)

