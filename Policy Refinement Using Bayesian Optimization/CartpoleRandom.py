"""
    Code for collecting failure trajectories using Bayesian Optimization
    Project : Policy correction using Bayesian Optimization
    Description : The file contains functions for computing failure trajectories given RL policy and
    safety specifications
"""

import numpy as np
import gym
import GPyOpt
from numpy.random import seed
from eval_policy import choose_best_action
import gym
from network import FeedForwardActorNN
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
def sample_trajectory(sample_1,sample_2,sample_3,sample_4):
    global policy, env, traj_spec_dic,traj_count
    x1 = sample_1
    x2 = sample_2
    x3 = sample_3
    x4 = sample_4
    obs = np.array([x1,x2,x3,x4])
    #print(f'obs =========== {obs}')
    max_steps = 400
    env.reset()
    env.env.state = obs
    traj = [obs]
    actions = []
    reward = 0
    iters= 0
    ep_ret = 0
    done = False
    for _ in range(max_steps):
        iters+=1
        action = choose_best_action(obs,policy)
        actions.append(action)
        obs, rew, done, _ = env.step(action)
        #add the observation state to the current trajectory
        traj.append(obs)
        ep_ret += rew
        if done:
            break
    additional_data = {'reward':ep_ret, 'mass':env.env.total_mass}
    #Create trajectory to be sent to safety specification
    traj = (traj, additional_data)
    #print(f'trajectory ========== {traj}')
    specification_evaluation = safety_spec(traj)
    #Store the set of trajectories with negative evaluation
    if specification_evaluation<0:
        traj_spec_dic[traj_count] = (traj[0],specification_evaluation)
        traj_count = traj_count + 1
    print(f'specification_evaluation ========== {specification_evaluation}')
    return specification_evaluation


def run_Random():
    x1_max = 0.05
    x1_min = -0.05
    x2_max = 0.05
    x2_min = -0.05
    x3_max = 0.05
    x3_min = -0.05
    x4_max = 0.05
    x4_min = -0.05
    # generate a random sample from the domain
    sample_1 = x1_min + rand(1000) * (x1_max - x1_min)
    sample_2 = x2_min + rand(1000) * (x2_max - x2_min)
    sample_3 = x3_min + rand(1000) * (x3_max - x3_min)
    sample_4 = x4_min + rand(1000) * (x4_max - x4_min)
    sample = list()
    step = 0.1
    print(f'sample length ========== {len(sample_1)}')
    for i in range(len(sample_1)):
        val = sample_trajectory(sample_1[i],sample_2[i],sample_3[i],sample_4[i])
        print(f'sample1 =========== {sample_1[i]} ======== sample2 ==== {sample_2[i]} ==== sample3 ===== {sample_3[i]}')
    '''sample = list()
    step = 0.04
    for sample_1 in arange(x1_min, x1_max+step, step):
        for sample_2 in arange(x2_min, x2_max+step, step):
            for sample_3 in arange(x3_min, x3_max+step, step):
                for sample_4 in arange(x4_min, x4_max+step, step):
                    sample.append([sample_1,sample_2,sample_3,sample_4])
    print(f'sample length ========== {len(sample)}')
    for i in range(len(sample)):
        val = sample_trajectory(sample[i][0],sample[i][1],sample[i][2],sample[i][3])
        print(f'sample1 =========== {sample[i][0]} ======== sample2 ==== {sample[i][1]} ==== sample3 ===== {sample[i][2]}')'''


# 1. Always stay within the region (-2.4, 2.4)
def safety_spec(traj):
    traj = traj[0]
    #print(f'traj ========== {traj}')
    x_s = np.array(traj).T[0]
    #print(f'min value ========== {x_s}')
    return min(2.4 - np.abs(x_s))

# 2. Maintain a momentum >=-2.0 and <= 2.0
def safet_spec_2(traj):
    traj_ = traj[0]
    #print(f'traj ========== {traj}')
    mass = traj[1]['mass']
    v_s = np.array(traj_).T[1]
    return min(2. - np.abs(mass*v_s))


# 3. The angle made by the cartpole should <=0.2 within the rest position
def safet_spec_3(traj):
    traj = traj[0]
    theta=np.array(traj).T[2]
    #print(f'theta ========== {theta}')
    return min(0.2 - np.abs(theta))



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    seed = 0
    env.seed(seed)
    actor_model = 'Policies/ppo_actor_updatedCartPole-v0.pth'
    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n #env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardActorNN(obs_dim, act_dim,True)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))
    run_Random()
    print(f'Length trajectory ========== {len(traj_spec_dic)}')
    with open('failure_trajectory_cartpole.data', 'wb') as filehandle1:
        # store the observation data as binary data stream
        pickle.dump(traj_spec_dic, filehandle1)

