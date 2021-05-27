"""
	The main file to run the policy correction code
"""



import torch
from ppoPolicyTraining import PPO, test
from network import FeedForwardActorNN
import pickle
from eval_policy import display
import numpy as np
#Update failure network
from UpdateNetwork import correct_policy
from Utility import compute_distance, set_environment
import argparse
import yaml


if __name__ == '__main__':
	#=============================== Environment and Hyperparameter Configuration Start ================================#
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--env', dest='env', action='store_true', help='environment_name')
	parser.add_argument('--train', dest='train', action='store_true', help='train model')
	parser.add_argument('--test', dest='test', action='store_true', help='test model')
	parser.add_argument('--display', dest='display', action='store_true', help='Display Failure Trajectories')
	parser.add_argument('--actor', dest='actor', action='store_true', help='Actor Model')
	parser.add_argument('--critic', dest='critic', action='store_true', help='Critic Model')
	parser.add_argument('--failuretraj', dest='failuretraj', action='store_true', help='File name with failure trajectories')
	parser.add_argument('--subtrain', dest='subtrain', action='store_true', help='Training a subpolicy')
	parser.add_argument('--correct', dest='correct', action='store_true', help='Correct the orginal policy')
	parser.add_argument('--distance', dest='distance', action='store_true', help='Computing distance between two subpolicy')
	parser.add_argument('--isdiscrete', dest='isdiscrete', action='store_true', help='Environment discrete or continuous')
	parser.add_argument('--oldactor', dest='oldactor', action='store_true', help='Old Actor Network')
	parser.add_argument('--oldcritic', dest='oldcritic', action='store_true', help='Old Critic Network')
	parser.add_argument('--subactor', dest='subactor', action='store_true', help='Sub Actor Network')
	parser.add_argument('--subcritic', dest='subcritic', action='store_true', help='Sub Critic Network')
	parser.add_argument('--newactor', dest='newactor', action='store_true', help='New Actor Network')
	args = parser.parse_args()
	actor_model = None
	critic_model = None
	failure_trajectory = None
	env_name = None
	is_discrete = False
	old_actor = None
	old_critic = None
	sub_actor = None
	sub_critic = None
	new_actor = None
	if args.env:
		env_name = args.env
	else:
		env_name = 'Pendulum-v0'
	if args.isdiscrete:
		is_discrete = args.isdiscrete
	if args.actor:
		actor_model = args.actor
	else:
		actor_model = 'Policies/ppo_actorPendulum-v0.pth'
	if args.critic:
		critic_model = args.critic
	else:
		critic_model = 'Policies/ppo_criticPendulum-v0.pth'
	if args.failuretraj:
		failure_trajectory = args.failuretraj
	else:
		failure_trajectory = 'Failure_Trajectories/failure_trajectory_pendulum.data'
	if args.oldactor:
		old_actor = args.oldactor
	else:
		old_actor = 'Policies/ppo_actorPendulum-v0.pth'
	if args.oldcritic:
		old_critic = args.oldcritic
	else:
		old_critic = 'Policies/ppo_criticPendulum-v0.pth'
	if args.subactor:
		sub_actor = args.subactor
	else:
		sub_actor = 'Policies/ppo_actor_subpolicyPendulum-v0.pth'
	if args.subcritic:
		sub_critic = args.subcritic
	else:
		sub_critic = 'Policies/ppo_critic_subpolicyPendulum-v0.pth'
	if args.newactor:
		new_actor = args.newactor
	else:
		new_actor = 'Policies/ppo_actor_updatedPendulum-v0.pth'

	env = set_environment(env_name,0)
	with open('hyperparameters.yml') as file:
		paramdoc = yaml.full_load(file)
	#=============================== Environment and Hyperparameter Configuration End ================================#
	#=============================== Original Policy Training Code Start ================================#
	if args.train:
		for item, param in paramdoc.items():
			if(str(item)==env_name):
				hyperparameters = param
				print(param)
		model = PPO(env=env, **hyperparameters)
		model.learn(env_name, [], False)
	#=============================== Original Policy Training Code End ================================#
	#=============================== Policy Testing Code Start ==========================#
	if args.test:
		test(env,actor_model, is_discrete)
	#=============================== Policy Testing Code End ============================#
	#=============================== Computing Failure Trajectories Code Start ==========================#

	#=============================== Computing Failure Trajectories Code End  ==========================#
	#=============================== Displaying Failure Trajectories Code Start  ==========================#
	if args.display:
		with open(failure_trajectory, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		print(f'Number of failure trajectories=========={len(failure_observations)}')
		obs_dim = env.observation_space.shape[0]
		if is_discrete:
			act_dim = env.action_space.n
		else:
			act_dim = env.action_space.shape[0]
		#act_dim = env.action_space.n
		policy = FeedForwardActorNN(obs_dim, act_dim,is_discrete)
		policy.load_state_dict(torch.load(actor_model))
		traj_count = 0
		traj_spec_dic = {}
		#Had to be done to set the environment to the same state as when it was sampled
		if env_name == 'BipedalWalker-v3':
			count = 0
			for i in range(len(failure_observations)):
				seed = failure_observations[i][2]
				#print(seed)
				env.seed(seed[0])
				env.reset()
				ep_ret, traj, iter  = display(failure_observations[i][0][0],policy,env,False)
		elif env_name == 'LunarLanderContinuous-v2':
			for i in range(len(failure_observations)):
				seed = failure_observations[i][2]
				#print(seed)
				env.seed(seed[0])
				env.reset()
				disturbances = failure_observations[i][3]
				env.env.lander.position[0] = env.lander.position[0]+disturbances[0]
				env.env.lander.position[1] = env.lander.position[1]+disturbances[1]
				env.env.lander.linearVelocity[0] = env.lander.linearVelocity[0]+3
				#print(f'Disturbances======{disturbances[0]}======={disturbances[1]}')
				#print(f'SAMPLED observation======{failure_observations[i][0][0]}')
				ep_ret, traj, iter  = display(failure_observations[i][0][0],policy,env,False)
		else:
			for i in range(len(failure_observations)):
				env.reset()
				#print(f'SAMPLED observation======{failure_observations[i][0]}=======ACTION======{failure_observations[i][1]}')
				ep_ret, traj, iter = display(failure_observations[i][0][0],policy,env,is_discrete)
	#=============================== Displaying Failure Trajectories Code End  ==========================#
	#=============================== Sub Policy Learning for Failure Trajectories Code Start  ==========================#
	if args.subtrain:
		env_sub_name = env_name+'-sub'
		for item, param in paramdoc.items():
			if(str(item)==env_sub_name):
				hyperparameters = param
				print(param)
		with open(failure_trajectory, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		model = PPO(env=env, **hyperparameters)
		model.learn(env_name, failure_observations , True)
	#=============================== Sub Policy Learning for Failure Trajectories Code End  ==========================#
	#=============================== Policy Correction for Failure Trajectories Code Start  ==========================#
	if args.correct:
		correct_policy(env,old_actor,old_critic,sub_actor,sub_critic,is_discrete,failure_trajectory)
	#=============================== Policy Correction for Failure Trajectories Code End  ==========================#
	#=============================== Compute Distance between two policies Code Start  ==========================#
	if args.distance:
		#For finding out standard deviation
		distance_list = []
		for i in range(20):
			dist = compute_distance(old_actor,new_actor,env,is_discrete)
			distance_list.append(dist)
		distance_list =	np.array(distance_list)
		print(f'distance_list ========== {distance_list}')
		mean = np.mean(distance_list)
		std_dev = np.std(distance_list)
		print(f'mean dis ========== {mean} std div ======= {std_dev}')
	#=============================== Compute Distance between two policies Code End  ==========================#
	


