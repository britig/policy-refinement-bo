"""
	Code for policy updation once the failure trajectories are obtained via Bayesian Optimization
	Project : Policy correction using Bayesian Optimization
"""


import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from network import FeedForwardActorNN, FeedForwardCriticNN
import pickle
import sys


#Integrating tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

timesteps_per_batch = 2048                 # Number of timesteps to run per batch
max_timesteps_per_episode = 200           # Max number of timesteps per episode
n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
cov_var = 0
cov_mat = 0
# Initialize the covariance matrix used to query the actor for actions
cov_var = None
cov_mat = None

is_discrete_action = False

logger = {
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'actor_network' : 0		# Actor network
		}

def update_policy(policy_old, critic_old, policy_new, critic_new, env, fail_file):
	"""
		The main learning function
	"""
	actor_optim = Adam(policy_old.parameters(), lr=0.005)
	critic_optim = Adam(critic_old.parameters(), lr=0.005)
	env_name = env.unwrapped.spec.id
	t_so_far = 0 # Timesteps simulated so far
	i_so_far = 0 # Iterations ran so far
	
	#Number for steps for gradient correction
	num_of_iterations = 10
	if(env.unwrapped.spec.id == 'CartPole-v0'):
		num_of_iterations = 100
	global logger
	while i_so_far < num_of_iterations:
		# Commence failure trajectory collection based on observation produced by BO
		batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = get_trajectory(policy_new,env,fail_file)
		# Calculate how many timesteps we collected this batch
		t_so_far += np.sum(batch_lens)

		# Increment the number of iterations
		i_so_far += 1

		# Logging timesteps so far and iterations so far
		logger['t_so_far'] = t_so_far
		logger['i_so_far'] = i_so_far

		# Calculate advantage at k-th iteration
		V, _ = evaluate(critic_new, policy_new, batch_obs, batch_acts)
		A_k = batch_rtgs - V.detach()

		# Same as main PPO code
		A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)



		#print(f'A_k======================={A_k}')

		if(len(batch_lens)!=0):
			# This is the loop where we update our network for some n epochs
			for _ in range(n_updates_per_iteration):
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = evaluate(critic_new, policy_old, batch_obs, batch_acts)
				#print(f'curr_log_probs======================={curr_log_probs}')
				#print(f'batch_log_probs======================={batch_log_probs}')
				ratios = torch.exp(curr_log_probs - batch_log_probs)
				surr1 = ratios
				#print(f'surr1======================={surr1}')
				surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2)
				#print(f'surr2======================={surr2}')
				actor_loss = (-torch.min(surr1, surr2)).mean()
				#print(f'actor_loss======================={actor_loss}')
				critic_loss = nn.MSELoss()(V, batch_rtgs)


				# Calculate gradients and perform backward propagation for actor network
				#Not updating the critic as it has been trained as a baseline
				actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				critic_optim.zero_grad()
				critic_loss.backward()
				critic_optim.step()

				logger['actor_losses'].append(actor_loss.detach())
				logger['actor_network'] = policy_old

			_log_summary()

		torch.save(policy_old.state_dict(), './ppo_actor_updated'+env_name+'.pth')


def get_trajectory(policy_new,env,fail_file):
	global max_timesteps_per_episode
	env_name = env.unwrapped.spec.id
	"""
		This is where we collect the failure trajectories
	"""
	# Batch data
	batch_obs = []
	batch_acts = []
	batch_log_probs = []
	batch_rews = []
	batch_rtgs = []
	batch_lens = []

	ep_rews = []
	t = 0

	#print(f'Environment variables after  modification {env.env.lander.position}')
	render = False
	global logger
	obs_count = 0
	if(env_name == 'BipedalWalker-v3'):
		max_timesteps_per_episode=1200

	while t < timesteps_per_batch:
		ep_rews = [] # rewards collected per episode
		env.reset()
		with open(fail_file, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		len_obs = len(failure_observations)
		index = obs_count%len_obs
		obs = failure_observations[index][0][0]
		if env_name == 'Pendulum-v0':
			env.env.state = obs[0:2]
		else:
			env.env.state = obs

		for ep_t in range(max_timesteps_per_episode):
			# Render environment if specified, off by default
			if render:
				env.render()
			
			t += 1 # Increment timesteps ran this batch so far
			# Track observations in this batch
			batch_obs.append(obs)

			# Calculate action and make a step in the env. 
			# Note that rew is short for reward.
			if is_discrete_action:
				action, log_prob = get_action_discrete(policy_new,obs)
			else:
				action, log_prob = get_action(policy_new,obs)
			obs, rew, done, _ = env.step(action)

			# Track recent reward, action, and action log probability
			ep_rews.append(rew)
			batch_acts.append(action)
			batch_log_probs.append(log_prob)

			# If the environment tells us the episode is terminated, break
			if done:
				break

		# Track episodic lengths and rewards
		batch_lens.append(ep_t + 1)
		batch_rews.append(ep_rews)

	avg_rtgs = [np.sum(ep_rews) for ep_rews in batch_rews]
	batch_rtgs = compute_rtgs(batch_rews).tolist()
	index_to_remove = []
	number_of_negative_traces = 0
	for i in range(len(avg_rtgs)):
		# remove the negative traces from collected episode as we only want to update 
		#print(f'Reward==={i}====={avg_rtgs[i]}') 
		if(avg_rtgs[i] < -300):
			number_of_negative_traces += 1
			#print(f'Number of negative traces {number_of_negative_traces} ====== Avg rtgs for {i} ===== {avg_rtgs[i]}')
			index_to_remove.append(i)

	#print(f'Number of negative traces {number_of_negative_traces}')

	for num in reversed(index_to_remove):
		batch_obs.pop(num)
		batch_acts.pop(num)
		batch_log_probs.pop(num)
		batch_rtgs.pop(num)
		batch_lens.pop(num)
		batch_rews.pop(num)

	# Reshape data as tensors in the shape specified in function description, before returning
	batch_obs = torch.tensor(batch_obs, dtype=torch.float)
	batch_acts = torch.tensor(batch_acts, dtype=torch.float)
	batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
	batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
	number_of_negative_traces = 0
	avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])


	#print(f'batch_lens =========={batch_lens} batch_rews ========{avg_ep_rews}')

	logger['batch_rews'] = batch_rews
	logger['batch_lens'] = batch_lens

	return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

def _log_summary():
	global logger
	t_so_far = logger['t_so_far']
	i_so_far = logger['i_so_far']
	avg_ep_lens = np.mean(logger['batch_lens'])
	avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in logger['batch_rews']])
	avg_actor_loss = np.mean([losses.float().mean() for losses in logger['actor_losses']])
	actor_model = logger['actor_network']

	# Round decimal places for more aesthetic logging messages
	avg_ep_lens = str(round(avg_ep_lens, 2))
	avg_ep_rews = str(round(avg_ep_rews, 2))
	avg_actor_loss = str(round(avg_actor_loss, 5))

	writer.add_scalar("Average Episodic Return", int(float(avg_ep_rews)), t_so_far)
	writer.add_scalar("Average actor Loss", int(float(avg_actor_loss)), t_so_far)

	for name, param in actor_model.named_parameters():
		if 'weight' in name:
			writer.add_histogram(name, param.detach().numpy(), t_so_far)

	# Print logging statements
	print(flush=True)
	print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
	print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
	print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
	print(f"Average Loss: {avg_actor_loss}", flush=True)
	print(f"Timesteps So Far: {t_so_far}", flush=True)
	print(f"------------------------------------------------------", flush=True)
	print(flush=True)

	# Reset batch-specific logging data
	logger['batch_lens'] = []
	logger['batch_rews'] = []
	logger['actor_losses'] = []


def compute_rtgs(batch_rews):
	"""
		Compute the Reward-To-Go of each timestep in a batch given the rewards.

		Parameters:
			batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

		Return:
			batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
	"""
	# The rewards-to-go (rtg) per episode per batch to return.
	# The shape will be (num timesteps per episode)
	batch_rtgs = []

	# Iterate through each episode
	for ep_rews in reversed(batch_rews):

		discounted_reward = 0 # The discounted reward so far

		# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
		# discounted return (think about why it would be harder starting from the beginning)
		for rew in reversed(ep_rews):
			discounted_reward = rew + discounted_reward * 0.95
			batch_rtgs.insert(0, discounted_reward)

	# Convert the rewards-to-go into a tensor
	batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

	return batch_rtgs


#Sample the action distributions from the sub policy for discrete actions

def get_action_discrete(policy_new, obs):
	#print(f'obs ================== {obs}')
	mean = policy_new(obs)
	#print(f'mean ================== {mean}')

	dist = Categorical(mean)

	#print(f'dist ================== {dist}')

	action = dist.sample()

	log_prob = dist.log_prob(action)
	#print(f'action ====== {action} ========= {log_prob}')

	return action.detach().numpy().item(), log_prob.detach().item()


#Sample the action distributions from the sub policy
def get_action(policy_new, obs):
	global cov_mat
	mean = policy_new(obs)
	dist = MultivariateNormal(mean, cov_mat)


	# Sample an action from the distribution
	action = dist.sample()

	# Calculate the log probability for that action
	log_prob = dist.log_prob(action)

	# Return the sampled action and the log probability of that action in our distribution
	return action.detach().numpy(), log_prob.detach()


def evaluate(critic, policy, batch_obs, batch_acts):
	"""
		Estimate the values of each observation, and the log probs of
		each action in the most recent batch with the most recent
		iteration of the actor network. Should be called from learn.

		Parameters:
			batch_obs - the observations from the most recently collected batch as a tensor.
						Shape: (number of timesteps in batch, dimension of observation)
			batch_acts - the actions from the most recently collected batch as a tensor.
						Shape: (number of timesteps in batch, dimension of action)

		Return:
			V - the predicted values of batch_obs
			log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
	"""
	# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
	global cov_mat,is_discrete_action
	V = critic(batch_obs).squeeze()

	# Calculate the log probabilities of batch actions using most recent actor network.
	# This segment of code is similar to that in get_action()
	mean = policy(batch_obs)
	#dist = MultivariateNormal(mean, cov_mat)
	if is_discrete_action:
		dist = Categorical(mean)
	else:
		dist = MultivariateNormal(mean, cov_mat)
	log_probs = dist.log_prob(batch_acts)

	# Return the value vector V of each observation in the batch
	# and log probabilities log_probs of each action in the batch
	return V, log_probs



def correct_policy(env, actor_model_old, critic_model_old,actor_model_new, critic_model_new,is_discrete,file_name):
	"""
		Updates the policy model to correct the failure trajectories.
		Parameters:
			env - the environment to test the policy on
			actor_model - the actor neural network model
			critic_model - critic neural network model
			observation - currently working with one observation to extract the failure trajectory
		Return:
			None
	"""
	print(f"Correcting {actor_model_old} for observation ", flush=True)
	global cov_mat,cov_var,is_discrete_action

	#Setting the discrete flag to denote action space
	is_discrete_action = is_discrete

	# If the actor model is not specified, then exit
	if actor_model_old == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	if is_discrete:
		act_dim = env.action_space.n
	else:
		act_dim = env.action_space.shape[0]


	# Build our policy and critic the same way we build our actor model in PPO
	policy_old = FeedForwardActorNN(obs_dim, act_dim,is_discrete_action)
	critic_old = FeedForwardCriticNN(obs_dim, 1)
	policy_new = FeedForwardActorNN(obs_dim, act_dim,is_discrete_action)
	critic_new = FeedForwardCriticNN(obs_dim, 1)

	# Load in the actor model saved by the PPO algorithm
	policy_old.load_state_dict(torch.load(actor_model_old))
	critic_old.load_state_dict(torch.load(critic_model_old))
	policy_new.load_state_dict(torch.load(actor_model_new))
	critic_new.load_state_dict(torch.load(critic_model_new))

	cov_var = torch.full(size=(act_dim,), fill_value=0.5)
	cov_mat = torch.diag(cov_var)

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	update_policy(policy_old=policy_old, critic_old=critic_old, policy_new=policy_new, critic_new=critic_new, env=env, fail_file=file_name)
