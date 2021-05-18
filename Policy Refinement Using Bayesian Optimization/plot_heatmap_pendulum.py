from matplotlib import pyplot as plt
import numpy as np
import torch
from network import FeedForwardActorNN, FeedForwardCriticNN
import gym
import pickle

# import parser
# parser.add_argument('--algo', type=str, default='ppo')


def ppo_heatmap_observations(env,critic_name,actor_name):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    x_pxl, y_pxl = 300, 400

    state = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])
    cnet = FeedForwardCriticNN(obs_dim, 1)
    cnet.load_state_dict(torch.load(critic_name))
    value_map = cnet(state).view(y_pxl, x_pxl).detach().numpy()

    anet = FeedForwardActorNN(obs_dim, act_dim)
    anet.load_state_dict(torch.load(actor_name))
    action_map = anet(state).view(y_pxl, x_pxl).detach().numpy()

    fig = plt.figure()
    fig.suptitle('PPO Pendulum Updated Policy')
    ax = fig.add_subplot(121)
    im = ax.imshow(value_map, cmap=plt.cm.spring, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Value Map')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])

    ax = fig.add_subplot(122)
    im = ax.imshow(action_map, cmap=plt.cm.winter, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Action Map ')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])
    plt.tight_layout()
    plt.savefig('img/ppo_heatmap_update_policy.png')
    plt.show()


def ppo_heatmap_failure(env,critic_name,actor_name):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cnet = FeedForwardCriticNN(obs_dim, 1)
    cnet.load_state_dict(torch.load(critic_name))

    anet = FeedForwardActorNN(obs_dim, act_dim)
    anet.load_state_dict(torch.load(actor_name))

    with open('failure_trajectory_pendulum.data', 'rb') as filehandle1:
        # read env_state
        failure_observations = pickle.load(filehandle1)

    obs = failure_observations[0][0]
    obs = obs[:-1]
    print(len(obs))
    state = torch.Tensor([np.array(x) for x in obs])
    value_map = cnet(state).view(20, 10).detach().numpy()

    action_map = anet(state).view(20, 10).detach().numpy()

    fig = plt.figure()
    fig.suptitle('Pendulum Failure Updated')
    ax = fig.add_subplot(121)
    im = ax.imshow(value_map, cmap=plt.cm.spring, interpolation='bicubic')
    plt.colorbar(im)
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, 10, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, 20, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])

    ax = fig.add_subplot(122)
    im = ax.imshow(action_map, cmap=plt.cm.winter, interpolation='bicubic')
    plt.colorbar(im)
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, 10, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, 20, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])

    plt.savefig('img/ppo_heatmap_failure_update_policy.png')
    plt.show()


env = gym.make('Pendulum-v0')
seed = 0
env.seed(seed)
#ppo_heatmap_observations(env,'ppo_criticPendulum-v0.pth','ppo_actorPendulumupdated.pth')
ppo_heatmap_failure(env,'ppo_criticPendulum-v0.pth','ppo_actorPendulumupdated.pth')