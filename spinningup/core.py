import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, a_Left=None, a_Right=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.      
        pi_Left, pi_Right = self._distribution(obs)
        logp_a_Left = None
        logp_a_Right = None
        if a_Left is not None and a_Right is not None:
            logp_a_Left = self._log_prob_from_distribution(pi_Left, a_Left)
            logp_a_Right = self._log_prob_from_distribution(pi_Right, a_Right)
        return pi_Left, pi_Right, logp_a_Left, logp_a_Right


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std_left = -0.5 * torch.ones(act_dim)
        log_std_right = -0.5 * torch.ones(act_dim)
        self.log_std_left = torch.nn.Parameter(log_std_left)
        self.log_std_right = torch.nn.Parameter(log_std_right)
        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.conv_net = nn.Sequential(
            layer_init(nn.Conv3d(1, 32, 8, stride=4)),
            nn.ReLU(True),
            layer_init(nn.Conv3d(32, 64, 4, stride=2)),
            nn.ReLU(True),
            layer_init(nn.Conv3d(64, 64, 3, stride=1)),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.linearNetLeft = nn.Sequential(
            layer_init(nn.Linear(64 * 16 * 6 * 4, 512)),
            nn.ReLU(True),
            layer_init(nn.Linear(512, act_dim), std=0.01),
            # nn.Tanh(),
        )
        self.linearNetRight = nn.Sequential(
            layer_init(nn.Linear(64 * 16 * 6 * 4, 512)),
            nn.ReLU(True),
            layer_init(nn.Linear(512, act_dim), std=0.01),
            # nn.Tanh(),
        )

    def _distribution(self, obs):
        _flatten = self.conv_net(obs / 2.0)
        mu_left = self.linearNetLeft(_flatten)
        mu_right = self.linearNetRight(_flatten)
        std_left = torch.exp(self.log_std_left)
        std_right = torch.exp(self.log_std_right)
        return Normal(mu_left, std_left), Normal(mu_right, std_right)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = nn.Sequential(
            layer_init(nn.Conv3d(1, 32, 8, stride=4)),
            nn.ReLU(True),
            layer_init(nn.Conv3d(32, 64, 4, stride=2)),
            nn.ReLU(True),
            layer_init(nn.Conv3d(64, 64, 3, stride=1)),
            nn.ReLU(True),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 16 * 6 * 4, 512)),
            nn.ReLU(True),
            layer_init(nn.Linear(512, 1), std=1),
        )

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs / 2.0), -1) # Critical to ensure v has right shape.


class MyMLPActorCritic(nn.Module):


    def __init__(self, observation_shape, action_shape, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()


        # policy builder depends on action space
        # if isinstance(action_space, Box):
        self.pi = MLPGaussianActor(observation_shape, action_shape[0], hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(observation_shape, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi_Left, pi_Right = self.pi._distribution(obs)
            a_Left = pi_Left.sample()
            a_Right = pi_Right.sample()
            # a = torch.clamp(a, min=-1.0, max=1.0)
            logp_a_Left = self.pi._log_prob_from_distribution(pi_Left, a_Left)
            logp_a_Right = self.pi._log_prob_from_distribution(pi_Right, a_Right)
            v = self.v(obs)
        return a_Left.squeeze_(0).cpu().numpy() if len(a_Left.shape) == 2 else a_Left, \
                a_Right.squeeze_(0).cpu().numpy() if len(a_Right.shape) == 2 else a_Right, \
                v.cpu().numpy(), \
                logp_a_Left.cpu().numpy(), logp_a_Right.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0], self.step(obs)[1]