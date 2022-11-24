import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
class CnnPNet(nn.Module):
    def __init__(self,in_channel, out_channel, agents=2, **kwargs):
        super(CnnPNet, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv3d(1, 32, 8, stride=4)),
            nn.Tanh(),
            layer_init(nn.Conv3d(32, 64, 4, stride=2)),
            nn.Tanh(),
            layer_init(nn.Conv3d(64, 64, 3, stride=1)),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(nn.Flatten(),
            layer_init(nn.Linear(64 * 16 * 6 * 4, 512)),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            layer_init(nn.Linear(512, 2), std=0.01),
                                    nn.BatchNorm1d(2),
                                    # nn.Sigmoid(),
        )
        self.rrr = nn.Sigmoid()
    def forward(self, x):
        hidden = self.network(x)
        action = self.actor(hidden)
        return 2*self.rrr(action)-1