from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F


class linearQNet(nn.Module):
    def __init__(self, in_channels, out_channels, hide_channels=100, **kwargs):
        super(linearQNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hide_channels),
            nn.ReLU(),
            nn.Linear(hide_channels, out_channels))
    def forward(self, x):
        x = self.net(x)
        return x
