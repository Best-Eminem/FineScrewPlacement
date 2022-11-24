import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
###############################################################################
# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS 256
BATCH_SIZE = 48#48
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (45, 45, 45)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4
# the frequency of updating the target network
UPDATE_FREQ = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9 #0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 1e5#5#6   # to debug on bedale use 1e4
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = MEMORY_SIZE // 20 #5e4
# each epoch is 100k played frames
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 2
# the number of episodes to run during evaluation
EVAL_EPISODE = 50

class CnnQNet(nn.Module):
    def __init__(self,in_channel, out_channel, agents=2, **kwargs):
        super(CnnQNet, self).__init__()
        self.agents = agents
        self.conv_net = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,out_channels=32, kernel_size=[5, 5, 5], stride=[1, 1, 1], padding='same'),                             
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32,out_channels=32, kernel_size=[5, 5, 5], stride=[1, 1, 1], padding='same'),                             
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32,out_channels=64, kernel_size=[4, 4, 4], stride=[1, 1, 1], padding='same'),                             
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64,out_channels=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding='same')
        )
        self.linear_net_1 = nn.Sequential(
            nn.Linear(102400, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.linear_net_2 = nn.Sequential(
            nn.Linear(66, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel),
            nn.Sigmoid(),
        )
    def forward(self, state, action):
        state = self.conv_net(state)
        _state = state.view(-1,102400)
        _state = self.linear_net_1(_state)
        _state_action = torch.cat((_state, action), dim=1)
        q_value = self.linear_net_2(_state_action)
        return q_value