"Actor-Critic model for PPO"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorNet(nn.Module):

    def __init__(self,state_size,action_size):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(state_size, 100)
        self.mu_head = nn.Linear(100, action_size)
        self.sigma_head = nn.Linear(100, action_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticNet(nn.Module):

    def __init__(self,state_size):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(state_size, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value


