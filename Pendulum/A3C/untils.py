import torch
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)

        return value


class PolicyNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.mu_head = nn.Linear(256, output_dim)
        self.sigma_head = nn.Linear(256, output_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))

        return mu,sigma


