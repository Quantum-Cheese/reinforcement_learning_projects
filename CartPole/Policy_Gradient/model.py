import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Policy(nn.Module):

    def __init__(self,state_size,action_size):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, action_size)
        # action_size=1: Bernoulli
        # action_size=2: Categorical

    def forward(self, state):
        """
        Build a network that maps state -> action probs.
        """

        out=self.fc1(state)
        out=F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        probs=torch.sigmoid(out)

        return probs

