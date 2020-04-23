
from collections import namedtuple,deque
import torch
import torch.nn as nn
import torch.nn.functional as F



class ActorNet(nn.Module):

    def __init__(self,state_size,action_size):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128,128)
        self.mu_head = nn.Linear(128, action_size)
        self.sigma_head = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticNet(nn.Module):

    def __init__(self,state_size):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128,128)
        self.v_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.v_head(x)
        return state_value


class Memory():
    def __init__(self):
        self.trajectory=[]
        self.Transition = namedtuple('Transition', ['state', 'action', 'prob', 'reward'])

    def add(self,state,action,prob,reward):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.trajectory.append(self.Transition(state,action,prob,reward))

    def clean_buffer(self):
        del self.trajectory[:]

    def __len__(self):
        return len(self.trajectory)

