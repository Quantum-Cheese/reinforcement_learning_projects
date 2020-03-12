import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1=nn.Linear(state_size,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out=self.fc1(state)
        out=F.relu(out)
        out=self.fc2(out)
        out=F.relu(out)
        q_a=self.fc3(out)

        return q_a


class Replay_Buffer():
    def __init__(self,buffer_size,batch_size):
        self.memory=deque(maxlen=buffer_size)
        self.batch_size=batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self,state,action,reward,next_state,done):
        exp=self.experience(state,action,reward,next_state,done)
        self.memory.append(exp)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)




