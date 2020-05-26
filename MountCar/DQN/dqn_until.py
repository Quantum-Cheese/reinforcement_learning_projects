import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.fc1=nn.Linear(state_size,128)
        self.fc2=nn.Linear(128,64)
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

        states=torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions=torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards=torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states=torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones=torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states,actions,rewards,next_states,dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)






