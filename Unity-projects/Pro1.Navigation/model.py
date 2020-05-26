import torch
import torch.nn as nn
import torch.nn.functional as F

Hidden_1_Unities=64
Hidden_2_Unities=64

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
        self.fc1=nn.Linear(state_size,Hidden_1_Unities)
        self.fc2=nn.Linear(Hidden_1_Unities,Hidden_2_Unities)
        self.fc3=nn.Linear(Hidden_2_Unities,action_size)
        # self.fc4=nn.Linear(Hidden_3_Unities,action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        out=self.fc1(state)
        out=F.relu(out)
        out=self.fc2(out)
        out=F.relu(out)
        q_a=self.fc3(out)
#         out=F.relu(out)
#         q_a=self.fc4(out)

        return q_a
