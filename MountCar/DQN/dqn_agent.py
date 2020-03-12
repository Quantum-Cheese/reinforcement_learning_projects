
import torch
import torch.optim as optim
from MountCar.DQN.dqn_until import QNetwork,Replay_Buffer

LR=0.001
GAMMA=1.0


class DQN():
    def __init__(self,state_size,action_size,seed):
        self.q_net=QNetwork(state_size,action_size,seed)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.memory=Replay_Buffer()

    def act(self,state,epsilon):
        pass

    def learn(self):
        pass

    def step(self,state,action,reward,next_state,done):
        self.memory.add(state,action,reward,next_state,done)





