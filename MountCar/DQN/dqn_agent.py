import random
import numpy as np
from collections import namedtuple, deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from MountCar.DQN.dqn_until import QNetwork,Replay_Buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR=0.001
GAMMA=0.99
TAU = 1e-3
BUFFER_SIZE=int(1e6)
BATCH_SIZE=64


class DQN():
    def __init__(self,state_size,action_size,seed):
        self.state_size=state_size
        self.action_size=action_size

        self.local_net=QNetwork(state_size,action_size,seed).to(device)
        self.target_net=QNetwork(state_size,action_size,seed).to(device)

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=LR)
        self.memory=Replay_Buffer(BUFFER_SIZE,BATCH_SIZE)

    def act(self,state,epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_net.eval()
        with torch.no_grad():
            qa_values = self.local_net(state).cpu().data.numpy()
        self.local_net.train()

        """ epsilon 贪婪策略选取 """
        probs = np.array([epsilon / self.action_size for _ in range(self.action_size)])
        probs[np.argmax(qa_values)] = 1 - epsilon + (epsilon / self.action_size)
        action=np.random.choice(np.arange(self.action_size), p=probs)

        return  action

    def learn(self,experiences):
        states, actions, rewards, next_states, dones = experiences
        # calculate Q targets:Q_target(s,a') (estimated from Q(next_state) using target network)
        max_actions=self.local_net(next_states).detach().argmax(1).unsqueeze(1)  # self.local_net(next_states).detach().argmax(1).unsqueeze(1)
        q_target_next=self.target_net(next_states).gather(1,max_actions)
        q_targets=rewards + GAMMA*q_target_next*(1-dones)  # [batch_size,1]

        # get Q expected: Q_local(s,a) (using local network)
        q_expects=self.local_net(states).gather(1,actions)  # [batch_size,1]

        # calculate MSE loss function
        loss = F.mse_loss(q_expects, q_targets)  # 对多个样本取平均

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_net,self.target_net,TAU)

        return loss.detach().numpy()


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def step(self,state,action,reward,next_state,done):
        self.memory.add(state,action,reward,next_state,done)

        if self.memory.__len__()>BATCH_SIZE:
            exps=self.memory.sample()
            loss=self.learn(exps)
            return loss






