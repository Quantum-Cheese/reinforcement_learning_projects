import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512       # minibatch size
GAMMA = 0.8            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0   # L2 weight decay
LEARN_EVERY=8          # agent only update every _ episode
N= 6                  # every update, call learn() for N times

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=256,fc2_units=256):
        """
        single layer MLP network
        ======
            Input dim: state_size
            Output dim: action_size
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3=nn.Linear(fc2_units,action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model.
    """

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256,fc3_units=128):
        """
        ======
            Input dim: state_size
            Output dim: 1
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


class DDPGAgent:
    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # --- 定义Actor策略网络
        self.actor_local=Actor(state_size,action_size,seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # --- 定义Critic值函数网络
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def reset(self):
        self.noise.reset()

    def act(self,state,noise=True):
        """
        与外部 env 直接交互
        :param state: 外部env传入的 numpy array
        :param noise: True/False
        :return: action （numpy array）
        """
        state=torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            org_action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if noise:
            action=org_action+self.noise.sample()
            return action
        else:
            return org_action

    def step(self,i_episode,state, action, reward, next_state, done):
        """
        1.与外部 env 直接交互，存储经验组；
        2. 调用learn()训练智能体
        :param state,action,next_state,reward : 每个 time step 从外部 env 传入的经验组

        """
        self.memory.add(state,action,next_state,reward,done)

        if self.memory.__len__() > BATCH_SIZE:
            if i_episode%LEARN_EVERY==0:
                for n in range(N):
                    experiences = self.memory.sample()
                    self.learn(experiences,GAMMA)

    def learn(self,experiences,gamma):
        """
        不与外部 env 直接交互，由 step() 调用
        ** 训练agent的核心部分 **
        :param experiences: 一个 batch 的经验元组 (Tuple[torch.Tensor])
        :param gamma: 折扣率
        """
        (states, actions, rewards, next_states, dones)=experiences

        # ------------ 更新 local Critic (batch update)---------------- ##
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states,next_actions)
        Q_targets = rewards+(gamma*Q_targets_next*(1-dones))
        Q_expecteds = self.critic_local(states,actions)  # *这里用的是实际 action

        # - 计算 critic的 MSE 损失函数
        critic_loss=F.mse_loss(Q_expecteds,Q_targets)

        # 梯度下降 minimize Loss，更新网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------- 更新 local Actor (batch update) -------------- ##
        actions_pred=self.actor_local(states)
        actor_loss=-self.critic_local(states,actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------- 更新 target networks-------------##
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    def __init__(self,action_size,buffer_size,batch_size,seed):
        """Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        # 存储经验元组的 memory （队列，有限大小）
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        # 每个经验元组 （namedtuple）
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self,state,action,next_state,reward,done):
        exp=self.experience(state,action,reward,next_state,done)
        self.memory.append(exp)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)





