import numpy as np

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from ddpg_untils import Actor,Critic
from ddpg_untils import OUNoise

TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay


class DDPGAgent:
    def __init__(self, state_size, action_size, seed,shared_memory,device):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device=device

        # --- 定义Actor策略网络
        self.actor_local=Actor(state_size,action_size,seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # --- 定义Critic值函数网络
        self.critic_local = Critic(state_size, action_size, seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # why weight decay here?

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = shared_memory

    def reset(self):
        self.noise.reset()

    def act(self, state, noise=True):
        """
        Called by MultiAgents()
        :param state: observation for single agent
        :param noise: True/False
        :return: action for single agent （numpy array）
        """
        state = torch.from_numpy(state).float().to(self.device)  # 转换为 tensor (torch.float32)

        #  s(t) ---> a(t)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # adding noise
        if noise:
            action += self.noise.sample()

        return np.clip(action,-1,1)

    def learn(self,experiences,gamma):
        """
        Called by MultiAgents()
        :param experiences: 一个 batch 的经验元组 (Tuple[torch.Tensor])
        :param gamma: 折扣率
        """
        (states, actions, rewards, next_states, dones)=experiences

        # ------------ 更新 local Critic (batch update)---------------- ##

        # ----- 根据 s(t+1) 从target actor 网络中得到预估的 a(t+1) * 非实际动作
        next_actions = self.actor_target(next_states)
        # ----- s(t+1)和a(t+1)输入 target critic 网络得到下个 time step 的 Q value 估计值
        Q_targets_next = self.critic_target(next_states,next_actions)
        # --- 计算 Q 目标：target critic网络在当前 time step 的 Q value 的估计目标值（Bellman 方程）
        Q_targets = rewards+(gamma*Q_targets_next*(1-dones))
        # --- 从 local critic 中得到 time step 下 Q value 的 local估计值
        Q_expecteds = self.critic_local(states,actions)  # *这里用的是实际 action

        # - 计算 critic的 MSE 损失函数
        critic_loss=F.mse_loss(Q_expecteds,Q_targets)

        # 梯度下降 minimize Loss，更新网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------- 更新 local Actor (batch update) -------------- ##

        # ---- 根据 s(t) 从 local actor 得到预估的 a(t) * 非实际动作
        actions_pred=self.actor_local(states)

        # - 计算 actor 的损失函数（ 取目标函数的 negative mean ）
        actor_loss=-self.critic_local(states,actions_pred).mean()

        # 梯度下降 minimize Loss，更新网络
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

