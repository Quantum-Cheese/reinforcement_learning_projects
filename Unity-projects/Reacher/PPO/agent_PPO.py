import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from Reacher.PPO.PPO_utils import ActorNet,CriticNet,Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=0.99
LR_a=0.0005
LR_c=0.0005
BATCH_SIZE=32
CLIP=0.2
UPDATE_TIME=5
BETA=0.01
max_grad_norm=0.5


class PPO():

    def __init__(self, state_size, action_size):
        self.memory=Memory()

        self.policy = ActorNet(state_size, action_size).to(device)
        self.critic = CriticNet(state_size).to(device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_c)

    def act(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            (mu, sigma) = self.policy(state)  # 2d tensors
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.numpy()[0], action_log_prob.numpy()[0]

    def update_network(self,exps):
        states, actions, old_probs, f_Rewrds = exps

        # -- update policy(actor) network -- #
        # get action probs from new policy
        (mus, sigmas) = self.policy(states)
        dists = Normal(mus, sigmas)
        new_probs=dists.log_prob(actions)
        ratios = torch.exp(new_probs - old_probs)

        # calculate advance from critic network
        V = self.critic(states)
        advantage = (f_Rewrds - V).detach()

        # calculate clipped surrogate function
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage
        clipped_surrogate = -torch.min(surr1, surr2)

        # adding entropy term to the loss function
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))
        # final Loss
        # 加入 entropy 项
        # policy_loss=torch.mean(clipped_surrogate+BETA*entropy)
        policy_loss=torch.mean(clipped_surrogate)

        # update parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.policy_optimizer.step()

        # -- update value(critic) network -- #
        value_loss = F.mse_loss(f_Rewrds, V)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()

    def learn(self):
        """
        agent learn after finishing every episode.
        learn from experiences of this trajectory
        :return:
        """
        states = torch.tensor([t.state for t in self.memory.trajectory], dtype=torch.float)
        actions=torch.tensor([t.action for t in self.memory.trajectory],dtype=torch.float)
        old_probs=torch.tensor([t.prob for t in self.memory.trajectory],dtype=torch.float)

        # -- calculate discount future rewards for every time step
        rewards = [t.reward for t in self.memory.trajectory]
        fur_Rewards = []
        for i in range(len(rewards)):
            discount = [GAMMA ** i for i in range(len(rewards) - i)]
            f_rewards = rewards[i:]
            fur_Rewards.append(sum(d * f for d, f in zip(discount, f_rewards)))
        fur_Rewards=torch.tensor(fur_Rewards,dtype=torch.float).view(-1,1)

        for i in range(UPDATE_TIME):
            # -- repeat the flowing update loop for several times
            # disorganize transitions in the trajectory into sub groups
            for index_set in BatchSampler(SubsetRandomSampler(range(len(self.memory.trajectory))), BATCH_SIZE, False):
                exps=(states[index_set],actions[index_set],old_probs[index_set],fur_Rewards[index_set])
                # -- update policy network for every sub groups
                self.update_network(exps)

        self.memory.clean_buffer()  # clear trajectory

    def train(self,env,brain_name):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the Environment
        state = env_info.vector_observations[0]  # get the initial state
        total_reward=0
        while True:
            # -- agent act following the current policy
            action,log_prob = self.act(state)
            # -- interact with the env
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            # --store transition in this current trajectory
            self.memory.add(state,action,log_prob,reward)
            state=next_state
            total_reward+=reward
            if done:
                break
        if BATCH_SIZE <= len(self.memory):
            self.learn()

        return total_reward
