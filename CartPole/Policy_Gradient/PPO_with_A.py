import random
import numpy as np
import gym
from collections import namedtuple
from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from CartPole.Policy_Gradient.model import Actor,Critic
from CartPole.Policy_Gradient.PPO_with_R import PPO_v1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=0.99
LR_a=0.001
LR_c=0.003
BATCH_SIZE=32
CLIP=0.2
UPDATE_TIME=10
max_grad_norm=0.5

Transition = namedtuple('Transition', ['state', 'action',  'prob', 'reward'])


class PPO_V2(PPO_v1):
    def __init__(self,state_size, action_size):
        self.policy = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_c)

        self.trajectory = []

    def update_policy(self,exps):
        states, actions, old_probs, f_Rewrds = exps

        # -- update policy(actor) network -- #
        # get action probs from new policy and calculate the ratio
        new_probs = self.policy(states).gather(1, actions)
        ratios = new_probs / old_probs

        # calculate advance from critic network
        V= self.critic(states)
        advantage=(f_Rewrds-V).detach()

        # calculate clipped surrogate function
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # update parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.policy_optimizer.step()

        # -- update value(critic) network -- #
        value_loss = F.mse_loss(f_Rewrds, V)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    agent=PPO_V2(state_size=4,action_size=2)
    n_episode=2000

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1,n_episode+1):
        Reward=agent.train(env)

        scores_deque.append(Reward)
        scores.append(Reward)
        if i_episode % 100 == 0:
            print('Episode {}\t Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        # if np.mean(scores_deque) >= 195.0:
        #     print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}\n----------\n'.format(i_episode,
        #                                                                                              np.mean(
        #                                                                                                  scores_deque)))
        #     torch.save(agent.policy.state_dict(), 'models/PPOv2_model-1.pth')
            # break

