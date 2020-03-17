import random
import numpy as np
import gym
from collections import namedtuple
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from CartPole.Policy_Gradient.model import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=1.0
LR=0.001
BATCH_SIZE=5

Transition = namedtuple('Transition', ['state', 'action',  'prob', 'reward'])

class PPO_v1():

    def __init__(self, state_size, action_size):
        self.policy=Policy(state_size,action_size).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(), lr=LR)
        self.trajectory=[]

    def update_policy(self,exps):
        """
        update policy for every sampled transition groups
        called by learn() multiple times for one episode
        """
        states,actions,old_probs,f_Rewrds=exps
        # get action probs from new policy
        new_probs=self.policy(states).gather(1,actions)



    def learn(self):
        states=torch.cat([t.state for t in self.trajectory])
        actions=torch.tensor([t.action for t in self.trajectory],dtype=torch.long).view(-1,1)
        old_probs=torch.tensor([t.prob for t in self.trajectory],dtype=torch.float).view(-1,1)

        # -- calculate discount future rewards for every time step
        rewards = [t.reward for t in self.trajectory]
        fur_Rewards = []
        for i in range(len(rewards)):
            discount = [GAMMA ** i for i in range(len(rewards) - i)]
            f_rewards = rewards[i:]
            fur_Rewards.append(sum(d * f for d, f in zip(discount, f_rewards)))
        fur_Rewards=torch.tensor(fur_Rewards,dtype=torch.float).view(-1,1)

        for i in range(1):
            # -- disorganize transitions in the trajectory into sub groups
            for index_set in BatchSampler(SubsetRandomSampler(range(len(self.trajectory))), BATCH_SIZE, False):
                exps=(states[index_set],actions[index_set],old_probs[index_set],fur_Rewards[index_set])
                # -- update policy network for every sub groups
                self.update_policy(exps)




    def train(self,env):
        state = env.reset()

        while True:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 升维 1d->2d
            result_dic = self.policy.act(state)
            next_state, reward, done, _ = env.step(result_dic['action'])

            # --store transition in this current trajectory
            self.trajectory.append(Transition(state,result_dic['action'],result_dic['prob'],reward))
            state=next_state

            # --agent learn after finish current episode, and if there is enough transitions
            if done:
                if BATCH_SIZE <= len(self.trajectory):
                    self.learn()
                break


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    agent=PPO_v1(state_size=4,action_size=2)

    for i_episode in range(1):
        agent.train(env)



