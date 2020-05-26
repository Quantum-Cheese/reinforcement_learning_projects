import random
import numpy as np
import gym
from collections import namedtuple
from collections import deque
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from CartPole.Policy_Gradient.model import Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=0.99
LR=0.001
BATCH_SIZE=32
CLIP=0.2
UPDATE_TIME=10
max_grad_norm=0.5

Transition = namedtuple('Transition', ['state', 'action',  'prob', 'reward'])


class PPO_v1():

    def __init__(self, state_size, action_size):
        self.policy=Actor(state_size,action_size).to(device)
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

        # calculate clipped surrogate function
        ratios=new_probs/old_probs
        surr1 = ratios * f_Rewrds
        surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * f_Rewrds

        # calculate policy loss for current update time
        policy_loss=-torch.min(surr1,surr2).mean()

        # update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.optimizer.step()

        # self.traintime_counter+=1

    def learn(self):
        """
        agent learn after finishing every episode.
        learn from experiences of this trajectory
        :return:
        """
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

        for i in range(UPDATE_TIME):
            # -- repeat the flowing update loop for several times
            # disorganize transitions in the trajectory into sub groups
            for index_set in BatchSampler(SubsetRandomSampler(range(len(self.trajectory))), BATCH_SIZE, False):
                exps=(states[index_set],actions[index_set],old_probs[index_set],fur_Rewards[index_set])
                # -- update policy network for every sub groups
                self.update_policy(exps)

        del self.trajectory[:]  # clear trajectory


    def train(self,env):
        state = env.reset()
        total_reward=0
        while True:
            # self.timesetp_counter+=1
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 升维 1d->2d
            result_dic = self.policy.act(state)
            next_state, reward, done, _ = env.step(result_dic['action'])

            # --store transition in this current trajectory
            self.trajectory.append(Transition(state,result_dic['action'],result_dic['prob'],reward))
            state=next_state
            total_reward+=reward
            if done:
                break
        # --agent learn after finish current episode, and if there is enough transitions
        if BATCH_SIZE <= len(self.trajectory):
            self.learn()

        return total_reward


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    agent=PPO_v1(state_size=4,action_size=2)
    n_episode=2000

    scores_deque=deque(maxlen=100)
    scores=[]
    for i_episode in range(1,n_episode+1):
        Reward=agent.train(env)
        # print("total time_step for one episode:{}\t,total trainning time for one episode:{}".format(agent.timesetp_counter,agent.traintime_counter))

        scores_deque.append(Reward)
        scores.append(Reward)
        if i_episode % 100 == 0:
            print('Episode {}\t Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}\n----------\n'.format(i_episode,
                                                                                       np.mean(scores_deque)))
            torch.save(agent.policy.state_dict(),'models/PPO_model-1.pth')
            break



