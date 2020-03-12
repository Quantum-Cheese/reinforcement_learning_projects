import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from CartPole.Policy_Gradient.model import Policy
from torch.autograd import Variable
import gym


class Agent_PG():

    def __init__(self, state_size, seed=0):
        self.state_size = state_size
        self.action_size = 1  # 输出维度为默认为1，表示选择left的概率


    def act(self,policy_net,state):
        """
        :param state -- 2d tensor
        :return:  action -- 1d tensor
                log_prob -- 1d tensor with grad
        """

        # probs=self.policy(Variable(state))
        probs=policy_net(state)
        c=Bernoulli(probs)
        # 根据动作概率选取一个action
        action=c.sample()
        # 所选 action 对应概率的对数值 log[p(a|s)]
        log_prob=-c.log_prob(action)

        # print("lgo_prob",log_prob,log_prob.size())
        return action,log_prob

    def calculate_loss(self,exps,gamma):
        # -- calculate discount rewards
        rewards=exps['rewards']
        discount = gamma ** np.arange(len(rewards))

        # --- old version
        # dis_rewards = np.asarray(rewards) * discount
        # rewards_future = np.cumsum(np.flipud(dis_rewards))
        # rewards_future = np.flipud(rewards_future)

        # -- Calculate future rewards sum (with discount)
        running_add = 0
        discount = list(reversed(discount))
        for i in reversed(range(len(rewards))):
            running_add = running_add * discount[i] + rewards[i]
            rewards[i] = running_add
        furRewards_dis=np.array(rewards)

        # -- Normalize reward
        mean = np.mean(furRewards_dis)
        std = np.std(furRewards_dis) + 1.0e-10
        rewards_normalized = (furRewards_dis - mean) / std

        # -- convert to tensor
        # rewards_final = torch.tensor(rewards_normalized, dtype=torch.float)
        log_probs = exps['logProbs']
        loss_arr=rewards_normalized*np.array(log_probs)

        # print(loss_arr)

        # log_probs=torch.tensor(log_probs,requires_grad=True)
        loss_tensor=torch.tensor(list(loss_arr),requires_grad=True)

        # final loss
        loss = torch.mean(loss_tensor)
        # print(loss)
        return loss


    def policy_update(self, policy_net,env, max_t, gamma=0.99,learning_rate=0.001):
        state=env.reset()

        # -- 收集一个episode的 trajectory
        experiences=defaultdict(list)

        for t in range(max_t):
            experiences["states"].append(state)
            # convert state to Tensor
            state = torch.from_numpy(state).float()
            state = Variable(state)

            # choose an action
            action,log_prob=self.act(policy_net,state)
            action=action.data.numpy().astype(int)[0]
            next_state, reward, done, _ = env.step(action)

            # check the finishing condition
            if done:
                break

            # collect s,a,r,s'...
            experiences["actions"].append(action)
            experiences["rewards"].append(reward)
            experiences["logProbs"].append(log_prob)
            state=next_state

        # -- calculate loss (every episode)
        loss=self.calculate_loss(experiences,gamma)

        # -- update policy network's parameters
        optimizer=torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_R=sum(experiences["rewards"])
        return total_R,loss.item()


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    env.seed(0)

    agent_pg = Agent_PG(state_size=4)
    agent_pg.policy_update(env,4)


















