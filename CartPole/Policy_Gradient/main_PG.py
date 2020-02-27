import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from CartPole.Policy_Gradient.agent_PG import Agent_PG
from CartPole.Policy_Gradient.model import Policy


def policy_gradient(env,agent,n_episode):

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    loss_pool=deque(maxlen=100)

    policy=Policy(state_size=4,action_size=1)

    for i in range(1,n_episode+1):
        total_reward,loss=agent.policy_update(policy,env,max_t=1000)
        scores.append(total_reward)
        scores_window.append(total_reward)
        loss_pool.append(loss)

        if i % 200 == 0:
            print("episode:{} \n Policy network loss:{} \t Total rewards: {}\n".
                  format(i, np.mean(loss_pool),np.mean(scores_window)))



if __name__=="__main__":
    env = gym.make('CartPole-v0')
    agent_pg = Agent_PG(state_size=4)
    policy_gradient(env,agent_pg,n_episode=5000)

