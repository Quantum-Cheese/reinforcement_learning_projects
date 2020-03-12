import torch
import gym
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from CartPole.Policy_Gradient.agent_PG import Agent_PG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_file="models/pg_model_3.pth"
# plot_file="plots/pg_3.png"


def watch_smart_agent(agent,model_name):
    agent.policy.load_state_dict(torch.load(model_name))
    state = env.reset()
    for t in range(1000):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action,_ = agent.policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            print("done in time step {}".format(t+1))
            break
    env.close()


def plot_scores(scores,plot_file):
    x=np.arange(1, len(scores) + 1)
    # plt.plot(x, scores)
    rolling_mean = pd.Series(scores).rolling(100).mean()
    plt.plot(x, rolling_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(plot_file)
    plt.show()


def test_agent(agent,n_episode,model_name):
    agent.policy.load_state_dict(torch.load(model_name))
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episode + 1):
        rewards=[]
        state = env.reset()
        while True:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 升维 1d->2d
            action, _ = agent.policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores.append(sum(rewards))

    return scores


def train_agent(env,agent,n_episode,max_t,model_file):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episode + 1):
        total_reward,loss=agent.train(env,max_t)

        # record scores(total rewards) per episode
        scores_deque.append(total_reward)
        scores.append(total_reward)
        if i_episode % 100 == 0:
            print('Episode {}\t Loss: {}\t Average Score: {:.2f}'.format(i_episode, loss,
                                                                         np.mean(scores_deque)))
        # if np.mean(scores_deque) >= 195.0:
        #     print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}\n----------\n'.format(i_episode,
        #                                                                                np.mean(scores_deque)))
        #     torch.save(agent.policy.state_dict(),model_file)
            # break

    return scores


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    agent_pg = Agent_PG(state_size=4,action_size=2,type="pg")
    agent_rf=Agent_PG(state_size=4,action_size=2,type="reinforce")

    """两种不同算法训练 agent"""
    n_episode=1500

    rf_scores=train_agent(env,agent_rf,n_episode,1500,'models/reinforce_model_7.pth')
    pg_socres=train_agent(env,agent_pg,n_episode,1500,'models/pg_model_4.pth')

    # 绘制训练曲线
    x=np.arange(1, n_episode+1)
    plt.plot(x,pd.Series(rf_scores).rolling(100).mean(),label="reinforce")
    plt.plot(x, pd.Series(pg_socres).rolling(100).mean(), label="pg")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig("plots/rf-vs-pg_4.png")
    plt.show()

    # 观察训练好的智能体
    # watch_smart_agent(agent_pg,model_file)

    """测试 agent"""
    # pg_scores=test_agent(agent_pg,1000,'models/pg_model_3.pth')






