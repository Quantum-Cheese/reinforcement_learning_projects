import arrow
import torch
import gym
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from MountCar_continuous.PPO.PPO import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def plot_scores(scores,file_name):
    "绘制 single agent 训练曲线：单次训练单曲线图；多次训练多条曲线"
    x = np.arange(1, len(scores) + 1)
    rolling_mean = pd.Series(scores).rolling(100).mean()
    plt.plot(x, rolling_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(file_name)
    plt.show()


def train_agent(env,agent,n_episode,model_file):
    scores_deque = deque(maxlen=100)
    scores = []
    start_time = arrow.now()
    for i_episode in range(1, n_episode + 1):
        total_reward=agent.train(env)

        # record scores(total rewards) per episode
        scores_deque.append(total_reward)
        scores.append(total_reward)

        if i_episode % 100 == 0:
            print('Episode {}\t Average Score: {:.2f}'.format(i_episode,np.mean(scores_deque)))
            print('Running time for the last 100 episode:{}'.format(arrow.now()-start_time))
            start_time = arrow.now()

        if np.mean(scores_deque) >= 90:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}\n----------\n'.format(i_episode,
                                                                                       np.mean(scores_deque)))
            torch.save(agent.policy.state_dict(),model_file)
            break

    return scores


if __name__=="__main__":
    env = gym.make('MountainCarContinuous-v0')
    agent = PPO(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])

    train_scores=train_agent(env,agent,3000,'ppo_1.pth')
    plot_scores(train_scores,'ppo_1.png')



