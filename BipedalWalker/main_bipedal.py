from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from BipedalWalker.DDPG.DDPG_agent import DDPGAgent


eps_start=1.0
eps_end=0.01
eps_decay=0.995


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores.size()) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def agent_train(env,agent_name,n_episodes=2000, max_t=700):

    if agent_name=='DDPG':
        agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], seed=10)

    scores_deque = deque(maxlen=100)
    scores = []
    eps = eps_start  # epsilon for TD3

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()

        score = 0
        while True:
            # 智能体生成与当前 state 对应的 action （行动策略）
            action = agent.act(state)
            # 与环境交互，得到 sars'
            next_state, reward, done, _ = env.step(action)
            # 把当前时间步的经验元组传给 agent
            agent.step(i_episode,state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        # epsilon decay with time
        eps = max(eps_end, eps_decay * eps)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'
              .format(i_episode, np.mean(scores_deque), score),end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'model_save/actor2.pth')
            torch.save(agent.critic_local.state_dict(), 'model_save/critic2.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


if __name__=="__main__":
    env = gym.make('BipedalWalker-v2')
    env.seed(10)

    train_scores=agent_train(env, 'DDPG', n_episodes=3000)
    plot_scores(train_scores)
