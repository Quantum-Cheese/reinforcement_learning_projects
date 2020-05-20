from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gym
import arrow
import torch
from BipedalWalker.TD3.TD3_agent import TD3

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 100       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.005              # for soft update of target parameters
NOISE=0.2
CLIP=0.5
POLICY_FREQ=2


def train_td3(env,agent,n_episodes):
    scores_deque = deque(maxlen=100)
    scores = []
    start_time = arrow.now()
    begin_time=arrow.now()
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        total_reward = 0
        time_step=0

        # loop over time steps
        while True:
            # 智能体选择动作（根据当前策略）
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # 存储经验
            agent.memory.add((state, action, next_state,reward, done))
            time_step+=1
            state = next_state
            total_reward += reward
            if done:
                break

        # recording scores
        scores.append(total_reward)
        scores_deque.append(total_reward)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'
              .format(i_episode, np.mean(scores_deque), total_reward),end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\t Running time for 100 episode:{}'
                  .format(i_episode, np.mean(scores_deque),arrow.now()-start_time))
        if np.mean(scores_deque)>=300:
            print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}\t Total running time"{}'
                  .format(i_episode,np.mean(scores_deque),arrow.now()-begin_time))
            agent.save('models','TD3')
            break
            return scores

        # train the agent after finishing current episode
        agent.train(iterations=time_step, batch_size=BATCH_SIZE, discount=GAMMA, tau=TAU, policy_noise=NOISE,
                    noise_clip=CLIP,policy_freq=POLICY_FREQ)

    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores.size()) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('TD3_1.png')
    plt.show()


if __name__=="__main__":
    env = gym.make('BipedalWalker-v2')
    env.seed(10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent=TD3(state_dim,action_dim,max_action,env)
    # 训练并保存 scores
    scores=train_td3(env,agent,2000)
    plot_scores(scores)
