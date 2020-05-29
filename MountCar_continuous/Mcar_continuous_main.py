from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import arrow

from MountCar_continuous.DDPG.agent_ddpg import DDPGAgent
from MountCar_continuous.TD3.TD3_v1 import TD3_V1
from MountCar_continuous.TD3.TD3_v2 import TD3_V2


def output_scores(start_time,i_episode,scores_deque,score):
    print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'
          .format(i_episode, np.mean(scores_deque), score), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}\t Running time til now :{}'
              .format(i_episode, np.mean(scores_deque),arrow.now()-start_time))
    if np.mean(scores_deque) >= 90:
        print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}\t Total running time :{}'
                .format(i_episode, np.mean(scores_deque),arrow.now()-start_time))
        return True

    return False


def train_td3_v1(env,agent,n_episodes):
    """更新频率：每个 time step 更新"""
    scores_deque = deque(maxlen=100)
    scores = []
    start_time = arrow.now()
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        # loop over time steps
        time_step=0
        while True:
            # 智能体选择动作（根据当前策略）
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # agent 存储经验并训练更新
            agent.step(time_step, state, action, next_state, reward, done)
            state = next_state
            time_step+=1
            total_reward += reward
            if done:
                break

        # recording scores
        scores.append(total_reward)
        scores_deque.append(total_reward)

        finished = output_scores(start_time, i_episode, scores_deque, total_reward)
        if finished:
            agent.save('model_save', 'TD3_v1')
            break

    return scores


def train_td3_v2(env,agent,n_episodes):
    """更新频率：跑完一整个episode再更新"""
    scores_deque = deque(maxlen=100)
    scores = []
    start_time = arrow.now()
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        time_step = 0

        # loop over time steps
        while True:
            # 智能体选择动作（根据当前策略）
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # 存储经验
            agent.memory.add((state, action, next_state, reward, done))
            time_step += 1
            state = next_state
            total_reward += reward
            if done:
                break

        # recording scores
        scores.append(total_reward)
        scores_deque.append(total_reward)
        finished = output_scores(start_time, i_episode, scores_deque, total_reward)
        if finished:
            agent.save('model_save', 'TD3_v2')
            break

        # train the agent after finishing current episode
        agent.train(iterations=time_step)

    return scores


def watch_agent(agent,filename_actor,filename_crtic):
    agent.actor_local.load_state_dict(torch.load(filename_actor))
    agent.critic_local.load_state_dict(torch.load(filename_crtic))
    state = env.reset()
    for t in range(1000):
        action = agent.act(state, noise=False)
        print(action)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break
    env.close()


def plot_scores(scores,filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)
    plt.show()


if __name__=="__main__":
    env = gym.make('MountainCarContinuous-v0')
    env.seed(10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 初始化 agent
    # td3_v1 = TD3_V1(state_dim, action_dim, max_action, env)
    td3_v2=TD3_V2(state_dim, action_dim, max_action, env)

    # 训练并保存 scores
    scores = train_td3_v2(env,td3_v2,2000)
    plot_scores(scores,'Mcar_TD3_v2.png')

    # watch_agent(agent,"actor1.pth","critic1.pth")


