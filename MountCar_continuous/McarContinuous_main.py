from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import arrow

from MountCar_continuous.DDPG.agent_ddpg import DDPGAgent
from MountCar_continuous.TD3.TD3_new import TD3


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


def train_td3(env,agent,n_episodes):
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
            agent.save_exp(state, action, next_state, reward, done)
            if agent.mode==1:
                agent.train(time_step)
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

        if agent.mode==0:
            agent.train(time_step)

    return scores


if __name__=="__main__":
    env = gym.make('MountainCarContinuous-v0')
    env.seed(10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent_0 = TD3(state_dim,action_dim,max_action,env,0)      # mode=0:update per episode
    agent_1 = TD3(state_dim, action_dim, max_action, env, 1)  # mode=1: update per time step

    scores=train_td3(env,agent_1,1000)