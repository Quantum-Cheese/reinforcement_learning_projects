from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
import arrow
from collections import deque
import matplotlib.pyplot as plt
import torch
from Reacher.DDPG.agent_DDPG import DDPG
from Reacher.PPO.agent_PPO import PPO

env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


def output_results(i_episode,scores_deque,startTime):
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}\t Training time till now: {}'
              .format(i_episode, np.mean(scores_deque),arrow.now()-startTime))
    if np.mean(scores_deque) >= 30.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                     np.mean(scores_deque)))
        return True


def save_model(agent,model_file):
    if agent['name']=="DDPG":
        torch.save(agent['model'].actor_local.state_dict(), model_file[0])
        torch.save(agent['model'].critic_local.state_dict(), model_file[1])
    elif agent['name']=='PPO':
        torch.save(agent['model'].policy.state_dict(), model_file[0])
        torch.save(agent['model'].critic.state_dict(), model_file[1])


def plot_single_scores(scores,file_name):
    "绘制 single agent 训练曲线：单次训练单曲线图"
    x = np.arange(1, len(scores) + 1)
    rolling_mean = pd.Series(scores).rolling(100).mean()
    plt.plot(x, rolling_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(file_name)
    plt.show()


def train_ddpg(agent,n_episodes,model_file):
    scores_deque = deque(maxlen=100)
    scores = []
    startTime = arrow.now()
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the Environment
        state = env_info.vector_observations  # get the initial state
        score = np.zeros(num_agents)
        agent.reset()  # reset the Agent
        # -- for every time step:
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            # agent training process
            agent.step(i_episode, state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):
                break

        scores_deque.append(score)
        scores.append(score)
        if output_results(i_episode, scores_deque, startTime):
            agent_dict = {"name": "DDPG", "model": agent}
            save_model(agent_dict, model_file)
            break

    return scores


def train_PPO(agent,n_episode,model_file):
    scores_deque = deque(maxlen=100)
    scores = []
    startTime=arrow.now()
    for i_episode in range(1, n_episode + 1):
        total_reward = agent.train(env,brain_name)

        # record scores(total rewards) per episode
        scores_deque.append(total_reward)
        scores.append(total_reward)
        if output_results(i_episode,scores_deque,startTime):
            agent_dict={"name":"PPO","model":agent}
            save_model(agent_dict, model_file)
            break

    return scores


if __name__=="__main__":
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size

    # create DDPG and PPO agents
    # agent_ddpg=DDPG(state_size,action_size,0)
    agent_ppo=PPO(state_size,action_size)

    # train_ddpg(agent_ddpg,10,'')
    # train_PPO(agent_ppo, 10, [])

    ppo_score=train_PPO(agent_ppo,2000,["ppo_actor.pth","ppo_critic.pth"])
    plot_single_scores(ppo_score, 'ppo-2.png')

