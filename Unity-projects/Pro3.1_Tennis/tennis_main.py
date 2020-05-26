from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
from multi_agents import MultiAgents
import matplotlib.pyplot as plt

NUM_AGENTS=2


def train_agents(multi_agents,n_episodes):
    all_scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        multi_agents.reset()
        # reset the env
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        # print(states.shape)
        scores = np.zeros(NUM_AGENTS)

        # --- for every time step
        while True:
            # -- get actions for all agents (based on their own observations)
            actions = multi_agents.agents_act(states)
            # -- all agents interact with env at the same time
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards   # list [2]
            next_states = env_info.vector_observations
            dones = env_info.local_done  # list [2]
            # -- store the experiments, update agents
            multi_agents.agents_step(states, actions, rewards, next_states, dones)
            states=next_states
            scores += rewards
            if np.any(dones):
                break

        # record the scores (所有 agent scores 的平均值)
        scores_window.append(np.mean(scores))
        all_scores.append(np.mean(scores))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 5,
                                                                                         np.mean(scores_window)))
            torch.save(multi_agents.ddpg_agents[0].actor_local.state_dict(), 'models_v3/agent1_actor.pth')
            torch.save(multi_agents.ddpg_agents[0].critic_local.state_dict(), 'models_v3/agent1_critic.pth')
            torch.save(multi_agents.ddpg_agents[1].actor_local.state_dict(), 'models_v3/agent2_actor.pth')
            torch.save(multi_agents.ddpg_agents[1].critic_local.state_dict(), 'models_v3/agent2_critic.pth')
            break

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('plot_test2.png')
    plt.show()


def test_agents(multi_agents,file_names,n_episode):
    # load network parameters for trained agents
    multi_agents.ddpg_agents[0].actor_local.load_state_dict(torch.load(file_names['a1']))
    multi_agents.ddpg_agents[1].actor_local.load_state_dict(torch.load(file_names['a2']))

    # interact with env
    score_test=[]
    for i_episode in range(1, n_episode + 1):
        # reset
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(NUM_AGENTS)

        while True:
            # -- get actions for all agents (based on their own observations)
            actions = multi_agents.agents_act(states)

            # -- all agents interact with env at the same time
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards  # list [2]
            next_states = env_info.vector_observations
            dones = env_info.local_done  # list [2]
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        score_test.append(np.mean(scores))

    return score_test


if __name__=="__main__":

    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    state_size=env_info.vector_observations.shape[1]
    action_size=brain.vector_action_space_size

    multi_agents = MultiAgents(state_size, action_size,NUM_AGENTS,seed=10)
    train_socres = train_agents(multi_agents,n_episodes=500)
    # plot_scores(train_socres)

    file_names={'a1':'models_v2/agent1_actor.pth','a2':'models_v2/agent2_actor.pth'}

    # test_scores=test_agents(multi_agents, file_names,n_episode=300)
    # plot_scores(test_scores)


