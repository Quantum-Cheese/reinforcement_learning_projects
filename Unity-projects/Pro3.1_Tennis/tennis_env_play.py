from unityagents import UnityEnvironment
import numpy as np
import time

env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

def random_play():
    for i in range(1, 100):  # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            actions=list(actions)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            time.sleep(5)
            next_states = env_info.vector_observations  # get next state (for each agent)

            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))


if __name__=="__main__":
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    # print('Number of agents:', num_agents)
    # # size of each action
    action_size = brain.vector_action_space_size
    # print('Size of each action:', action_size)
    # # examine the state space
    # states = env_info.vector_observations
    # state_size = states.shape[1]
    # print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    # print('The state for the first agent looks like:', states[0])

    random_play()





