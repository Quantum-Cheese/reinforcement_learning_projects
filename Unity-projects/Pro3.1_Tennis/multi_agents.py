import torch
import numpy as np
from ddpg_agent import DDPGAgent
from ddpg_untils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512       # minibatch size
UPDATE_EVERY=5
NUM_UPDATE=10
GAMMA = 0.99


class MultiAgents():

    def __init__(self,state_size, action_size,num,seed):
        # shared memory for all agents
        self.memory=ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,seed, device)
        # define each agent
        self.ddpg_agents=[DDPGAgent(state_size, action_size, seed,self.memory,device) for _ in range(num)]

        self.t_step=0
        self.num_agents=num

    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()

    def agents_act(self,states_all):
        """
        states_all: np 2d array (2,24)
        return: list of 2d array (1,2)
        """
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.ddpg_agents, states_all)]
        return actions

    def agents_step(self,states, actions, rewards, next_states, dones):
        """
        """
        self.t_step+=1
        # -- 存储经验元组，把不同 agent 的经验组拆散，依次存放入 memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) >= BATCH_SIZE:
            if self.t_step%UPDATE_EVERY==0:
                for _ in range(NUM_UPDATE):
                    for n in range(self.num_agents):
                        # sample 得到的是经验是从memory里随机批量采样的(可能包括不同 agent 的经验组，在更新的时候每个 agent 都不加区分的共用经验)
                        experiences = self.memory.sample()

                        # update every agent
                        self.ddpg_agents[n].learn(experiences, GAMMA)

                        # print("time step{}, update agent!".format(self.t_step))






