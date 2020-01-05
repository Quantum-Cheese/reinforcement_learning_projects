import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from LunarLander.DQN.dqn import Agent

from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


# 观察一个未经训练的随机智能体
state = env.reset()
for _ in range(100):
    env.render()
    next_state, reward, done, _ =env.step(env.action_space.sample())


# 观察训练好的DQN智能体
agent = Agent(state_size=8, action_size=4, seed=0)
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

state = env.reset()
for i in range(3):
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(200):
        action = agent.act(state)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')

        state, reward, done, _ = env.step(action)
        if done:
            break

