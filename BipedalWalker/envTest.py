import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v2')
env.seed(10)

print('observation space:', env.observation_space)
print('action space:', env.action_space)

# 观察一个未经训练的随机智能体
state = env.reset()
print(state,type(state))

for _ in range(1000):
    # env.render()
    action = np.random.uniform(low=-1.0, high=1.0,size=4)
    print("action",action)
    next_state, reward, done, _ = env.step(action)
    print(next_state, reward)
    if done:
        break

env.close()
