import gym
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt


def make_normal_env():
    env = gym.make('BipedalWalker-v3')
    env.seed(10)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    return env


def make_hardcore():
    env = gym.make('BipedalWalkerHardcore-v3')
    env.seed(10)
    return env


# 观察一个未经训练的随机智能体
def random_without_break(env):
    for _ in range(3000):
        env.reset()
        env.render()
        for t in range(1000):
            action = np.random.uniform(low=-1.0, high=1.0, size=4)
            next_state, reward, done, _ = env.step(action)
    env.close()


def random_with_break(env):
    env.reset()
    for _ in range(1000):
        env.render()
        action = np.random.uniform(low=-1.0, high=1.0, size=4)
        next_state, reward, done, _ = env.step(action)
        if done:
            break
    env.close()


normal_env=make_normal_env()
hardcore_env=make_hardcore()
random_with_break(hardcore_env)


