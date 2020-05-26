import gym
import numpy as np

env = gym.make("AirRaid-v0")
observation = env.reset()
print(observation)

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()

