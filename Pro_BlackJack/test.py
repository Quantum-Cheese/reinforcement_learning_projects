import sys
import gym
import numpy as np
from collections import defaultdict


env = gym.make('Blackjack-v0')

print(env.observation_space)
print(env.action_space)

# 用随机策略玩 BlackJack
for i_episode in range(3):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break
