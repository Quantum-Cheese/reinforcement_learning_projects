# Import common libraries
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# Create an environment
env = gym.make('Acrobot-v1')
env.seed(505)

# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Explore action space
print("Action space:", env.action_space)

state = env.reset()
for _ in range(1000):
    env.render()
    action = np.random.uniform(low=-1.0, high=1.0,size=3)
    print(action)
    next_state, reward, done, _ = env.step(action)
    print(next_state, reward)
    if done:
        break
