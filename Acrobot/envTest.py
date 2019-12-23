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