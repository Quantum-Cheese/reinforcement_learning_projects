import gym
import numpy as np
env = gym.make('MountainCarContinuous-v0')
env.seed(101)

print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('  - low:', env.action_space.low)
print('  - high:', env.action_space.high)

env.reset()
for _ in range(1000):
    env.render()
    action=np.random.uniform(low=env.action_space.low, high=env.action_space.high)

    next_state, reward, done, _ =env.step(action)
    if done:
        break

