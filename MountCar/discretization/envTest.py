import gym
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
plt.ion()


# 创建 gym 的 MountainCar环境
env = gym.make('MountainCar-v0')
env.seed(505)

# 运行一个随机的 agent
state = env.reset()
print(state)
img = plt.imshow(env.render(mode='rgb_array'))
for t in range(1000):
    action = env.action_space.sample()
    img.set_data(env.render(mode='rgb_array'))
    plt.axis('off')
    state, reward, done, _ = env.step(action)
    if done:
        print('Score: ', t + 1)
        break

env.close()

# Explore state (observation) space
print("State space:", env.observation_space)
print("Action space:", env.action_space)


print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Generate some samples from the state space
print("State space samples:")
print(np.array([env.observation_space.sample() for i in range(10)]))

# Explore the action space
print("Action space:", env.action_space)

# Generate some samples from the action space
print("Action space samples:")
print(np.array([env.action_space.sample() for i in range(10)]))


# 创建一个用于离散化空间的网格世界
# state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
# # 从 env 中取样，离散化这些点的坐标并可视化
# state_samples = np.array([env.observation_space.sample() for i in range(10)])
# discretized_state_samples = np.array([discretize(sample, state_grid) for sample in state_samples])
# visualize_samples(state_samples, discretized_state_samples, state_grid,
#                       env.observation_space.low, env.observation_space.high)
# plt.xlabel('position')
# plt.ylabel('velocity')  # axis labels for MountainCar-v0 state space



