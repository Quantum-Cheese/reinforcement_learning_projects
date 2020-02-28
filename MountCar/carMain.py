import sys
import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd

from MountCar.agent_ql import QLearningAgent
from MountCar.carRun import run
from MountCar.discretization import create_uniform_grid,discretize
from MountCar.visual import visualize_samples
from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

if __name__=="__main__":
    # 创建 gym 的 MountainCar环境
    env = gym.make('MountainCar-v0')
    env.seed(505)

    """创建不同的网格空间，分别训练智能体,并测试对比最终得分"""
    state_grid_1 = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
    q_agent_1 = QLearningAgent(env, state_grid_1)
    scores_1 = run(q_agent_1, env)
    test_scores_1 = run(q_agent_1, env, num_episodes=100, mode='test')

    state_grid_2 = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
    q_agent_2 = QLearningAgent(env, state_grid_2)
    scores_2 = run(q_agent_2, env)
    test_scores_2 = run(q_agent_2, env, num_episodes=100, mode='test')

    print("\nAgent_1 [TEST] Completed {} episodes with avg. score = {}".format(len(test_scores_1), np.mean(test_scores_1)))
    print("\nAgent_2 [TEST] Completed {} episodes with avg. score = {}".format(len(test_scores_2), np.mean(test_scores_2)))

    state = env.reset()
    score = 0
    img = plt.imshow(env.render(mode='rgb_array'))
    # todo 智能体动画图像显示不出来？
    for t in range(1000):
        action = q_agent_2.act(state, mode='test')
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        # display.display(plt.gcf())
        # display.clear_output(wait=True)
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print('Score: ', score)
            break

    env.close()





