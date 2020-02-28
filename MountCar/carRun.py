import sys
import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd

from MountCar.agent_ql import QLearningAgent
from MountCar.discretization import create_uniform_grid,discretize
from MountCar.visual import visualize_samples


def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes + 1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        #print("action",action)
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            try:
                state, reward, done, info = env.step(action)
                total_reward += reward
                action = agent.act(state, reward, done, mode)
            except Exception as e:
                print(e)
                print("episode",i_episode)
                print(action)
                print(type(action))

        # Save final score
        scores.append(total_reward)

        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores


def plot_scores(scores, filename,rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores)
    plt.title("Scores")
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()

    plt.plot(rolling_mean)
    plt.savefig(filename)
    plt.show()
    return rolling_mean


def plot_q_table(q_table,filename):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(q_image, cmap='jet')
    cbar = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    plt.savefig(filename)
    plt.show()


if __name__=="__main__":
    # 创建 gym 的 MountainCar环境
    env = gym.make('MountainCar-v0')
    env.seed(505)

    # 创建一个用于离散化空间的网格世界
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
    print(state_grid)
    # 创建一个q-learning agent (train 模式)
    q_agent = QLearningAgent(env, state_grid)

    # -----------------------------------------------------------------

    """训练：学习 q table"""
    # 运行并得到总奖励序列（每个episode一个累计奖励）
    scores = run(q_agent, env)
    # 获取累计奖励的 rolling mean
    rolling_mean = plot_scores(scores,'images/trainScore_1.png')
    print(rolling_mean)
    # 绘制Q table 可视化
    plot_q_table(q_agent.q_table, 'images/qtable_1.png')

    """测试：冻结学习过程，用已有的q table直接进行贪婪策略选择"""
    # Run in test mode and analyze scores obtained
    test_scores = run(q_agent, env, num_episodes=100, mode='test')
    print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
    _ = plot_scores(test_scores, 'images/testScore_1.png',rolling_window=10)



