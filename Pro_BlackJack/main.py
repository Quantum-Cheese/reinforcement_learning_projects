"""
使用 GLIE MC Control 算法解决 BlackJack
"""
import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')


# 定义获取概率的函数(根据 epsilon-greedy policy 设定动作戏选择概率)
def get_probs(q_s,nA,epsilon):
    # 先把所有的动作概率都设置为 epsilon/nA
    probs=np.array([epsilon/nA for i in range(nA)])
    # 根据Q[S]找到最优动作，将其概率更新为 1-epsilon+epsilon/nA
    best_a=np.argmax(q_s)
    probs[best_a]=1-epsilon+(epsilon/nA)
    return probs


# 根据epsilon-greedy 策略产生一个 episode
def generate_episode(Q,env,nA,epsilon):
    episode = []
    state = env.reset()
    while True:
        # choose the action in that state based on probs
        if state in Q:
            probs=get_probs(Q[state],nA,epsilon)
            action = np.random.choice(np.arange(nA), p=probs)
        else:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


# 根据 一个episode 更新 Q table
def update_q(Q,episode,gamma,alpha):
    states,actions,rewards=zip(*episode)
     # set discounts
    discounts=np.array([gamma**i for i in range(len(rewards))])
    # for every time step in the episode(loop over states)
    for i,state in enumerate(states):
        old_Q=Q[state][actions[i]]
        # check if current state is the first visit
        if(states.index(state)==i):
            # caculate the Gt for that state
            sum_rewards=sum(rewards[i:]*discounts[:(len(discounts)-i)])
            # update Q value for the current state,action
            Q[state][actions[i]]=old_Q+alpha*(sum_rewards-old_Q)
    return Q


# MC Control 算法主体
def mc_control(env, num_episodes, alpha, gamma=1.0, eps_decay=.99999, eps_start=1.0, eps_min=0.05):
    """
    :param env: BlackJack环境对象
    :param num_episodes: 总迭代次数（实验的episode数目）
    :param alpha: Q tabel 更新公式的参数
    :param gamma: 折扣率
    :param eps_decay:  epsilon 衰减率
    :param eps_start: epsilon 初始值
    :param eps_min: epsilon 衰减下限
    :return:
    """
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # set the initial value for epsilon
    epsilon = eps_start

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # set current epsilon value
        epsilon = max(epsilon * eps_decay, eps_min)

        # generate an episode
        episode = generate_episode(Q, env, nA, epsilon)

        # update the Q table
        Q = update_q(Q, episode, gamma, alpha)

    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k, np.argmax(v)) for k, v in Q.items())

    return policy, Q


if __name__=="__main__":
    # 设定实验次数和 alpha 参数
    policy, Q = mc_control(env, 500000, 0.02)

    # 根据最终的Q table 绘制值函数
    # obtain the corresponding state-value function
    V = dict((k, np.max(v)) for k, v in Q.items())
    # plot the state-value function
    plot_blackjack_values(V)

    # 绘制生成的最佳策略
    plot_policy(policy)
