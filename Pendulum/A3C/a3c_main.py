import gym
import numpy as np
from Pendulum.A3C.agent_a3c import A3C
import matplotlib.pyplot as plt

def train_a3c():
    agent = A3C(env,GLOBAL_MAX_EPISODE,lr,gamma)
    scores = agent.train_worker()
    return scores


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    gamma = 0.9
    lr = 1e-4
    GLOBAL_MAX_EPISODE = 5000

    train_scores = train_a3c()

    plt.plot(np.arange(1, len(train_scores) + 1), train_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('Pendulum_A3C_1.png')
    plt.show()