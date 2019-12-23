from collections import deque
import numpy as np
import gym

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]

# Memory parameters
memory_size = 10000  # memory capacity
batch_size = 20  # experience mini-batch size
pretrain_length = batch_size  # number experiences to pretrain the memory
# Create the Cart-Pole game environment
env = gym.make('CartPole-v1')

memory=Memory()

