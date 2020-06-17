import random
import torch
import torch.optim as optim
import multiprocessing as mp
from multiprocessing import Process
from Pendulum.A3C.untils import ValueNetwork,PolicyNetwork
from Pendulum.A3C.worker import Worker

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A3C():
    def __init__(self,env,max_episode,lr,gamma):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.max_episode=max_episode
        self.global_episode = mp.Value('i', 0)  # 进程之间共享的变量
        self.global_epi_rew = mp.Value('d',0)
        self.rew_queue = mp.Queue()
        self.worker_num = mp.cpu_count()

        # define the global networks
        self.global_policyNet = PolicyNetwork(self.state_size,self.action_size).to(device)
        self.global_policyNet.share_memory()      # global 的网络参数放入 shared memory，以便复制给各个进程中的 worker网络
        self.global_valueNet= ValueNetwork(self.state_size,1).to(device)
        self.global_valueNet.share_memory()
        # global optimizer
        self.global_optimizer_policy = optim.Adam(self.global_policyNet.parameters(), lr=lr)
        self.global_optimizer_value = optim.Adam(self.global_valueNet.parameters(),lr=lr)

        # define the workers
        self.workers=[Worker(env,i,self.global_valueNet,self.global_optimizer_value,
                             self.global_policyNet,self.global_optimizer_policy,
                             self.global_episode,self.global_epi_rew,self.rew_queue,
                             self.max_episode,gamma)
                      for i in range(self.worker_num)]

    def train_worker(self):
        scores=[]
        [w.start() for w in self.workers]
        while True:
            r = self.rew_queue.get()
            if r is not None:
                scores.append(r)
            else:
                break
        [w.join() for w in self.workers]

        return scores

    def save_model(self):
        torch.save(self.global_valueNet.state_dict(), "a3c_value_model.pth")
        torch.save(self.global_policyNet.state_dict(), "a3c_policy_model.pth")




