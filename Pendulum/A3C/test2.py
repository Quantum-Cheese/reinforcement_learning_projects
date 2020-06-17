import os
import torch.multiprocessing as mp
import random
import time

max_epidoe=20


class Worker(mp.Process):
    def __init__(self,global_ep, global_ep_r, res_queue):
        super(Worker, self).__init__()
        self.global_epi=global_ep
        self.global_re=global_ep_r
        self.scores=res_queue

    def run(self):
        while self.global_epi.value < max_epidoe:
            re=random.randint(0,100)
            with self.global_epi.get_lock():
                self.global_epi.value += 1
            with self.global_re.get_lock():
                self.global_re.value=re
            self.scores.put(self.global_re.value)

        self.scores.put(None)


if __name__=="__main__":
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    results = []

    workers = [Worker(global_ep, global_ep_r, res_queue) for i in range(4)]
    [w.start() for w in workers]
    while True:
        r = res_queue.get()
        if r is not None:
            results.append(r)
        else:
            break
    [w.join() for w in workers]

    print(global_ep.value, len(results),results)

