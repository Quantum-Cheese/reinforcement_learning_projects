import os
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import time
import random


def loop(s_num,scores,num):
    # while True:
    for _ in range(10):
        time.sleep(1)
        print('parent process:', os.getppid())  # 获取父进程的pid
        print('process id:', os.getpid())  # 获取子进程的pid
        s_num.value+=1      # 进程之间共享的变量
        num+=1              # 非公享变量，进程独立
        # print("shared: ",s_num.value)
        # print("individual : ",num)
        scores.put(random.randint(1,20))
        pass


def f(s_num,scores):
    num=0
    loop(s_num,scores,num)


def multi_process():
    # 定义共享变量，在进程之间共享
    shared_num=mp.Value('i', 0)
    scores=mp.Queue()
    processes = []
    results=[]
    for i in range(4):
        p = Process(target=f,args=(shared_num,scores))
        p.start()
        processes.append(p)
    while True:
        r = scores.get()
        if r is not None:
            results.append(r)
            print("!!! ", results)
        else:
            break
    for p in processes:
        p.join()

    print("EDN !!! ",results)



if __name__=="__main__":
    multi_process()
    # loop(0)
