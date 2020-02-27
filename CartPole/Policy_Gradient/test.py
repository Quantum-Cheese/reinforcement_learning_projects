import torch
import numpy as np


# a1=torch.tensor([-0.6551],requires_grad=True)
# a2=torch.tensor([-2.6551],requires_grad=True)
# a3=torch.tensor([-2.6551],requires_grad=True)
#
# torch_lst=[a1,a2,a3]
# print(torch_lst)
#
# b=torch.tensor(torch_lst,requires_grad=True)
# print(b)
# print(b.requires_grad)

# reward_pool=[-1.2,-3.4,-5.6,0, -3.5,1.2,5,2,0, -4.6,3.5,-1,0]
# print(reward_pool)
#
# running_add = 0
# for i in reversed(range(len(reward_pool))):
#     if reward_pool[i] == 0:
#         running_add = 0
#     else:
#         running_add = running_add * 0.99 + reward_pool[i]
#         reward_pool[i] = running_add
#
# print(reward_pool)
#
# reward_pool=[-1.2,-3.4,-5.6,0, -3.5,1.2,5,2,0, -4.6,3.5,-1,0]
# print(reward_pool)

reward_pool=[1,1,1,1,1,1,1]
running_add = 0
discount = 0.99 ** np.arange(len(reward_pool))
discount=list(reversed(discount))
for i in reversed(range(len(reward_pool))):
    running_add = running_add * discount[i] + reward_pool[i]
    reward_pool[i] = running_add
print(reward_pool)


rewards=[1,1,1,1,1,1,1]
discount = 0.99 ** np.arange(len(rewards))
dis_rewards = np.asarray(rewards) * discount

rewards_2=np.cumsum(np.flipud(dis_rewards))
rewards_2=np.flipud(rewards_2)
print(rewards_2)