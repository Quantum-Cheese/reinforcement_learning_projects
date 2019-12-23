import numpy as np
import torch
from DNN_pytorch.model_mlp import Network
from torch.autograd import Variable

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

"""获取数据"""
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, D_out = 3, 8, 4

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

print("org data: input shape {};output shape {}".format(x.shape,y.shape))
"""定义网络"""

# 根据定义好的网络结构，获取mlp网络对象
net=Network()
# 定义损失函数和优化器
criterion=torch.nn.MSELoss()  # 均方误差
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


"""训练"""
epochs = 5
steps = 0
running_loss = 0
print_every = 20
for e in range(epochs):
    optimizer.zero_grad()
    inputs,targets=x,y
    output = net.forward(inputs)
    print(output.shape)
    trans_put=output[:,1]
    trans_tar=targets[:,1]
    print(trans_put.shape,trans_tar.shape)
    loss = criterion(trans_put, trans_tar)
    loss.backward()
    optimizer.step()

    running_loss+=loss.data.item()
    print(running_loss)

