import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


"""搭建网络结构"""
class Network(nn.Module):
    # 定义层
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 64)  # 输入维度 （N，8）
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # 输出的维度为 (N,4)

    # 前向传播
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        out=torch.sigmoid(x)

        return out

    # # 预测函数
    # def predict(self, x):
    #     ''' This function for predicts classes by calculating the softmax '''
    #     logits = self.forward(x)
    #     return F.softmax(logits)


net = Network()
print(net)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
