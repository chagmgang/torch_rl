import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(5, 30)
        self.layer_2 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.softmax(self.layer_2(x))
        return x

net = Model()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

input = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
output = [[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]]

input = np.array(input)
output = np.array(output)

input = torch.from_numpy(input).float()
output = torch.from_numpy(output).float()

input = Variable(input)
output = Variable(output)

for i in range(160):
    optimizer.zero_grad()
    result = net(input)
    result = torch.log(result)
    loss = -torch.mul(result, output)
    loss = torch.sum(loss)
    loss.backward()
    optimizer.step()
    
result = net(input)
print(result)
net_1 = Model()
result = net_1(input)
print(result)
net_1.load_state_dict(net.state_dict())
result = net_1(input)
print(result)