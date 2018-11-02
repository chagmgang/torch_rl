import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy

class RND(nn.Module):
    def __init__(self):
        super(RND, self).__init__()
        self.rnd_layer_1 = nn.Linear(4, 60)
        self.rnd_layer_2 = nn.Linear(60, 60)
        self.rnd_layer_3 = nn.Linear(60, 4)

    def forward(self, x):
        rnd = F.relu(self.rnd_layer_1(x))
        rnd = F.relu(self.rnd_layer_2(rnd))
        rnd = self.rnd_layer_3(rnd)
        return rnd

def predict_intrincit_reward(target, main, state):
    s = Variable(torch.from_numpy(state)).float()
    main_embed = main(s)
    target_embed = target(s)
    error = main_embed - target_embed
    reward = error.pow(2)
    reward = torch.sum(reward).data.numpy()
    return reward

def rnd_train(target, main, state):
    optimizer = torch.optim.Adam(main.parameters(), lr=0.003)
    optimizer.zero_grad()
    s = Variable(torch.from_numpy(state)).float()
    main_embed = main(s)
    target_embed = target(s)
    error = main_embed - target_embed
    reward = error.pow(2)
    loss = torch.sum(reward, dim=1).mean()
    loss.backward()
    optimizer.step()
    return loss.data.numpy()