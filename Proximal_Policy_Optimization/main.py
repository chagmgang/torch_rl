import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.actor_layer_1 = nn.Linear(4, 60)
        self.actor_layer_2 = nn.Linear(60, 60)
        self.actor_layer_3 = nn.Linear(60, 2)
        self.critic_layer_1 = nn.Linear(4, 60)
        self.critic_layer_2 = nn.Linear(60, 60)
        self.critic_layer_3 = nn.Linear(60, 1)

    def forward(self, x):
        actor = F.relu(self.actor_layer_1(x))
        actor = F.relu(self.actor_layer_2(actor))
        actor = F.softmax(self.actor_layer_3(actor))
        critic = F.relu(self.critic_layer_1(x))
        critic = F.relu(self.critic_layer_2(critic))
        critic = self.critic_layer_3(critic)
        return actor, critic