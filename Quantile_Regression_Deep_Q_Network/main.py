import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import deque
import random

class Model(nn.Module):
    def __init__(self, num_support):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(4, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 256)
        self.layer_4 = nn.Linear(256, 2 * num_support)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x.view(-1, 2, num_support)

def assign_parameter(target_network, main_network):
    target_network.load_state_dict(main_network.state_dict())

def choose_action(net, state):
    state = Variable(torch.from_numpy(state)).float()
    action = net(state).data.numpy()[0]
    action_length = len(action)
    action = np.mean(action, axis=1)
    action = np.argmax(action)
    action = np.eye(action_length)[action]
    return action


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def train(batch_size, num_support, memory, target_network, main_network):
    optimizer = torch.optim.Adam(main_network.parameters(), lr=1e-3)
    optimizer.zero_grad()
    minibatch = random.sample(memory, batch_size)
    state_stack = [mini[0] for mini in minibatch]
    next_state_stack = [mini[1] for mini in minibatch]
    action_stack = [mini[2] for mini in minibatch]
    reward_stack = [mini[3] for mini in minibatch]
    done_stack = [int(mini[4]) for mini in minibatch]

    tensor_state = Variable(torch.from_numpy(np.array(state_stack))).float()
    tensor_next_state = Variable(torch.from_numpy(np.array(next_state_stack))).float()
    tensor_action = Variable(torch.from_numpy(np.array(action_stack))).float()
    tensor_reward = Variable(torch.from_numpy(np.array(reward_stack))).float()
    tensor_reward = torch.unsqueeze(tensor_reward, dim=1)
    tensor_done = Variable(torch.from_numpy(np.array(done_stack))).float()
    tensor_done = torch.unsqueeze(tensor_done, dim=1)

    tau = torch.Tensor((2 * np.arange(num_support) + 1) / (2.0 * num_support)).view(1, -1)

    Znext = target_network(tensor_next_state)
    Znext = Znext.detach()
    Znext_max = Znext[np.arange(batch_size), Znext.mean(2).max(1)[1]]
    Ttheta = tensor_reward + 0.99 * (1 - tensor_done) * Znext_max
    
    Z = main_network(tensor_state)
    tensor_action = torch.unsqueeze(tensor_action, dim=2)
    Z = torch.mul(Z, tensor_action)
    theta = torch.sum(Z, dim=1)

    diff = Ttheta.t().unsqueeze(-1) - theta
    loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
    loss = loss.mean()
    
    loss.backward()
    optimizer.step()

num_support = 5
batch_size = 32
main_network = Model(num_support)
target_network = Model(num_support)
assign_parameter(target_network, main_network)

env = gym.make('CartPole-v0')
memory_size = 10000
memory = deque(maxlen=memory_size)

e = 0
for episode in range(1000):
    state = env.reset()
    e = 1. / ((episode / 10) + 1)
    done = False
    global_step = 0
    state_list, action_list, next_state_list, reward_list, done_list = [], [], [], [], []

    while not done:
        global_step += 1
        if np.random.rand() < e:
            action = np.eye(2)[env.action_space.sample()]
        else:
            action = choose_action(main_network, state)
            
        next_state, reward, done, _ = env.step(np.argmax(action))

        if done:
            reward = -1
        else:
            reward = 0
        memory.append([state, next_state, action, reward, done])

        if len(memory) > 1000:
            train(batch_size, num_support, memory, target_network, main_network)
            if global_step % 5 == 0:
                assign_parameter(target_network, main_network)
        state = next_state
    #train(batch_size, num_support, memory, target_network, main_network)
    print(episode, global_step, len(memory))