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
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(4, 60)
        self.layer_2 = nn.Linear(60, 60)
        self.layer_3 = nn.Linear(60, 2)

    def forward(self, x):
        x = F.selu(self.layer_1(x))
        x = F.selu(self.layer_2(x))
        x = self.layer_3(x)
        return x

def assign_parameter(target_network, main_network):
    target_network.load_state_dict(main_network.state_dict())

def choose_action(net, state):
    tensor_state = Variable(torch.from_numpy(state).float())
    Q_s = net(tensor_state)
    action = torch.argmax(Q_s).data.numpy()
    action = np.eye(2)[action]
    return action

def train(batch_size, memory, target_network, main_network):
    optimizer = torch.optim.Adam(main_network.parameters(), lr=0.003)
    optimizer.zero_grad()

    minibatch = random.sample(memory, batch_size)
    state_stack = np.array([mini[0] for mini in minibatch])
    next_state_stack = np.array([mini[1] for mini in minibatch])
    action_stack = np.array([mini[2] for mini in minibatch])
    reward_stack = np.array([mini[3] for mini in minibatch])
    done_stack = np.array([int(mini[4]) for mini in minibatch])

    tensor_next_state = Variable(torch.from_numpy(next_state_stack).float())
    Qtarget_next_state = target_network(tensor_next_state)
    next_action = torch.argmax(Qtarget_next_state, dim=1).data.numpy()
    one_hot_next_action = Variable(torch.from_numpy(np.array([np.eye(2)[ac] for ac in next_action])).float())
    Qtarget_s_a = torch.sum(torch.mul(one_hot_next_action, Qtarget_next_state), dim=1)
    tensor_reward = Variable(torch.from_numpy(reward_stack).float())
    tensor_done = Variable(torch.from_numpy(done_stack).float())
    target = tensor_reward + (1-tensor_done)*0.99*Qtarget_s_a

    tensor_state = Variable(torch.from_numpy(state_stack).float())
    tensor_action = Variable(torch.from_numpy(action_stack).float())
    Q_state = main_network(tensor_state)
    Q_s_a = torch.sum(torch.mul(Q_state, tensor_action), dim=1)

    loss = torch.sum(torch.mul(target - Q_s_a, target - Q_s_a))
    loss.backward()
    for param in main_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

main_network = Model()
target_network = Model()
assign_parameter(target_network, main_network)
memory_size = 10000
memory = deque(maxlen=memory_size)

env = gym.make('CartPole-v0')
for episode in range(10000):
    state = env.reset()
    done = False
    global_step = 0
    state_list, action_list, next_state_list, reward_list, done_list = [], [], [], [], []

    while not done:
        global_step += 1
        action = choose_action(main_network, state)
        next_state, reward, done, _ = env.step(np.argmax(action))

        if done: reward = -1
        memory.append([state, next_state, action, reward, done])

        if len(memory) > 100:
            train(64, memory, target_network, main_network)
            if global_step % 10 == 0:
                assign_parameter(target_network, main_network)
        state = next_state

    print(episode, global_step, len(memory))