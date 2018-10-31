import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Variable
import tensorflow as tf
from collections import deque

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(4, 24)
        self.layer_2 = nn.Linear(24, 24)
        self.layer_3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
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
    minibatch = random.sample(memory, self.batch_size)
    state_stack = [mini[0] for mini in minibatch]
    next_state_stack = [mini[1] for mini in minibatch]
    action_stack = [mini[2] for mini in minibatch]
    reward_stack = [mini[3] for mini in minibatch]
    done_stack = [mini[4] for mini in minibatch]
    done_stack = [int(i) for i in done_stack]

main_network = Model()
target_network = Model()
assign_parameter(target_network, main_network)
memory_size = 10000
memory = deque(maxlen=memory_size)

env = gym.make('CartPole-v0')
for episode in range(1):
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

        state = next_state

