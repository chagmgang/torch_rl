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
        self.layer_1 = nn.Linear(4, 24)
        self.layer_2 = nn.Linear(24, 24)
        self.layer_3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.softmax(self.layer_3(x))
        return x

def choose_action(net, state):
    tensor_state = Variable(torch.from_numpy(state).float())
    action_prob = net(tensor_state).data.numpy()
    action = np.random.choice(2, p=action_prob)
    action = np.eye(2)[action]
    return action

def train(optimizer, net, state, reward, action):
    optimizer.zero_grad()
    tensor_state = Variable(torch.from_numpy(state).float())
    tensor_reward = Variable(torch.from_numpy(reward).float())
    tensor_action = Variable(torch.from_numpy(action).float())
    action_selected_prob = torch.mul(net(tensor_state), tensor_action)
    action_selected_prob = torch.log(torch.sum(action_selected_prob, dim=1))
    log_pi_reward = torch.sum(torch.mul(tensor_reward, action_selected_prob))
    loss = -log_pi_reward
    loss.backward()
    optimizer.step()

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r

net = Model()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

env = gym.make('CartPole-v0')
for episode in range(10000):
    state = env.reset()
    done = False
    state_list, reward_list, action_list = [], [], []
    global_step = 0
    while not done:
        global_step += 1
        action = choose_action(net, state)
        next_state, reward, done, _ = env.step(np.argmax(action))

        if done: reward = -1

        state_list.append(state)
        reward_list.append(reward)
        action_list.append(action)

        state = next_state
    
    state_list = np.array(state_list)
    reward_list = np.array(reward_list)
    action_list = np.array(action_list)
    
    discounted_rewards = discount_rewards(reward_list)

    train(optimizer=optimizer, net=net, state=state_list, reward=discounted_rewards, action=action_list)
    print(episode, global_step)