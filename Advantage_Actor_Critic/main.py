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

def choose_action(net, state):
    tensor_state = Variable(torch.from_numpy(state).float())
    action, value = net(tensor_state)
    action = action.data.numpy()
    value = value.data.numpy()[0]
    action = np.random.choice(2, p=action)
    action = np.eye(2)[action]
    return action, value

def train(optimizer, net, state, next_state, reward, action, done):
    optimizer.zero_grad()
    tensor_state = Variable(torch.from_numpy(state).float())
    tensor_next_state = Variable(torch.from_numpy(next_state).float())
    tensor_reward = Variable(torch.from_numpy(reward).float())
    tensor_action = Variable(torch.from_numpy(action).float())
    tensor_done = Variable(torch.from_numpy(done).float())

    next_action, next_value = net(tensor_next_state)
    next_value = next_value.view(-1)

    action_prob, value = net(tensor_state)
    value = value.view(-1)

    advantage = tensor_reward + (1-tensor_done) * 0.99 * next_value - value
    pi_s_a = torch.sum(torch.mul(action_prob, tensor_action), dim=1)
    pi_s_a = torch.clamp(pi_s_a, min=1e-10, max=1.0)
    log_pi_adv = torch.mul(torch.log(pi_s_a), advantage)
     
    value_loss = torch.mul(advantage, advantage)

    loss = -torch.sum(log_pi_adv - value_loss)
    loss.backward()
    for param in net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r

a2c = Model()
optimizer = torch.optim.Adam(a2c.parameters(), lr=0.001)
env = gym.make('CartPole-v0')

for episode in range(1000):
    state = env.reset()
    done = False
    state_list, next_state_list, reward_list, action_list, done_list = [], [], [], [], []
    global_step = 0
    while not done:
        global_step += 1

        action, value = choose_action(a2c, state)
        next_state, reward, done, _ = env.step(np.argmax(action))
        if done: reward = -1            

        state_list.append(state)
        next_state_list.append(next_state)
        reward_list.append(reward)
        action_list.append(action)
        done_list.append(int(done))

        state = next_state
    
    array_state_list = np.array(state_list)
    array_next_state_list = np.array(next_state_list)
    array_reward_list = np.array(reward_list)
    array_action_list = np.array(action_list)
    array_done_list = np.array(done_list)

    discounted_rewards = discount_rewards(array_reward_list)

    train(optimizer, a2c, array_state_list, array_next_state_list,
        discounted_rewards, array_action_list, array_done_list)
    print(episode, global_step)