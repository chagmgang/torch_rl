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
from rnd import RND, rnd_train, predict_intrincit_reward
from ppo import Model, get_gaes, assign_parameter, choose_action, train

main_RND = RND()
target_RND = RND()
Policy = Model()
Old_Policy = Model()
assign_parameter(Old_Policy, Policy)
optimizer = torch.optim.Adam(Policy.parameters(), lr=0.001)
env = gym.make('CartPole-v0')
batch_size = 32
train_epoch = 4

for episodes in range(100):
    done = False
    state = env.reset()
    observations, actions, v_preds, rewards, next_observations = [], [], [], [], []
    global_step = 0
    while not done:
        global_step += 1
        action, value = choose_action(Policy, state)
        next_state, reward, done, _ = env.step(np.argmax(action))
        extrincit_reward = reward
        if done: extrincit_reward = -1
        intrincit_reward = predict_intrincit_reward(target_RND, main_RND, next_state)
        reward = extrincit_reward + intrincit_reward

        observations.append(state)
        actions.append(action)
        v_preds.append(value)
        rewards.append(extrincit_reward)
        next_observations.append(next_state)

        state = next_state

    v_preds_next = v_preds[1:] + [0]
    gaes = get_gaes(rewards, v_preds, v_preds_next)
    observations = np.array(observations)
    actions = np.array(actions).astype(dtype=np.int32)
    rewards = np.array(rewards).astype(dtype=np.float32)
    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
    gaes = np.array(gaes).astype(dtype=np.float32)
    next_observations = np.array(next_observations)

    assign_parameter(Old_Policy, Policy)
    train(batch_size, train_epoch, optimizer, Old_Policy, Policy,
          observations, actions, rewards, v_preds_next, gaes)

    rnd_train(target_RND, main_RND, next_observations)

    print(episodes, global_step)