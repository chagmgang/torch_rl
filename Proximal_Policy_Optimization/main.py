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
from tensorboardX import SummaryWriter

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

def assign_parameter(target_network, main_network):
    target_network.load_state_dict(main_network.state_dict())

def get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + 0.99 * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + 0.99 * gaes[t + 1]
    return gaes

def train(batch_size, train_epoch, optimizer, Old_Policy, Policy, observations, actions, rewards, v_preds_next, gaes):
    inp = [observations, actions, rewards, v_preds_next, gaes]
    for i in range(train_epoch):
        optimizer.zero_grad()
        sample_indices = np.random.randint(low=0, high=observations.shape[0], size=batch_size)
        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]

        tensor_observations = Variable(torch.from_numpy(sampled_inp[0]).float())
        tensor_actions = Variable(torch.from_numpy(sampled_inp[1]).float())
        tensor_rewards = Variable(torch.from_numpy(sampled_inp[2]).float())
        tensor_v_preds_next = Variable(torch.from_numpy(sampled_inp[3]).float())
        tensor_gaes = Variable(torch.from_numpy(sampled_inp[4]).float())

        tensor_action_probs, tensor_value = Policy(tensor_observations)
        tensor_prev_action_probs, tensor_prev_value = Old_Policy(tensor_observations)
        tensor_action_probs = torch.clamp(tensor_action_probs, min=1e-10, max=1.0)
        tensor_prev_action_probs = torch.clamp(tensor_prev_action_probs, min=1e-10, max=1.0)

        tensor_action_probs = torch.log(torch.sum(torch.mul(tensor_action_probs, tensor_actions), dim=1))
        tensor_prev_action_probs = torch.log(torch.sum(torch.mul(tensor_prev_action_probs, tensor_actions), dim=1))

        action_prob_ratio = torch.exp(torch.sub(tensor_action_probs, tensor_prev_action_probs))
        action_prob_ratio_adv = torch.mul(tensor_gaes, action_prob_ratio)
        clipped = torch.clamp(action_prob_ratio_adv, min=1-0.2, max=1+0.2)

        loss_action = torch.min(action_prob_ratio_adv, clipped)
        loss_action = torch.sum(loss_action)

        loss_value = tensor_rewards + 0.99 * tensor_v_preds_next - tensor_value.view(-1)
        loss_value = torch.mul(loss_value, loss_value)
        loss_value = torch.sum(loss_value)

        loss = loss_action - loss_value
        loss = -loss
        loss.backward()
        optimizer.step()

writer = SummaryWriter()
Policy = Model()
Old_Policy = Model()
assign_parameter(Old_Policy, Policy)
optimizer = torch.optim.Adam(Policy.parameters(), lr=0.005)
env = gym.make('CartPole-v1')
batch_size = 64
train_epoch = 1

for episodes in range(1000):
    done = False
    state = env.reset()
    observations, actions, v_preds, rewards, dones = [], [], [], [], []
    global_step = 0
    while not done:
        if episodes % 20 == 0:
            env.render()
        global_step += 1
        action, value = choose_action(Policy, state)
        next_state, reward, done, _ = env.step(np.argmax(action))

        if done:
            if global_step == 500:
                reward = 1
            else:
                reward = -1
        else: reward = 0

        observations.append(state)
        actions.append(action)
        v_preds.append(value)
        rewards.append(reward)
        dones.append(done)

        state = next_state
    
    v_preds_next = v_preds[1:] + [0]
    gaes = get_gaes(rewards, v_preds, v_preds_next)
    observations_array = np.array(observations)
    actions_array = np.array(actions).astype(dtype=np.int32)
    rewards_array = np.array(rewards).astype(dtype=np.float32)
    v_preds_next_array = np.array(v_preds_next).astype(dtype=np.float32)
    gaes = np.array(gaes).astype(dtype=np.float32)
    adv = (gaes - gaes.mean()) / (gaes.std() + 1e-7)


    train(batch_size, train_epoch, optimizer, Old_Policy, Policy,
        observations_array, actions_array, rewards_array, v_preds_next_array, gaes)
    assign_parameter(Old_Policy, Policy)
    writer.add_scalar('data/reward', global_step, episodes)
    print(episodes, global_step)