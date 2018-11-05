import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import gym, copy

def ortho_weights(shape, scale=1.0):
    shape = tuple(shape)
    
    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError
    
    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

if len(shape) == 2:
    return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.actor_layer_1 = nn.Linear(4, 60)
        self.actor_layer_2 = nn.Linear(60, 60)
        self.actor_layer_3 = nn.Linear(60, 2)
        self.critic_layer_1 = nn.Linear(4, 60)
        self.critic_layer_2 = nn.Linear(60, 60)
        self.critic_layer_3 = nn.Linear(60, 1)
        
        self.actor_layer_3.weight.data = ortho_weights(self.actor_layer_3.weight.size(), scale=1.0)
        self.critic_layer_3.weight.data = ortho_weights(self.critic_layer_3.weight.size(), scale=1.0)
    
    
    def forward(self, x):
        actor = F.relu(self.actor_layer_1(x))
        actor = F.relu(self.actor_layer_2(actor))
        actor = self.actor_layer_3(actor)
        
        critic = F.relu(self.critic_layer_1(x))
        critic = F.relu(self.critic_layer_2(critic))
        critic = self.critic_layer_3(critic)
        
        return actor, critic

def choose_action(net, state):
    tensor_state = Variable(torch.from_numpy(state).float())
    action, value = net(tensor_state)
    action = F.softmax(action)
    action = action.data.numpy()
    prob = np.clip(action, 1e-10, 1.0)
    length_action = len(action)
    value = value.data.numpy()[0]
    #action = np.argmax(action)
    action = np.random.choice(length_action, p=action)
    action = np.eye(length_action)[action]
    entropy = sum(-action * np.log(prob))
    return action, value, entropy

def assign_parameter(target_network, main_network):
    target_network.load_state_dict(main_network.state_dict())

def get_next_vpreds(v_preds):
    return v_preds[1:] + [0]

def get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + 0.99 * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + 0.99 * gaes[t + 1]
    return gaes

def train(Old_Policy, Policy, observations, actions, rewards, v_preds_next, gaes):
    tensor_observations = Variable(torch.from_numpy(observations)).float()
    tensor_actions = Variable(torch.from_numpy(actions)).float()
    tensor_rewards = Variable(torch.from_numpy(rewards)).float()
    tensor_v_preds_next = Variable(torch.from_numpy(v_preds_next)).float()
    tensor_gaes = Variable(torch.from_numpy(gaes)).float()
    
    pi, value = Policy(tensor_observations)
    value = value.squeeze(1)
    pi_old, _ = Old_Policy(tensor_observations)
    
    prob = F.softmax(pi)
    log_prob = F.log_softmax(pi)
    action_prob = torch.sum(torch.mul(prob, tensor_actions), dim=1)
    
    prob_old = F.softmax(pi_old)
    action_prob_old = torch.sum(torch.mul(prob_old, tensor_actions), dim=1)
    
    ratio = action_prob / (action_prob_old + 1e-10)
    advantage = (tensor_gaes - tensor_gaes.mean()) / (tensor_gaes.std() + 1e-5)
    
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, min=1. - 0.2, max=1. + 0.2) * advantage
    
    policy_loss = -torch.min(surr1, surr2).mean()
    value_diff = tensor_v_preds_next * 0.99 + tensor_rewards - value
    value_loss = torch.mul(value_diff, value_diff).mean()
    entropy_loss = (prob * log_prob).sum(1).mean()
    
    total_loss = policy_loss + value_loss + entropy_loss * 0.001
    
    optimizer = torch.optim.Adam(Policy.parameters(), lr=0.005)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()



Policy = Model()
Old_Policy = Model()
env = gym.make('CartPole-v1')
batch_size = 64
train_epoch = 2

for episodes in range(10000):
    done = False
    state = env.reset()
    observations, actions, rewards, v_preds = [], [], [], []
    global_step = 0
    while not done:
        global_step += 1
        action, value, entropy = choose_action(Policy, state)
        next_state, reward, done, _ = env.step(np.argmax(action))
        if done:
            if global_step == 500:
                reward = 1
            else:
                reward = -1
        else: reward = 0
        
        observations.append(state)
        v_preds.append(value)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    v_preds_next = get_next_vpreds(v_preds)
    gaes = get_gaes(rewards, v_preds, v_preds_next)

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    v_preds_next = np.array(v_preds_next)
    gaes = np.array(gaes)

inp = [observations, actions, rewards, v_preds_next, gaes]
train(observations=observations,
      actions=actions,
      rewards=rewards,
      v_preds_next=v_preds_next,
      gaes=gaes,
      Old_Policy=Old_Policy,
      Policy=Policy)
assign_parameter(Old_Policy, Policy)
print(episodes, global_step)
