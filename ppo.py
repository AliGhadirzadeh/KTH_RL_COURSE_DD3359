# MIT License
# Copyright (c) 2020 Ali Ghadirzadeh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset,DataLoader

class RLDataset(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.returns = []
        self.advantages = []

    def upload_data(self, states, actions, returns, advantages):
        self.states = torch.clone(states)
        self.actions = torch.clone(actions)
        self.returns = torch.clone(returns)
        self.advantages = torch.clone(advantages)
    def add_data(self, states, actions, returns, advantages):
        self.states = torch.cat((self.states, states), 0)
        self.actions = torch.cat((self.actions, actions), 0)
        self.returns = torch.cat((self.returns, returns), 0)
        self.advantages = torch.cat((self.advantages, advantages), 0)

    def __getitem__(self, item):
        states = self.states[item]
        actions = self.actions[item]
        returns = self.returns[item]
        advantages = self.advantages[item]
        return states, actions, returns, advantages
    def __len__(self):
        return len(self.states)

class Policy(torch.nn.Module):
    def __init__(self, dim_state, dim_action, num_neurons = 64):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action

        self.actor_fc1 = torch.nn.Linear(dim_state, num_neurons)
        self.actor_fc2 = torch.nn.Linear(num_neurons, num_neurons)
        self.actor_fc3 = torch.nn.Linear(num_neurons, num_neurons)
        self.actor_mean = torch.nn.Linear(num_neurons, dim_action)
        self.actor_sigma = torch.nn.Parameter(torch.zeros(self.dim_action))

        self.critic_fc1 = torch.nn.Linear(dim_state, num_neurons)
        self.critic_fc2 = torch.nn.Linear(num_neurons, num_neurons)
        self.critic_fc3 = torch.nn.Linear(num_neurons, num_neurons)
        self.critic_value = torch.nn.Linear(num_neurons, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight, -1e-2, 1e-2)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # actor
        a = F.relu(self.actor_fc1(x))
        a = F.relu(self.actor_fc2(a))
        a = F.relu(self.actor_fc3(a))
        a_mean = self.actor_mean(a)
        a_sigma = torch.diag(F.softplus(self.actor_sigma))
        a_dist = MultivariateNormal(a_mean, a_sigma)
        # critic
        v = F.relu(self.critic_fc1(x))
        v = F.relu(self.critic_fc2(v))
        v = F.relu(self.critic_fc3(v))
        v = self.critic_value(v)
        return a_dist, v

class PPO(object):
    def __init__(self, config, env, device='cpu'):
        self.env = env
        self.device = device

        n_policy_neurons = config['policy']['num_neurons']

        self.policy = Policy(self.env.dim_state, self.env.dim_action, n_policy_neurons).to(self.device)
        self.old_policy = Policy(self.env.dim_state, self.env.dim_action, n_policy_neurons).to(self.device)

        lr = config['learning_params']['learning_rate']
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = config['learning_params']['gamma']
        self.clip = config['learning_params']['clip']
        self.batch_size = config['learning_params']['batch_size']
        self.ppo_batch_size = config['learning_params']['ppo_batch_size']
        self.max_iter_policy_training = config['learning_params']['max_iter_policy_training']

        self.actor_weight = config['loss_params']['actor_weight']
        self.critic_weight = config['loss_params']['critic_weight']
        self.entropy_weight = config['loss_params']['entropy_weight']

        self.dataset = RLDataset()

        self.replace_policy()
        self.old_policy.eval()
        self.policy.eval()

    def sample_batch(self,batch_size):
        max_step = 100
        reward_sum = 0

        for batch in range(batch_size):
            batch_states = torch.zeros(max_step, 2)
            batch_actions = torch.zeros(max_step, 1)
            batch_rewards = torch.zeros(max_step, 1)
            batch_values = torch.zeros(max_step, 1)
            done = False
            state = torch.tensor(self.env.reset(),dtype=torch.float)
            for step in range(max_step):
                action, value = self.run_policy(state)
                new_state, reward, done = self.env.step(action[0])
                reward_sum += reward
                value = value.detach()
                batch_states[step, :] = state
                batch_actions[step, :] = action
                batch_values[step, 0] = value
                batch_rewards[step, 0] = reward
                state = torch.tensor(new_state, dtype=torch.float)
                if done or step == (max_step-1):
                    batch_states = batch_states[:step+1, :]
                    batch_actions = batch_actions[:step+1, :]
                    batch_rewards = batch_rewards[:step+1, :]
                    batch_values = batch_values[:step+1, :]
                    # calc returns and advantages
                    batch_returns = torch.zeros(step+1)
                    batch_advantages = torch.zeros(step+1)
                    return_so_far = 0
                    for i in range(step+1):
                        return_so_far += batch_rewards[step-i]
                        batch_returns[step-i] = return_so_far
                        batch_advantages[step-i] = return_so_far - batch_values[step-i]
                        return_so_far *= self.gamma
                    break

            if batch == 0:
                self.dataset.upload_data(batch_states, batch_actions, batch_returns, batch_advantages)
            else:
                self.dataset.add_data(batch_states, batch_actions, batch_returns, batch_advantages)
        self.dataloader = DataLoader(self.dataset, batch_size=self.ppo_batch_size, shuffle=True)

        print( reward_sum)

    def run_policy(self, state, evaluation=False):
        dist, value = self.old_policy.forward(state)
        if evaluation:
            action = dist.mean
        else:
            action = dist.sample()
        return action, value

    def train_policy(self, n_training_iter):
        self.policy.train()
        for _ in range(n_training_iter):
            self.sample_batch(self.batch_size)
            for iter in range(self.max_iter_policy_training):
                actor_loss_sum = 0.0
                critic_loss_sum = 0.0
                advantage_sum = 0.0
                for states, actions, returns, advantages in self.dataloader:
                    states = states.to(self.device).float()
                    actions = actions.to(self.device).float()
                    returns = returns.to(self.device).float()
                    advantages = advantages.to(self.device).detach().float()

                    policy_dist, states_value = self.policy(states)
                    actions_logprob = policy_dist.log_prob(actions)

                    states_value = states_value.squeeze(-1)

                    old_policy_dist, _ = self.old_policy(states)
                    actions_logprob_old = old_policy_dist.log_prob(actions)

                    # critic loss
                    critic_loss = (states_value - returns.detach())**2
                    critic_loss = critic_loss.mean()

                    # actor loss
                    log_ratios = actions_logprob - actions_logprob_old.detach()
                    ratio = torch.exp(log_ratios)
                    clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)
                    objective = torch.min(ratio*advantages, clipped_ratio*advantages)
                    actor_loss = torch.mean(objective)

                    # entropy loss
                    entropy = policy_dist.entropy().mean()

                    # Total loss
                    loss = -self.actor_weight*actor_loss + self.critic_weight*critic_loss - self.entropy_weight*entropy

                    # update the network
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    actor_loss_sum += actor_loss.item()
                    critic_loss_sum += critic_loss.item()
                    advantage_sum += advantages.mean().item()

                actor_loss_sum = actor_loss_sum / len(self.dataloader)
                critic_loss_sum = critic_loss_sum / len(self.dataloader)
                advantage_sum = advantage_sum / len(self.dataloader)
                if iter % 100 == 99:
                    print('{:d} actor loss: {:.6e}, advantages: {:.6e}, critic loss: {:.6e}'.format(iter + 1, actor_loss_sum, advantage_sum, critic_loss_sum))

            self.replace_policy()

    def replace_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()
