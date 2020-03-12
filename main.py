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

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from envs.grid_world import GridWorld
from envs.mountain_car import MountainCar
from envs.inverted_pendulum import InvertedPendulum

from dynamic_programming import ValueIteration
from supervised_learner import SupervisedLearner, TorchDataset
from ppo import PPO

from utils.utils import sample_data

from importlib.machinery import SourceFileLoader




if False:
    env = GridWorld(10)
    vi = ValueIteration(env, 0.9)
    vi.update(5)
    env.print_value_function(vi.V)


if False:
    # train dynamic forward model

    # set the environment
    env = InvertedPendulum()

    # collect train and test data samples
    data_in, data_out = sample_data(env, 512)
    train_dataset = TorchDataset(data_in, data_out)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    data_in, data_out = sample_data(env, 64)
    test_dataset = TorchDataset(data_in, data_out)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # train the model
    agent = SupervisedLearner(3,2)
    agent.train_model(train_dataloader, test_dataloader, 1000)

    # save the model
    torch.save(agent.net.state_dict(), 'dynamic_model.mdl')

    # load the model
    state_dict = torch.load('dynamic_model.mdl')
    agent.net.load_state_dict(state_dict)
    agent.net.eval()

    #evaluate the trained model
    data_in, data_out = sample_data(env, 4)

    for i in range(4):
        yhat = agent.net.forward(data_in[i])
        print ('input:' , data_in[i].detach().numpy(),
               'output:', data_out[i].detach().numpy(),
               'predicted:' , yhat.detach().numpy())


if True:
    # train ppo

    # set the environment
    env = InvertedPendulum()

    # set ppo
    config = SourceFileLoader("config", 'ppo_config.py').load_module().config
    agent = PPO( config['ppo'] , env)
    agent.train_policy(20)
