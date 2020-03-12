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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class TorchDataset(Dataset):
    def __init__(self, data_in, data_out):
        self.data_in = torch.clone(data_in)
        self.data_out = torch.clone(data_out)
    def __len__(self):
        return len(self.data_in)
    def __getitem__(self, idx):
        return self.data_in[idx], self.data_out[idx]

class FullyConnectedNetwork(nn.Module):
    def __init__(self, dim_input, dim_output, num_neurons = 64):
        super(FullyConnectedNetwork,self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.fc1 = nn.Linear(dim_input, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, num_neurons)
        self.fc4 = nn.Linear(num_neurons, num_neurons)
        self.fc5 = nn.Linear(num_neurons, dim_output)

    def forward(self,x):
        x=x.view(-1,self.dim_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class SupervisedLearner(nn.Module):
    def __init__(self, dim_input, dim_output, num_neurons=64, device='cpu'):
        super(SupervisedLearner,self).__init__()
        self.device = device
        self.net = FullyConnectedNetwork(dim_input, dim_output,num_neurons).to(self.device)
        self.init_param()

    def train_model(self, train_data_loader, test_data_loader, num_epoch=10000, learning_rate=0.001):
        self.train()
        optimizer = optim.Adam(self.net.parameters(), learning_rate)
        for epoch in range(num_epoch):
            sum_loss = 0.0
            for x, y in train_data_loader:
                optimizer.zero_grad()
                yhat = self.net.forward(x)
                loss = F.mse_loss(y, yhat)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()/(x.size(0)*x.size(1))
            avg_loss = sum_loss / len(train_data_loader)
            if epoch % 100 == 0 or epoch == (num_epoch-1):
                train_loss = math.sqrt(avg_loss)
                test_loss = self.evaluate_model(test_data_loader)
                print('{:d} train loss: {:.6e} \t test loss: {:.6e}'.format(epoch, train_loss, test_loss))
                self.train()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def evaluate_model(self, data_loader):
        self.eval()
        sum_loss = 0.0
        for x,y in data_loader:
            yhat = self.net.forward(x)
            loss = F.mse_loss(y, yhat)
            sum_loss += loss.item()/(x.size(0)*x.size(1))
        avg_loss = sum_loss / (len(data_loader))
        return math.sqrt(avg_loss)
