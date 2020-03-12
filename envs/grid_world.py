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

class GridWorld():
    def __init__(self, width = 10, height = 10):
        self.width = width
        self.height = height
        self.n_states = width * height
        self.state_max = torch.tensor([width-1, height-1], dtype=torch.int8)
        self.state_min = torch.zeros(1,2, dtype=torch.int8)
        self.goal_state = torch.tensor([width-1, height-1], dtype=torch.int8)

    def step(self, state, action):
        state += action
        state = torch.min(state, self.state_max)
        state = torch.max(state, self.state_min)
        reward=1 if torch.sum(torch.abs(state - self.goal_state)) == 0 else -1
        return state, reward

    def get_states(self):
        states = torch.zeros(self.width*self.height, 2, dtype=torch.int8)
        for i in range(self.width):
            for j in range(self.height):
                states[i*self.width + j, :] = torch.tensor([i,j], dtype=torch.int8)
        return states

    def get_actions(self):
        actions = torch.zeros(4, 2, dtype=torch.int8)
        actions[0,:] = torch.tensor([0,1],dtype=torch.int8)
        actions[1,:] = torch.tensor([0,-1],dtype=torch.int8)
        actions[2,:] = torch.tensor([1,0],dtype=torch.int8)
        actions[3,:] = torch.tensor([-1,0],dtype=torch.int8)
        return actions

    def get_state_index(self, state):
        state = state.squeeze()
        return state[0]*self.width + state[1]

    def print_value_function(self, V):
        for j in range(self.height):
            str = ''
            for i in range(self.width):
                idx = i*self.width + j
                str = '{} {:+02.2f}'.format(str, V[idx])
            print (str)
