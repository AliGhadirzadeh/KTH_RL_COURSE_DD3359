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

class ValueIteration():
    def __init__(self, env, gamma = 0.9):
        self.gamma = gamma
        self.env = env
        self.V = torch.zeros(self.env.n_states, dtype=torch.float)

    def update(self, n_iteration):
        states = self.env.get_states()
        actions = self.env.get_actions()
        for _ in range(n_iteration):
            for state in states:
                state_idx = self.env.get_state_index(state)
                v_max = -1e5
                for action in actions:
                    new_state, reward = self.env.step(state, action)
                    v = reward + self.gamma * self.get_value(new_state)
                    v_max = v if v > v_max else v_max
                self.V[state_idx] = v_max

    def get_value(self, state):
        state_index = self.env.get_state_index(state)
        return self.V[state_index]
