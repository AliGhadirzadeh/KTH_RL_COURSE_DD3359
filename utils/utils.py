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

def sample_data(env, max_samples=512):
    sample_idx = 0
    x = torch.zeros(max_samples, 3) # state, action
    y = torch.zeros(max_samples, 2) # new_state
    while sample_idx < max_samples:
        done = False
        state = torch.tensor(env.reset(),dtype=torch.float)
        while done == False:
            action = torch.empty(1).normal_(mean=0,std=0.5)
            new_state, reward, done = env.step(action[0])
            if done == False:
                x[sample_idx, :] = torch.cat((state, action))
                y[sample_idx, :] = torch.tensor(new_state,dtype=torch.float)
                sample_idx += 1
                if sample_idx >= max_samples:
                    break
            state = torch.tensor(new_state, dtype=torch.float)
    return x, y
