import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).normal_(0.0, v)

# ----------------------------------------------------
# actor model, MLP
# ----------------------------------------------------
# 2 hidden layers, 400 units per layer, tanh output to bound outputs between -1 and 1
class actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 400)
        #self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 400)
        #self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, output_size)
        self.init_weights()
    
    def init_weights(self, init_w=10e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.normal_(0, 3e-3)

    def forward(self, state):
        out = self.fc1(state)
        #out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        #out = self.bn2(out)
        out = F.relu(out)
        action = F.tanh(self.fc3(out))
        return action


# ----------------------------------------------------
# critic model, MLP
# ----------------------------------------------------
# 2 hidden layers, 300 units per layer, outputs rewards therefore unbounded
# Action not to be included until 2nd layer of critic (from paper). Make sure 
# to formulate your critic.forward() accordingly

class critic(nn.Module):
    def __init__(self, state_size, action_size, output_size):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 300)
        #self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300 + action_size, 300)
        self.fc3 = nn.Linear(300, output_size)
        self.init_weights()
    
    def init_weights(self, init_w=10e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.normal_(0, 3e-4)
        
    def forward(self, state, action):
        out = self.fc1(state)
        #out = self.bn1(out)
        out = F.relu(out)
        out = F.relu(self.fc2(torch.cat([out,action],1)))
        qvalue = self.fc3(out)
        return qvalue
