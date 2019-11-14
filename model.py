import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, leakiness=0.01, kaiming=False, all_batchnorm=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units) 
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.leakiness = leakiness
        self.kaiming = kaiming
        self.all_bn = all_batchnorm
        self.reset_parameters()

    def reset_parameters(self):
        if self.leakiness > 0 and self.kaiming:
            nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leakiness, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leakiness, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)        
        else:    
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.all_bn: x = self.bn0(state) 
        else:           x = state    
        if self.leakiness > 0:
            x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=self.leakiness)
            if self.all_bn: x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=self.leakiness)
            else:           x = F.leaky_relu(self.fc2(x), negative_slope=self.leakiness)
        else:    
            x = F.relu(self.bn1(self.fc1(x)))
            if self.all_bn: x = F.relu(self.bn2(self.fc2(x)))
            else:           x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300, leakiness=0.01, kaiming=False, all_batchnorm=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.leakiness = leakiness
        self.kaiming = kaiming
        self.all_bn = all_batchnorm
        self.reset_parameters()

    def reset_parameters(self):
        if self.leakiness > 0 and self.kaiming:
            nn.init.kaiming_normal_(self.fcs1.weight.data, a=self.leakiness, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leakiness, mode='fan_in', nonlinearity='leaky_relu') 
            nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        else:    
            self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
   
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.all_bn: xs = self.bn0(state) 
        else:           xs = state    
        if self.leakiness > 0:
            xs = F.leaky_relu(self.bn1(self.fcs1(xs)), negative_slope=self.leakiness)
            x = torch.cat((xs, action), dim=1)
            if self.all_bn: x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=self.leakiness)
            else:           x = F.leaky_relu(self.fc2(x), negative_slope=self.leakiness)  
        else:
            xs = F.relu(self.bn1(self.fcs1(xs)))
            x = torch.cat((xs, action), dim=1)
            if self.all_bn: x = F.relu(self.bn2(self.fc2(x)))
            else:           x = F.relu(self.fc2(x))

        return self.fc3(x)
