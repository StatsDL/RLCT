import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, act_fn, dr):
        super(Actor, self).__init__()

        layers = []

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()

        # Add input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(p=dr))

        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already added the input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p=dr))

        # Add output layer
        layers.append(nn.Linear(hidden_dim, action_dim))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

        logstds_param = nn.Parameter(torch.full((action_dim,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, state):
        x = self.model(state)
        # means = torch.sigmoid(x)
        means = x
        stds = torch.clamp(self.logstds.exp(), 1e-3, 0.2)
        cov_mat = torch.diag_embed(stds)
        return torch.distributions.MultivariateNormal(means, cov_mat)




class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers, act_fn, dr):
        super(Critic, self).__init__()

        layers = []

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()

        # Add input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(p=dr))

        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already added the input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p=dr))

        # Add output layer
        layers.append(nn.Linear(hidden_dim, 1))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        x = self.model(state)
        return x