import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import multivariate_normal



class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, log_probs, next_state, done):
        experience = (state, action, np.array([reward]), log_probs, next_state, done)
        # experience = (state, action, reward, log_probs, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        log_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, log_probs, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            log_batch.append(log_probs)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        log_batch = np.array(log_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)

        return state_batch, action_batch, log_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    
    
    
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
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, act_fn, dr):
        super(Critic, self).__init__()

        layers = []

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()

        # Add input layer
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
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

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.model(x)
        return x


class Value_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers, act_fn, dr):
        super(Value_Network, self).__init__()

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()

        layers = []

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