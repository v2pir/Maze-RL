import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)  # First hidden layer
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)     # Second hidden layer
        self.fc3 = nn.Linear(fc2_dims, n_actions)    # Output layer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.to(T.float32)))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions