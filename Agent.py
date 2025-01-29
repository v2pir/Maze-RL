import random
from DQN import DeepQNetwork
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=7e-7):
        self.gamma = gamma # hyperparameter that determines weight of future rewards
        self.epsilon = epsilon # ratio of explore/exploit
        self.eps_min = eps_end # 
        self.eps_dec = eps_dec # rate of decrement
        self.lr = lr #learning rate
        self.action_space = [i for i in range(n_actions)] # available actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, 
                                   fc1_dims=64, fc2_dims=64)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), 
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    # choose action; epsilon-greedy
    def pick_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state).detach()  # Detach from computation graph
            action = T.argmax(actions).item()
            print("True")
        else:
            action = np.random.choice(self.action_space)
            print("False")

        return action

    # DQL algorithm
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.float32)

        state_batch = T.tensor(self.state_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.float32).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        q_next = self.Q_eval.forward(new_state_batch)

        q_next[terminal_batch.to(T.long)] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = max(self.eps_min, self.epsilon * 0.99995)  # Exponential decay