import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, x):
        return self.fc(x)


class Agent:
    def __init__(self, state_space, action_space, learning_rate=0.01):
        self.state_space = state_space
        self.action_space = action_space

        self.policy = PolicyNetwork(state_space, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.saved_log_probs = []
        self.rewards = []

    def get_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(logits=probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def get_action_probabilities(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.policy(state)
        return torch.softmax(probs, dim=1).detach().numpy().flatten()

    def update(self):
        R = 0
        policy_loss = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards = []

    def store_reward(self, reward):
        self.rewards.append(reward)