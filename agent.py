import numpy as np
import random

class QAgent:
    def __init__(self, states, actions, alpha, gamma, epsilon):
        self.Q = np.zeros((states, actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )