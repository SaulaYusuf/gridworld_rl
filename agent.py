import numpy as np
import random

class QAgent:
    """
    (c, d) Q-learning agent for tabular RL in GridWorld.
    - Q-table: Stores state-action values (c)
    - Epsilon-greedy exploration (d)
    """
    def __init__(self, states, actions, alpha, gamma, epsilon):
        """
        Args:
            states (int): Number of states
            actions (int): Number of actions
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
        """
        self.Q = np.zeros((states, actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        """
        (a, d) Epsilon-greedy action selection.
        With probability epsilon, choose random action (exploration).
        Otherwise, choose best action (exploitation).
        Args:
            state (int): Current state index
        Returns:
            action (int): Action index (0=N, 1=S, 2=E, 3=W)
        """
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        """
        (c) Q-learning update rule (off-policy TD control).
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state
        """
        best_next = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )