
from env import GridWorld
from agent import QAgent
from experiments import EXPERIMENTS
from config import CONFIG
from visualize import plot_rewards, plot_heatmap

def state_index(pos):
    """
    (b) State representation: Converts a (row, col) tuple to a unique state index for the Q-table.
    The grid is 0-indexed: (row, col) in [0,4].
    """
    return pos[0] * CONFIG["grid_size"] + pos[1]

def train(alpha, epsilon):
    """
    (c, d, e) Trains a Q-learning agent in the GridWorld environment.
    Args:
        alpha (float): Learning rate (0 < alpha <= 1)
        epsilon (float): Epsilon for epsilon-greedy exploration
    Returns:
        rewards (list): Episode rewards
        Q (np.ndarray): Learned Q-table
    Implements:
        - Q-table creation (c)
        - Epsilon-greedy exploration (d)
        - Early stopping (e)
    """
    env = GridWorld()
    # 25 states (5x5), 4 actions (N,S,E,W)
    agent = QAgent(25, 4, alpha, CONFIG["gamma"], epsilon)

    rewards = []

    for ep in range(CONFIG["episodes"]):
        pos = env.reset()
        total = 0

        for _ in range(CONFIG["max_steps"]):
            state = state_index(pos)

            # (a) Exploration vs Exploitation: agent.choose_action uses epsilon-greedy
            action = agent.choose_action(state)
            new_pos, reward, done = env.step(action)

            next_state = state_index(new_pos)
            agent.update(state, action, reward, next_state)

            pos = new_pos
            total += reward

            if done:
                break

        # (e) Early stopping: stop if avg reward > 10 over last 30 episodes
        if len(rewards) >= 30:
            avg_reward = sum(rewards[-30:]) / 30
            if avg_reward > 10:
                print(f"Early stopping triggered at episode {ep} with avg reward {avg_reward}")
                break

        rewards.append(total)

    return rewards, agent.Q

def run_experiments():
    """
    Runs experiments for different alpha/epsilon values (see EXPERIMENTS list).
    Returns:
        results (dict): Mapping label -> reward list
        last_Q (np.ndarray): Q-table from last experiment (for visualization)
    """
    results = {}
    last_Q = None

    for exp in EXPERIMENTS:
        print(f"Running: {exp}")

        rewards, Q = train(exp["alpha"], exp["epsilon"])
        label = f"a={exp['alpha']} e={exp['epsilon']}"

        results[label] = rewards
        last_Q = Q  # keep last one for heatmap

    return results, last_Q

if __name__ == "__main__":
    """
    (f) Visualize results: Plots smoothed rewards and state value heatmap.
    """
    results, Q = run_experiments()

    plot_rewards(results)
    plot_heatmap(Q)