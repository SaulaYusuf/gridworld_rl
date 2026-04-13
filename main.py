from env import GridWorld
from agent import QAgent
from experiments import EXPERIMENTS
from config import CONFIG
from visualize import plot_rewards, plot_heatmap

def state_index(pos):
    return pos[0] * CONFIG["grid_size"] + pos[1]


def train(alpha, epsilon):
    env = GridWorld()
    agent = QAgent(25, 4, alpha, CONFIG["gamma"], epsilon)

    rewards = []

    for ep in range(CONFIG["episodes"]):
        pos = env.reset()
        total = 0

        for _ in range(CONFIG["max_steps"]):
            state = state_index(pos)

            action = agent.choose_action(state)
            new_pos, reward, done = env.step(action)

            next_state = state_index(new_pos)
            agent.update(state, action, reward, next_state)

            pos = new_pos
            total += reward

            if done:
                break

        if len(rewards) >= 30:
            avg_reward = sum(rewards[-30:]) / 30
            if avg_reward > 10:
                print(f"Early stopping triggered at episode {ep} with avg reward {avg_reward}")
                break

        rewards.append(total)

    return rewards, agent.Q


def run_experiments():
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
    results, Q = run_experiments()

    plot_rewards(results)
    plot_heatmap(Q)