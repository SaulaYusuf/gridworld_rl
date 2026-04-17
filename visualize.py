import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window=10):
    """
    Computes moving average for smoothing reward curves.
    Args:
        data (list): List of rewards
        window (int): Smoothing window size
    Returns:
        list: Smoothed values
    """
    return [sum(data[i:i+window]) / window for i in range(len(data)-window)]

def plot_rewards(results):
    """
    (f) Plots smoothed episode rewards for each experiment.
    Args:
        results (dict): label -> reward list
    """
    for label, rewards in results.items():
        smoothed = moving_average(rewards)
        plt.plot(smoothed, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title("Training Performance (Smoothed)")
    plt.legend()
    plt.show()

def plot_heatmap(Q):
    """
    (f) Visualizes state values as a heatmap (max Q per state).
    Args:
        Q (np.ndarray): Q-table
    """
    values = np.max(Q, axis=1).reshape((5, 5))

    fig, ax = plt.subplots()
    cax = ax.imshow(values, cmap='viridis')
    fig.colorbar(cax)

    # Annotate each cell with value
    for i in range(5):
        for j in range(5):
            text_color = "white" if values[i, j] < (np.max(values) / 2) else "black"
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", color=text_color)

    ax.set_title("State Value Heatmap")
    plt.show()