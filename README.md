# Q-Learning Grid World Agent (COM762 - Coursework 2)

## Overview
This repository contains my modular Python implementation of a Q-learning agent designed to solve the custom 5x5 grid world environment defined in the coursework brief. 

Rather than writing a monolithic, single-file script, I chose to architect the codebase using **Separation of Concerns (SoC)**. By splitting my logic into distinct modules (Environment, Agent, Configuration, Visualization, and Execution), I ensured the code remains highly readable, debuggable, and demonstrates professional software engineering standards alongside the core mathematical Reinforcement Learning principles.

## File Architecture & Coursework Mapping

Here is the exact breakdown of my repository structure and how I designed each file to directly address the assignment brief.

### 1. `env.py` (The Environment Logic)
* **What it is:** This file contains my `GridWorld` class, which handles all the physical rules of the board, tracks the agent's position, and dispenses rewards.
* **Why I separated it:** I wanted to isolate the "game logic" from the "learning logic." The environment shouldn't know *how* the agent learns; it should only tell the agent what happens when it takes an action.
* **Relation to Brief:** Addresses **Task (b)** and the core rules (1-6).
    * I defined the 5x5 boundary (`self.size = 5`).
    * I set the start cell at `[1, 0]` (which is the Python 0-indexed equivalent for the brief's `[2,1]`).
    * I configured the `self.obstacles` list to accurately block the black cells using Python's grid coordinates.
    * I implemented the `step()` function to dispense the +10 terminal reward, the +5 jump reward (with a continuous episode flag), and the -1 standard movement penalty.

### 2. `agent.py` (The Mathematical Brain)
* **What it is:** This file contains my `QAgent` class. I designed this to handle the Q-table, the action selection logic, and the mathematical updating of state-action values.
* **Why I separated it:** I wanted to cleanly encapsulate the Reinforcement Learning mathematics. By doing this, the environment could theoretically be swapped out for a completely different game, and my agent would still function.
* **Relation to Brief:** Addresses **Tasks (c) and (d)** and rule (7).
    * I initialized the value table (Q-table) as a matrix of zeros for the 25 discrete states and 4 possible actions.
    * I implemented the `choose_action()` method using the required epsilon-greedy exploration strategy.
    * I implemented the `update()` method using the exact iteration function provided in the brief: $V(S_{t})\leftarrow V(S_{t})+\alpha[V(S_{t+1})-V(S_{t})]$.

### 3. `experiments.py` & `config.py` (The Variables)
* **What it is:** I used `config.py` to hold static project variables (grid size, discount factor $\gamma$, max episodes). I used `experiments.py` to hold a list of dictionaries defining my different testing runs with varying $\alpha$ (learning rate) and $\epsilon$ (exploration rate).
* **Why I separated it:** Hardcoding variables inside the main logic is poor practice. Extracting them allowed me to rapidly test different hyperparameters without digging through core algorithm files to change numbers.
* **Relation to Brief:** Directly addresses the **Task (c)** requirement to "set the learning rate to be 1 and various values in between [0, 1] to show running results."

### 4. `main.py` (The Execution Engine)
* **What it is:** This serves as the entry point of my program. It acts as the controller, importing the Environment and Agent, running the training loops, logging the data, and triggering the visualizer.
* **Relation to Brief:** Addresses **Task (e)**.
    * I implemented the loop to "train for 100 episodes".
    * I used a `while` loop that only breaks on `done=True`, satisfying the "unlimited paths" requirement per episode.
    * I built custom **early stopping logic** that safely breaks the training loop if my agent successfully achieves an average cumulative reward > 10 over 30 consecutive episodes.

### 5. `visualize.py` (The Output Generator)
* **What it is:** I created this utility file to contain `matplotlib` functions for generating graphs from the training data.
* **Why I separated it:** Plotting logic requires heavy UI/math libraries that clutter up terminal execution files, so I abstracted it away from the core execution loop.
* **Relation to Brief:** Addresses **Task (f)**.
    * My code iterates over the finalized Q-table to extract the maximum values for each state.
    * It renders a `matplotlib` color heatmap with numerical overlays to clearly "visualise the state values in each of the grid cells with the board layout".

### 6. `pyproject.toml` & `uv.lock` (Dependency Management)
* **What it is:** Modern Python package management files utilized by the `uv` tool.
* **Why I included it:** To ensure proper management of my virtual environment and to strictly lock my project dependencies (`numpy` and `matplotlib`) for reproducible execution across different machines.

## How to Run My Code
1. Ensure Python 3.13+ is installed on your system.
2. Install dependencies using `uv sync` (or standard `pip install numpy matplotlib`).
3. Execute the simulation by running: `python main.py`
4. The terminal will output my early-stopping metrics, and two matplotlib visualization windows (training performance and the state value heatmap) will appear upon completion.