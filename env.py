class GridWorld:
    """
    (b) GridWorld environment for RL agent.
    - 5x5 grid, 0-indexed (row, col)
    - Start: (1,0) == [2,1] in brief
    - Goal: (4,4) == [5,5] in brief
    - Jump: (1,3) -> (3,3) == [2,4] -> [4,4] in brief
    - Obstacles: L-shaped wall (see self.obstacles)
    """
    def __init__(self):
        self.size = 5
        self.start = (1, 0)
        self.goal = (4, 4)

        # from spec: jump from [2,4] to [4,4] (0-indexed: (1,3) -> (3,3))
        self.jump_from = (1, 3)
        self.jump_to = (3, 3)

        # L-shaped wall: obstacles at (2,2), (2,3), (2,4), (3,2)
        self.obstacles = [(2, 2), (2, 3), (2, 4), (3, 2)]

        self.reset()

    def reset(self):
        """
        Resets agent to starting position.
        Returns:
            pos (tuple): Start position (row, col)
        """
        self.pos = self.start
        return self.pos

    def step(self, action):
        """
        (b, c) Executes action in environment.
        Args:
            action (int): 0=N, 1=S, 2=E, 3=W
        Returns:
            new_pos (tuple): New position
            reward (int): Reward for action
            done (bool): True if terminal state
        """
        moves = {
            0: (-1, 0),  # North
            1: (1, 0),   # South
            2: (0, 1),   # East
            3: (0, -1)   # West
        }

        move = moves[action]
        new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])

        # boundary or obstacle
        if not self.valid(new_pos):
            return self.pos, -1, False

        # jump pad
        if new_pos == self.jump_from:
            self.pos = self.jump_to
            return self.pos, 5, False

        # goal
        if new_pos == self.goal:
            self.pos = new_pos
            return new_pos, 10, True

        self.pos = new_pos
        return new_pos, -1, False

    def valid(self, pos):
        """
        Checks if position is within grid and not an obstacle.
        Args:
            pos (tuple): (row, col)
        Returns:
            bool: True if valid
        """
        r, c = pos
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return False
        if pos in self.obstacles:
            return False
        return True