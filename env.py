class GridWorld:
    def __init__(self):
        self.size = 5
        self.start = (1, 0)
        self.goal = (4, 4)

        # from spec
        self.jump_from = (1, 3)
        self.jump_to = (3, 3)

        # self.obstacles = [(2,2), (3,1)]  # allowed by spec
        self.obstacles = [(2, 2), (2, 3), (2, 4), (3, 2)]

        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, 1),
            3: (0, -1)
        }

        move = moves[action]
        new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])

        # boundary or obstacle
        if not self.valid(new_pos):
            return self.pos, -1, False

        # jump
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
        r, c = pos
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return False
        if pos in self.obstacles:
            return False
        return True