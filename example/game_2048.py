import numpy as np
import random

class Game2048:
    def __init__(self):
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self._add_new_tile()
        self._add_new_tile()

    def _add_new_tile(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j]==0]
        if empty_cells:
            row, col = random.choice(empty_cells)
            if random.random() < 0.9:
                self.board[row][col] = 2 
            else:
                self.board[row][col] = 4

    def _merge(self, row):
        non_zero = row[row != 0]
        new_row = np.zeros_like(row)
        i = 0
        skip = False
        for j in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                new_row[i] = non_zero[j] * 2
                self.score += new_row[i]
                skip = True
            else:
                new_row[i] = non_zero[j]
            i += 1
        return new_row

    def _move(self, board):
        new_board = np.zeros_like(board)
        for i in range(self.size):
            new_board[i, :] = self._merge(board[i, :])
        return new_board

    def step(self, action):
        original_board = np.copy(self.board)
        if action == 0:  # up
            self.board = self._move(self.board.T).T
        elif action == 1:  # down
            self.board = np.flipud(self._move(np.flipud(self.board).T).T)
        elif action == 2:  # left
            self.board = self._move(self.board)
        elif action == 3:  # right
            self.board = np.fliplr(self._move(np.fliplr(self.board)))
        
        if not np.array_equal(original_board, self.board):
            self._add_new_tile()

        done = not self._can_move()
        return self.board, self.score, done

    def _can_move(self):
        if np.any(self.board == 0):
            return True
        for i in range(self.size):
            for j in range(self.size):
                if (i + 1 < self.size and self.board[i][j] == self.board[i + 1][j]) or \
                   (j + 1 < self.size and self.board[i][j] == self.board[i][j + 1]):
                    return True
        return False

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self._add_new_tile()
        self._add_new_tile()
        return self.board

    def render(self, mode='human'):
        for row in self.board:
            print('\t'.join(map(str, row)))
        print(f"Score: {self.score}")
        print()
