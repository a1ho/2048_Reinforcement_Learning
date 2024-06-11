import numpy as np
import random
import time

class Game2048:
    def __init__(self):
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.n_moves = 0
        self.reward = 0
        self.max_tile = 2
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
                self.reward += new_row[i]
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
        self.n_moves += 1
        action = int(action)

        original_board = np.copy(self.board)
        actions = set([1,2,3])
        found = False
        while not found:

            self.reward = 0

            if action == 0:  # Up
                self.board = self._move(self.board.T).T
                    
            elif action == 1:  # Down
                self.board = np.flipud(self._move(np.flipud(self.board).T).T)

            elif action == 2:  # Left
                self.board = self._move(self.board)
            
            elif action == 3:  # Right
                self.board = np.fliplr(self._move(np.fliplr(self.board)))

            if np.array_equal(original_board, self.board):
                self.board = original_board
                if action != 0:
                    actions.remove(int(action))
                if actions:
                    action = random.choice(list(actions))
                elif not actions and action != 0:
                    action = 0
                else:
                    found = True
                    self.reward = 0
            else:
                found = True

        self.score += self.reward


        # if np.count_nonzero(self.board) <= 12:
        #     self.reward *= self.calculate_bonus(self.board, original_board)
            
        # elif np.count_nonzero(original_board) > np.count_nonzero(self.board):
        #         self.reward *= (np.count_nonzero(original_board) - np.count_nonzero(self.board))
                
        # if action == 0:
        #     self.reward = -20
        # if action == 3 and (0 not in self.board[-1]):
        #      self.reward = -20

        if not np.array_equal(original_board, self.board):
            self._add_new_tile()

        done = not self._can_move()

        if done:
            self.reward = -1000

        return self.board, self.reward, done

    def calculate_bonus(self, board, original_board):
        bonus = 1
        mean_val = np.mean(board)
        max_val = np.max(board)
        top_five = np.sort(board, axis=None)[-5:]
        
        unique = np.count_nonzero(board[-1]) == len(np.unique(board[-1][np.nonzero(board[-1])]))

        if unique:
            for val in top_five:
                if val in board[-1]:
                    bonus += val

        if board[-1,0] == max_val:
            bonus *= np.log2(max_val) * 3

        if max_val > self.max_tile:
            bonus *= np.log2(max_val)
            self.max_tile = max_val

        # if np.count_nonzero(original_board) > np.count_nonzero(board):
        #     bonus *= (np.count_nonzero(original_board) - np.count_nonzero(board))

        # if (np.count_nonzero(board) > 10) and (2 in board[-1]):
        #     bonus -= (board[-1] == 2).sum()

        # if (board == 2).sum() >= 7:
        #     bonus -= (board == 2).sum()

        return float(bonus)


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
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.n_moves = 0
        self.reward = 0
        self.max_tile = 2
        self._add_new_tile()
        self._add_new_tile()
        return self.board, {}

    def render(self, mode='human'):
        for row in self.board:
            print('\t'.join(map(str, row)))
        print(f"Score: {self.score}")
        print()
