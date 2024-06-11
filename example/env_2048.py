import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from game_2048 import Game2048


class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(Game2048Env, self).__init__()
        self.game = Game2048()
        
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(4, 4), dtype=np.int64)
        self.fig, self.ax = plt.subplots()
        self.tile_colors = {
            0: (204, 192, 179),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }


    def step(self, action):
        observation, reward, terminated = self.game.step(action)
        truncated = False
        info = {}
        #observation = self._convert_to_3d(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation, _ = self.game.reset()
        return observation, {}

    def _convert_to_3d(self, board):
        one_hot_encoded = np.zeros((4, 4, 11), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                value = board[i, j]
                if value == 0:
                    one_hot_encoded[i, j, 0] = 1
                else:
                    index = int(np.log2(value))
                    one_hot_encoded[i, j, index] = 1
        return one_hot_encoded

    def render(self, mode='human'):
        if mode == 'human':
            self.game.render()
        elif mode == 'rgb_array':
            self.ax.clear()
            rgb_array = self.get_rgb_array()
            self.ax.imshow(rgb_array)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])

            # Add borders and text annotations for each tile
            for i in range(4):
                for j in range(4):
                    value = self.game.board[i, j]
                    # Draw the border
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='lightgrey', facecolor='none')
                    self.ax.add_patch(rect)
                    if value != 0:
                        # Adjust text color based on the tile value for better visibility
                        text_color = 'white' if value >= 8 else 'black'
                        self.ax.text(j, i, str(value), ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')
            self.ax.text(0, -0.5, f'Move count: {self.game.n_moves}', ha='left', va='center', fontsize=12, fontweight='bold', color='black')

            plt.draw()
            plt.pause(0.1)
            return rgb_array

    def get_rgb_array(self):
        board_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                value = self.game.board[i, j]
                if value in self.tile_colors:
                    board_rgb[i, j] = self.tile_colors[value]
                else:
                    # Fallback color for values not in the colormap
                    board_rgb[i, j] = (0, 0, 0)
        return board_rgb


    def close(self):
        plt.close(self.fig)
