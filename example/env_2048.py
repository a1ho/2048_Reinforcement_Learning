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

    def step(self, action):
        observation, reward, done = self.game.step(action)
        info = {}
        return observation, reward, done, info

    def reset(self):
        return self.game.reset()

    def render(self, mode='human'):
        if mode == 'human':
            self.game.render()
        elif mode == 'rgb_array':
            self.ax.clear()
            self.ax.imshow(self.get_rgb_array())
            plt.draw()
            plt.pause(0.01)

    def get_rgb_array(self):
        colormap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=2**16)
        return colormap(norm(self.game.board))

    def close(self):
        plt.close(self.fig)
