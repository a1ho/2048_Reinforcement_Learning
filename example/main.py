from gymnasium.envs.registration import register
import gymnasium as gym
from env_2048 import Game2048Env

env = Game2048Env()

observation = env.reset()
env.render(mode='human')

done = False
while done == False:
    action = env.action_space.sample()  # Take a random action
    observation, reward, done, info = env.step(action)
    env.render(mode='human')
    if done:
        print("Game Over! Final Score:", reward)
        #observation = env.reset()

env.close()