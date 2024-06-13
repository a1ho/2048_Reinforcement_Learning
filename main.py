import gymnasium as gym
from env_2048 import Game2048Env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import numpy as np
import torch
from torch import nn

env = Game2048Env(rand = True)
print('Baseline Model:\n')
scores = []
max_tiles = {}
for eps in range(5):
    terminated = False
    truncated = False
    observation, _ = env.reset()
    while not terminated or truncated:
        action = env.action_space.sample()  # Take a random action
        observation, reward, terminated, truncated, info = env.step(action)
        moves = {0: 'up', 1: 'down', 2: 'left', 3:'right'}
        env.render(mode='rgb_array')
        if terminated or truncated:
            print(f'\reps {eps + 1}: score - {env.game.score}', end='')
            #observation = env.reset()
    scores.append(env.game.score)
    if np.max(env.game.board) in max_tiles:
        max_tiles[np.max(env.game.board)] += 1
    else:
        max_tiles[np.max(env.game.board)] = 1
max_tiles = {key : val/100 for (key, val) in max_tiles.items()}
max_tiles = dict(sorted(max_tiles.items()))
print(f'''
    Average Score: {np.mean(scores)}
    Std. Dev. of Scores: {np.std(scores)}
    Max Tiles Achieved out of 100 episodes: {max_tiles}
    ''')



env = Game2048Env()

model = DQN(
    'MlpPolicy',
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=0.0005,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    target_update_interval=500,
    gamma=0.99,
    train_freq=(4, 'step'),
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02
    )
    
check_env(env, warn=True)

model.learn(total_timesteps=500000, log_interval=100)
model.save("dqn_2048")

print('DQN Model: \n')
scores = []
max_tiles = {}

for eps in range(100):
    terminated = False
    truncated = False
    obs, _ = env.reset()
    while not terminated or truncated:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        if eps > 89:
            env.render('rgb_array')
        if terminated or truncated:
            print(f'\reps {eps + 1}: score - {env.game.score}\t', end='')
    scores.append(env.game.score)
    if np.max(env.game.board) in max_tiles:
        max_tiles[np.max(env.game.board)] += 1
    else:
        max_tiles[np.max(env.game.board)] = 1

max_tiles = {key : val/100 for (key, val) in max_tiles.items()}
max_tiles = dict(sorted(max_tiles.items()))
print(f'''
    Average Score: {np.mean(scores)}
    Std. Dev. of Scores: {np.std(scores)}
    Max Tiles Achieved out of 100 episodes: {max_tiles}
    ''')

