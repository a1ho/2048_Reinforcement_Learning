from gymnasium.envs.registration import register
import gymnasium as gym
from env_2048 import Game2048Env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import torch
from torch import nn

env = Game2048Env()

check_env(env, warn=True)

# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         n_input_channels = 1  # The game board is a single channel

#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None, None]).float()).shape[1]

#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU()
#         )

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         # Add a channel dimension to the observations
#         observations = observations.unsqueeze(1)
#         return self.linear(self.cnn(observations))

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=256),
# )
policy_kwargs = dict(
    net_arch=[256, 256, 256, 256],  # Two hidden layers with 256 units each
    activation_fn=torch.nn.ReLU  # Activation function for the hidden layers
)
# Create the DQN model
model = DQN(
    'MlpPolicy',
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    # learning_rate=0.0003,
    # gae_lambda=0.95,
    # ent_coef=0.01,
    # clip_range=0.2,
    # n_epochs=10,
    # batch_size=64,
    # vf_coef=0.5
    # )
    learning_rate=0.0005,  # Lower learning rate
    buffer_size=100000,  # Larger buffer size
    learning_starts=1000,
    batch_size=64,  # Larger batch size
    target_update_interval=500,
    gamma=0.99,
    train_freq=(4, 'step'),
    exploration_fraction=0.5,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02)

    # learning_rate=0.001, 
    # buffer_size=10000, 
    # learning_starts=1000, 
    # batch_size=32, 
    # target_update_interval=1000, 
    # gamma=0.99, 
    # train_freq=4)

# Train the model

# Save the model
#model.save("dqn_2048")


# scores = []
# for eps in range(5):
#     terminated = False
#     truncated = False
#     observation, _ = env.reset()
#     while not terminated or truncated:
#         action = env.action_space.sample()  # Take a random action
#         observation, reward, terminated, truncated, info = env.step(action)
#         moves = {0: 'up', 1: 'down', 2: 'left', 3:'right'}
#         env.render(mode='rgb_array')
#         if terminated or truncated:
#             print(f'\reps {eps + 1}: score - {env.game.score}', end='')
#             #observation = env.reset()
#     scores.append(env.game.score)
# print(f'Average Score with Baseline Model: {np.mean(scores)}')

model.learn(total_timesteps=1000000, log_interval=100)


scores = []
for eps in range(5):
    terminated = False
    truncated = False
    obs, _ = env.reset()
    while not terminated or truncated:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render('rgb_array')
        if terminated or truncated:
            print(f'\reps {eps + 1}: score - {env.game.score}\t', end='')
    scores.append(env.game.score)
print(f'\nAverage Score with DQN Model: {np.mean(scores)}')

