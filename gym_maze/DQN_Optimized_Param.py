import sys
import os
import numpy as np
import math
import random
import gym
import gym_maze
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

time_log = int(time.time())
models_dir = f"models/{time_log}/"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = make_vec_env("maze-warehouse-picks-v0")
env.reset()


model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=f"./warehouse_tensorboard/{time_log}", batch_size=128,
            learning_rate=6.3e-4, buffer_size=50000, gamma=0.99, target_update_interval=250, exploration_fraction=0.20,
            exploration_final_eps=0.1, policy_kwargs=dict(net_arch=[256, 256]))


maze_env = env.envs[0].unwrapped
print('current state:\n', maze_env0.state.T)
env.reset()

TIMESTEPS = 100000
for i in range(26):
    log = env.render()
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="dqn_run", log_interval=10)
    model.save(f"{models_dir}/{TIMESTEPS * i}")
# Enjoy trained agent
obs = env.reset()

for i in range(25000):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)