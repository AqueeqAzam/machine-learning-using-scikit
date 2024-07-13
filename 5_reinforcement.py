# State Space -> The State Space is the set of all possible situations our taxi could inhabit.
# Action Space -> The action in our case can be to move in a direction or decide to pickup/dropoff a passenger.
# import liberary

import os
import gym
import numpy as np
import random
from IPython.display import clear_output
import pyglet
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# load the environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)

episodes = 5
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episodes : {} score:{}'.format(episodes, score))
    env.close()

# understand environment
print(env.action_space)
print(env.action_space.sample())
print(env.observation_space.sample())

# Train an RL model
log_path = os.path.join('training', 'Logs')
print(log_path)
env = gym.make(environment_name)
env = DummyVecEnv(lambda : env)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# environment
