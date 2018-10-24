import gym
import numpy as np
from gym import logger
from dqn_agent import DQNAgent
from runner import constant_decay_epsilon, run_epoch

logger.set_level(logger.INFO)
env = gym.make('Taxi-v2')
# env = gym.make('FrozenLake-v0')
env.seed(0)

agent = DQNAgent(env.action_space.n, env.observation_space.n, gamma=0.95)

epochs = 30
episodes = 600

# train
for e in range(epochs):
    epsilon = constant_decay_epsilon(e, initial_epsilon=1, decay_rate=0.8, min_epsilon=0.01)
    run_epoch(env, agent, epsilon, e, episodes, max_episode_steps=200)

# demonstrate
run_epoch(env, agent, epsilon=0, epoch=None, episodes=100, max_episode_steps=100, training=False)

# debug
# agent.print_q_map()

agent.save()
