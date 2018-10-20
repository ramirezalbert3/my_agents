import gym
import numpy as np
from gym import logger
from agent import Agent
from table_agent import TableAgent
from runner import decaying_epsilon, run_episode, run_epoch

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

logger.set_level(logger.INFO)
env = gym.make('Taxi-v2')
env.seed(0)

agent = TableAgent(env.action_space.n, env.observation_space.n)

epochs = 10
episodes = 500

# train
for e in range(epochs):
    epsilon = decaying_epsilon(e, epochs)
    run_epoch(env, agent, epsilon, e, episodes)
    agent.train()

# demonstrate
rewards = []
demonstration = 100
for i in range(demonstration):
    rewards.append(run_episode(env, agent, 0, training=False))
logger.info('Demonstration over {} episodes with average reward/episode = {:.2}'.format(demonstration,
                                                                                        np.mean(rewards)))

# run_episode(env, agent, 0, training=False, render=True)

# debug
agent.print_q_map()
