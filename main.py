import gym
import numpy as np
from gym import logger
from nn_agent import DQNAgent
from table_agent import TableAgent
from runner import linear_decay_epsilon, quadratic_decay_epsilon, run_episode, run_epoch

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

logger.set_level(logger.INFO)
# env = gym.make('Taxi-v2')
env = gym.make('FrozenLakeNotSlippery-v0')
env.seed(0)

agent = DQNAgent(env.action_space.n, env.observation_space.n)

epochs = 5
episodes = 100

# train
for e in range(epochs):
    epsilon = quadratic_decay_epsilon(e, epochs)
    run_epoch(env, agent, epsilon, e, episodes)

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
