import gym
import numpy as np
from gym import logger
from dqn_agent import DQNAgent
from table_agent import TableAgent
from runner import linear_decay_epsilon, quadratic_decay_epsilon, constant_decay_epsilon, run_episode, run_epoch

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
env = gym.make('FrozenLake-v0')
env.seed(0)

agent = DQNAgent(env.action_space.n, env.observation_space.n, gamma=0.9)

epochs = 60
episodes = 400

# train
for e in range(epochs):
    epsilon = constant_decay_epsilon(e, initial_epsilon=1, decay_rate=0.9, min_epsilon=0.01)
    run_epoch(env, agent, epsilon, e, episodes)

# demonstrate
rewards = []
demonstration = 100
for i in range(demonstration):
    r, _, _ = run_episode(env, agent, epsilon=0, max_iterations=100, training=False)
    rewards.append(r)
logger.info('Demonstration over {} episodes with average reward/episode = {:.3}'.format(demonstration,
                                                                                        np.mean(rewards)))
# debug
agent.print_q_map()

run_episode(env, agent, 0, training=False, render=True)

agent.save()
