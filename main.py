import gym
import numpy as np
from gym import logger
from agents.dqn_agent import DQNAgent
from core.states import StateSerializer
from core.runner import constant_decay_epsilon, Runner

logger.set_level(logger.INFO)

env_name = 'CartPole-v0' # 'CartPole-v0'  'Taxi-v2'    'FrozenLake-v0'
env = gym.make(env_name)
env.seed(0)

serializer = StateSerializer(env.observation_space.shape)
# serializer = StateSerializer.from_num_states(env.observation_space.n)

agent = DQNAgent(env.action_space.n, serializer.shape, gamma=0.85)
# agent = DQNAgent.from_h5(file_path=env_name+'.h5', gamma=0.9)

epochs = 20
episodes = 200

runner = Runner(env, serializer, agent,
                epsilon_policy = lambda e: constant_decay_epsilon(e,
                                                                  initial_epsilon=1,
                                                                  decay_rate=0.9,
                                                                  min_epsilon=0.01),
                max_episode_steps = 200)

history = runner.train(epochs, episodes)

# demonstrate
results = runner.demonstrate(num_episodes=100)

agent.save(env_name)
