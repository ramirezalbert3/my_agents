import gym
import numpy as np
from gym import logger
from agents.dqn_agent import DQNAgent
from agents.new import DQNAgent as NewAgent
from core.states import StateSerializer
from core.runner import constant_decay_epsilon, Runner
import time

logger.set_level(logger.INFO)

epochs = 5
episodes = 100

env_name = 'CartPole-v0' # 'CartPole-v0'  'Taxi-v2'    'FrozenLake-v0'
env = gym.make(env_name)
env.seed(0)

serializer = StateSerializer(env.observation_space.shape)

times = []
for i in range(5):
    agent = DQNAgent(env.action_space.n, serializer.shape, gamma=0.85)
    runner = Runner(env, serializer, agent,
                    epsilon_policy = lambda e: constant_decay_epsilon(e,
                                                                    initial_epsilon=1,
                                                                    decay_rate=0.8,
                                                                    min_epsilon=0.01),
                    max_episode_steps = 200)
    start = time.time()
    history = runner.train(epochs, episodes)
    results = runner.demonstrate(num_episodes=100)
    times.append(time.time() - start)

new_times = []
for i in range(5):
    agent = NewAgent(env.action_space.n, serializer.shape, gamma=0.85)
    runner = Runner(env, serializer, agent,
                    epsilon_policy = lambda e: constant_decay_epsilon(e,
                                                                    initial_epsilon=1,
                                                                    decay_rate=0.8,
                                                                    min_epsilon=0.01),
                    max_episode_steps = 200)
    start = time.time()
    history = runner.train(epochs, episodes)
    results = runner.demonstrate(num_episodes=100)
    new_times.append(time.time() - start)

print('old agent times:\n', times, '\naverage time per training {}s'.format(np.mean(times)))
print('new agent times:\n', new_times, '\naverage time per training {}s'.format(np.mean(new_times)))
