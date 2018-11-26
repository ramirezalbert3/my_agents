import gym
import numpy as np
from gym import logger
from agents.ddqn_agent import DDQNAgent
from agents.nstep_agent import NStepDDQNAgent
from core.states import StateSerializer
from core.runner import constant_decay_epsilon, Runner
from core.visualization import rolling_mean
import time
import matplotlib.pyplot as plt

logger.set_level(logger.INFO)

runs = 5
epochs = 20
episodes = 400

env_name = 'CartPole-v0' # 'CartPole-v0'  'Taxi-v2'    'FrozenLake-v0'
env = gym.make(env_name)
env.seed(0)

serializer = StateSerializer(env.observation_space.shape)

times = []
rewards = []
axis = None
legend = []
for i in range(runs):
    agent = DDQNAgent(env.action_space.n, serializer.shape, gamma=0.9)
    runner = Runner(env, serializer, agent,
                    epsilon_policy = lambda e: constant_decay_epsilon(e,
                                                                    initial_epsilon=1,
                                                                    decay_rate=0.8,
                                                                    min_epsilon=0.01),
                    training_period=50,
                    max_episode_steps = 200)
    runner.warm_up()
    start = time.time()
    history = runner.train(epochs, episodes)
    _, reward, _, _ = runner.demonstrate(num_episodes=100)
    times.append(time.time() - start)
    rewards.append(reward)
    axis = rolling_mean([history['reward'], history['loss']], window=100, axis=axis, show=False)
    legend.append('{}_{}'.format(type(agent).__name__, i))

new_times = []
new_rewards = []
for i in range(runs):
    agent = NStepDDQNAgent(env.action_space.n, serializer.shape, update_horizon=5, gamma=0.9)
    runner = Runner(env, serializer, agent,
                    epsilon_policy = lambda e: constant_decay_epsilon(e,
                                                                    initial_epsilon=1,
                                                                    decay_rate=0.8,
                                                                    min_epsilon=0.01),
                    training_period=50,
                    max_episode_steps = 200)
    runner.warm_up()
    start = time.time()
    history = runner.train(epochs, episodes)
    _, reward, _, _  = runner.demonstrate(num_episodes=100)
    new_times.append(time.time() - start)
    new_rewards.append(reward)
    axis = rolling_mean([history['reward'], history['loss']], window=100, label='{}_{}'.format(type(agent).__name__, i), axis=axis, show=False)
    legend.append('{}_{}'.format(type(agent).__name__, i))

times = np.array(times)
new_times = np.array(new_times)
rewards = np.array(rewards)
new_rewards = np.array(new_rewards)

print('old agent times:\n', times, '\naverage time per training {:.4}s'.format(np.mean(times)))
print('old agent rewards:\n', rewards, '\naverage reward per training {:.4}'.format(np.mean(rewards)))
print('old agent rewards/time:\n', rewards/times, '\naverage reward/time {:.4}'.format(np.mean(rewards/times)))
print('new agent times:\n', new_times, '\naverage time per training {:.4}s'.format(np.mean(new_times)))
print('new agent rewards:\n', new_rewards, '\naverage reward per training {:.4}'.format(np.mean(new_rewards)))
print('new agent rewards/time:\n', new_rewards/new_times, '\naverage reward/time {:.4}'.format(np.mean(new_rewards/new_times)))

for ax in axis:
    ax.legend(legend)
plt.show()
