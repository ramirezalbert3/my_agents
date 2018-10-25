from agents.dqn_agent import DQNAgent
from core.runner import run_episode, run_epoch
import gym
from gym import logger
from core.states import StateSerializer

logger.set_level(logger.INFO)

env_name = 'CartPole-v0'
env = gym.make(env_name)
env._max_episode_steps = 500

serializer = StateSerializer(env.observation_space.shape)

# agent = DQNAgent(env.action_space.n, serializer.shape, gamma=0.95)
agent = DQNAgent.from_h5(file_path=env_name+'.h5')

# Render
run_episode(env, serializer, agent, epsilon=0, max_episode_steps=5000, training=False, render=True)

# Demonstrate
run_epoch(env, serializer, agent, epsilon=0, epoch=None, episodes=100, max_episode_steps=5000, training=False)
