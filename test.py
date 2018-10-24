from dqn_agent import DQNAgent
from runner import run_episode, run_epoch
import gym
from gym import logger

logger.set_level(logger.INFO)

env_name = 'Taxi-v2' # 'FrozenLake-v0'
env = gym.make(env_name)
agent = DQNAgent.from_h5(file_path=env_name+'.h5')

# Render
# run_episode(env, agent, 0, training=False, render=True)

# Demonstrate
run_epoch(env, agent, epsilon=0, epoch=None, episodes=100, max_episode_steps=100, training=False)
