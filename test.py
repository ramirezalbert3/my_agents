from dqn_agent import DQNAgent
from runner import run_episode, run_epoch
import gym
from gym import logger
from states import StateSerializer

logger.set_level(logger.INFO)

env_name = 'FrozenLake-v0'
env = gym.make(env_name)

serializer = StateSerializer.from_num_states(env.observation_space.n)

agent = DQNAgent.from_h5(file_path=env_name+'.h5')

# Render
# run_episode(env, agent, 0, training=False, render=True)

# Demonstrate
run_epoch(env, serializer, agent, epsilon=0, epoch=None, episodes=100, max_episode_steps=100, training=False)
