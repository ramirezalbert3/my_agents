from dqn_agent import DQNAgent
from runner import run_episode
import gym
agent = DQNAgent.from_h5()
env = gym.make('FrozenLake-v0')
run_episode(env, agent, 0, training=False, render=True)
