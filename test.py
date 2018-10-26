import gym
from gym import logger
from core.states import StateSerializer
from core.runner import Runner
from agents.dqn_agent import DQNAgent

logger.set_level(logger.INFO)

env_name = 'CartPole-v0'
env = gym.make(env_name)
env._max_episode_steps = 500

serializer = StateSerializer(env.observation_space.shape)

agent = DQNAgent.from_h5(file_path=env_name+'.h5')

runner = Runner(env, serializer, agent,
                epsilon_policy = lambda e: 0,
                max_episode_steps = 500)

runner.render()

runner.demonstrate(num_episodes=100)
