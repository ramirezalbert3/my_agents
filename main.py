import gym
import numpy as np
from gym import logger
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.prioritized_ddqn_agent import PrioritizedDDQNAgent
from agents.distributional_agent import DistributionalAgent
from agents.nstep_agent import NStepDDQNAgent
from core.states import StateSerializer
from core.runner import constant_decay_epsilon, Runner
from core.visualization import rolling_mean

try:
    import yappi
except:
    pass

logger.set_level(logger.INFO)

env_name = 'CartPole-v0' # 'CartPole-v0'  'Taxi-v2'    'FrozenLake-v0'
env = gym.make(env_name)
env.seed(0)

serializer = StateSerializer(env.observation_space.shape)
# serializer = StateSerializer.from_num_states(env.observation_space.n)

# agent = DQNAgent(env.action_space.n, serializer.shape, gamma=0.85)
# agent = DDQNAgent(env.action_space.n, serializer.shape, gamma=0.95)
# agent = PrioritizedDDQNAgent(env.action_space.n, serializer.shape, gamma=0.95)
agent = NStepDDQNAgent(env.action_space.n, serializer.shape, update_horizon=3, gamma=0.95)
# agent = DistributionalAgent(env.action_space.n, serializer.shape, v_min=0, v_max=100, num_atoms=51, gamma=0.95)
# agent = DQNAgent.from_h5(file_path=env_name+'.h5', gamma=0.9)

epochs = 15
episodes = 200

try:
    yappi.set_clock_type('cpu')
    yappi.start(builtins=True)
    logger.info('Profiling training')
except:
    pass

runner = Runner(env, serializer, agent,
                epsilon_policy = lambda e: constant_decay_epsilon(e,
                                                                  initial_epsilon=1,
                                                                  decay_rate=0.75,
                                                                  min_epsilon=0.01),
                training_period=50,
                max_episode_steps = 200)

runner.warm_up()
history = runner.train(epochs, episodes)

try:
    stats = yappi.get_func_stats()
    stats.save('callgrind.out', type='callgrind')
except:
    pass

# demonstrate
results = runner.demonstrate(num_episodes=100)

rolling_mean([history['reward'], history['loss']])

agent.save(env_name)

