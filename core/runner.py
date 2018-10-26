import random
import time
import numpy as np
from collections import namedtuple
from gym import logger

def constant_decay_epsilon(epoch: int,
                           initial_epsilon: float = 1,
                           decay_rate: float = 0.9,
                           min_epsilon: float = 0.01):
    epsilon = initial_epsilon * decay_rate ** epoch
    return max(epsilon, min_epsilon)

class Runner:
    record = namedtuple('Record', ['time', 'num_episodes', 'mean_reward', 'mean_steps', 'aborted_episodes'])
    
    def __init__(self, env, serializer, agent,
                 epsilon_policy = lambda e: constant_decay_epsilon(e),
                 max_episode_steps = 200):
        self._env = env
        self._serializer = serializer
        self._agent = agent
        self._epsilon_policy = epsilon_policy
        self._max_episode_steps = max_episode_steps
    
    def train(self, num_epochs: int, num_episodes: int):
        history = []
        for e in range(num_epochs):
            epsilon = self._epsilon_policy(e)
            time, mean_reward, mean_steps, aborted_episodes = self._run_epoch(epsilon, num_episodes, training=True)
            logger.info('({:.3}s)\t==> Epoch {}:\tepsilon = {:.2}\tAverage reward/episode = {:.3}\tAverage steps/episode = {:.3}\twith {} aborted episodes'.format(
                        time, e, epsilon, mean_reward, mean_steps, aborted_episodes))
            history.append(Runner.record(time, num_episodes, mean_reward, mean_steps, aborted_episodes))
        return history
    
    def demonstrate(self, num_episodes):
        time, mean_reward, mean_steps, aborted_episodes = self._run_epoch(epsilon=0, num_episodes=num_episodes, training=False)
        logger.info('({:.3}s)\t==> Demonstration over {} episodes:\tAverage reward/episode = {:.3}\tAverage steps/episode = {:.3}\twith {} aborted episodes'.format(
                        time, num_episodes, mean_reward, mean_steps, aborted_episodes))
        return Runner.record(time, num_episodes, mean_reward, mean_steps, aborted_episodes)
    
    def render(self):
        total_reward, done, steps = self._run_episode(epsilon=0, training=False, render=True)
        return total_reward, done, steps
    
    def _run_episode(self, epsilon: float, training: bool = True, render: bool = False):
        state = self._env.reset()
        if render:
            self._env.render()
        total_reward = 0
        for step in range(self._max_episode_steps):
            action = self._agent.act(self._serializer.serialize(state))
            if random.random() < epsilon:
                action = self._env.action_space.sample()
            previous_state = state
            state, reward, done, _ = self._env.step(action)
            total_reward += reward
            if training:
                self._agent.process_observation(self._serializer.serialize(previous_state), action, reward, self._serializer.serialize(state), done)
            else:
                logger.debug('Choosing {} in state {} for Q-Values: {}'.format(action,
                                                                               previous_state,
                                                                               self._agent.Q(self._serializer.serialize(previous_state))))
            if render:
                self._env.render()
            if done:
                break
        if training:
            self._agent.train()
        return total_reward, done, step+1
    
    def _run_epoch(self, epsilon: float, num_episodes: int, training: bool = True, render: bool = False):
        rewards = []
        steps = []
        aborted_episodes = 0
        start = time.time()
        for i in range(num_episodes):
            r, done, step = self._run_episode(epsilon, training, render)
            rewards.append(r)
            steps.append(step)
            if not done:
                aborted_episodes += 1
        end = time.time()
        
        return end-start, np.mean(rewards), np.mean(steps), aborted_episodes
