import random
import time
import numpy as np
import pandas as pd
from gym import logger

"""
Got the idea from:
T. Schaul 2015: https://arxiv.org/pdf/1511.05952.pdf
to train every X number of steps instead of:
- every step as in other articles/implementations
- at the end of every episode (as was current implementation)
This is better because:
- training decoupled from env.steps, we can do this because we learn from memory
- we can tune train frequency which is computationally intensive
- agents do not to keep track of trained steps, as it is fed to train()
"""


def constant_decay_epsilon(epoch: int,
                           initial_epsilon: float = 1,
                           decay_rate: float = 0.9,
                           min_epsilon: float = 0.01):
    epsilon = initial_epsilon * decay_rate ** epoch
    return max(epsilon, min_epsilon)


class Runner:

    def __init__(self, env, serializer, agent,
                 epsilon_policy=lambda e: constant_decay_epsilon(e),
                 training_period: int = 50,
                 max_episode_steps: int = 200):
        self._env = env
        self._serializer = serializer
        self._agent = agent
        self._epsilon_policy = epsilon_policy
        self._max_episode_steps = max_episode_steps
        self._history = pd.DataFrame(columns=['epsilon', 'reward', 'steps', 'aborted', 'loss'])
        self._epochs_trained = 0
        self._train_period = training_period
        self._train_steps = 0

    def warm_up(self, num_steps: int = 128):
        """ "warm-up" agents by initially populating memories with random actions """
        state = self._env.reset()
        for s in range(num_steps):
            action = self._env.action_space.sample()
            next_state, reward, done, _ = self._env.step(action)
            self._agent.process_observation(self._serializer.serialize(state), action,
                                            reward, self._serializer.serialize(next_state), done)
            state = next_state
            if done:
                state = self._env.reset()

    def train(self, num_epochs: int, num_episodes: int, render_frequency: int = 0):
        for _ in range(num_epochs):
            epsilon = self._epsilon_policy(self._epochs_trained)
            t, rewards, steps, aborted_episodes = self.run_epoch(epsilon, num_episodes,
                                                                 training=True, render_frequency=render_frequency)
            logger.info(
                '{:5.2f}s - {:5.1f}ms/ep ==> Epoch {:2d}: epsilon = {:4.2f} | Average reward/episode = {:5.2f} | '
                'Average steps/episode = {:5.1f} | {:4.1f}%% ({}) aborted episodes'.format(
                    t, t * 1000 / num_episodes, self._epochs_trained, epsilon, np.mean(rewards), np.mean(steps),
                    aborted_episodes * 100 / num_episodes, aborted_episodes))
            self._epochs_trained += 1
        return self.history

    def demonstrate(self, num_episodes: int):
        t, rewards, steps, aborted_episodes = self.run_epoch(epsilon=0, num_episodes=num_episodes, training=False)
        logger.info(
            '{:5.2f}s - {:5.1f}ms/ep ==> Demonstration over {:3d} episodes: Average reward/episode = {:5.2f} | '
            'Average steps/episode = {:5.1f} | {:4.1f}%% ({}) aborted episodes'.format(
                t, t * 1000 / num_episodes, num_episodes, np.mean(rewards), np.mean(steps),
                aborted_episodes * 100 / num_episodes, aborted_episodes))
        return t, np.mean(rewards), np.mean(steps), aborted_episodes

    def render(self):
        total_reward, done, steps = self.run_episode(epsilon=0, training=False, render=True)
        return total_reward, done, steps

    @property
    def history(self):
        return self._history

    def run_episode(self, epsilon: float, training: bool = True, render: bool = False):
        state = self._env.reset()
        if render:
            self._env.render()
        total_reward = 0
        h = None
        for step in range(self._max_episode_steps):
            action = self._agent.act(self._serializer.serialize(state))
            if random.random() < epsilon:
                action = self._env.action_space.sample()
            next_state, reward, done, _ = self._env.step(action)
            total_reward += reward
            if training:
                self._agent.process_observation(self._serializer.serialize(state), action,
                                                reward, self._serializer.serialize(next_state), done)
                self._train_steps += 1
                if self._train_steps % self._train_period == 0:
                    h = self._agent.train(self._train_steps)
            if render:
                self._env.render()
            if done:
                break
            state = next_state
        if training and h is not None:
            self._history = self._history.append({'epsilon': epsilon,
                                                  'reward': total_reward,
                                                  'steps': step + 1,
                                                  'aborted': not done,
                                                  'loss': np.mean(h.history['loss'])},  # mean across num_epochs
                                                 ignore_index=True)
        return total_reward, done, step + 1

    def run_epoch(self, epsilon: float, num_episodes: int, training: bool = True, render_frequency: int = 0):
        rewards = []
        steps = []
        aborted_episodes = 0
        start = time.time()
        if render_frequency > num_episodes:
            logger.warn('render_frequency: {} >= num_episodes: {}'.format(render_frequency, num_episodes))
        for i in range(num_episodes):
            render = False
            if render_frequency != 0 and (i + 1) % render_frequency == 0:
                render = True
            r, done, step = self.run_episode(epsilon, training, render)
            if render_frequency != 0 and (i + 1) % render_frequency == 0:
                logger.info('Rendered example episode ({} steps) with reward: {:6.2f}'.format(step, r))
            rewards.append(r)
            steps.append(step)
            if not done:
                aborted_episodes += 1
        end = time.time()

        return end - start, rewards, steps, aborted_episodes
