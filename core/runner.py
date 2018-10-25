import random
import time
import numpy as np
from gym import logger

def linear_decay_epsilon (iteration: int, max_iterations: int, min_epsilon: float = 0.1):
    epsilon = 1 - iteration / max_iterations
    return max(epsilon, min_epsilon)

def quadratic_decay_epsilon(iteration: int, max_iterations: int, min_epsilon: float = 0.1):
    epsilon = linear_decay_epsilon(iteration, max_iterations, 0) ** 2
    return max(epsilon, min_epsilon)

def constant_decay_epsilon(iteration: int,
                           initial_epsilon: float,
                           decay_rate: float = 0.995,
                           min_epsilon: float = 0.05):
    epsilon = initial_epsilon * decay_rate ** iteration
    return max(epsilon, min_epsilon)

def run_episode(env, serializer, agent, epsilon: float, max_episode_steps:int = 100, training: bool = True, render: bool = False):
        state = env.reset()
        if render:
            env.render()
        done = False
        steps = 0
        total_reward = 0
        for step in range(max_episode_steps):
            action = agent.act(serializer.serialize(state))
            if random.random() < epsilon:
                action = env.action_space.sample()
            previous_state = state
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if training:
                agent.process_observation(serializer.serialize(previous_state), action, reward, serializer.serialize(state), done)
            else:
                logger.debug('Choosing {} in state {} for Q-Values: {}'.format(action,
                                                                          previous_state,
                                                                          agent.Q(serializer.serialize(previous_state))))
            if render:
                env.render()
            if done:
                break
        if training:
            agent.train()
        return total_reward, done, step+1

def run_epoch(env, serializer, agent, epsilon: float, epoch: int, episodes: int, max_episode_steps: int = 100, training: bool = True, render: bool = False):
    # TODO: pass epsilon policy?
    rewards = []
    steps = []
    aborted_episodes = 0
    start = time.time()
    for i in range(episodes):
        r, done, step = run_episode(env, serializer, agent, epsilon, max_episode_steps, training, render)
        rewards.append(r)
        steps.append(step)
        if not done:
            aborted_episodes += 1
    end = time.time()
    if training:
        logger.info('({:.3}s)\t==> Epoch {}:\tepsilon = {:.2}\tAverage reward/episode = {:.3}\tAverage steps/episode = {:.3}\twith {} aborted episodes'.format(
                    end-start, epoch, epsilon, np.mean(rewards), np.mean(steps), aborted_episodes))
    else:
        logger.info('({:.3}s)\t==> Demonstration over {} episodes:\tAverage reward/episode = {:.3}\tAverage steps/episode = {:.3}\twith {} aborted episodes'.format(
                    end-start, episodes, np.mean(rewards), np.mean(steps), aborted_episodes))