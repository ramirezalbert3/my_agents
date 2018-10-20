import random
import time
import numpy as np
from gym import logger

def linear_decay_epsilon (e, epochs, min_decay = 0.1):
    epsilon = 1 - e / epochs
    if epsilon < min_decay:
        return min_decay
    return epsilon

def quadratic_decay_epsilon(e, epochs, min_decay = 0.1):
    epsilon = linear_decay_epsilon(e, epochs, 0) ** 2
    if epsilon < min_decay:
        return min_decay
    return epsilon

def run_episode(env, agent, epsilon: float, training: bool =True, render: bool =False):
        state = env.reset()
        if render:
            env.render()
        done = False
        steps = 0
        total_reward = 0
        for step in range(50):
            action = agent.act(state)
            if random.random() < epsilon:
                action = env.action_space.sample()
            previous_state = state
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if training:
                agent.process_observation(state, previous_state, action, reward, done)
            else:
                logger.debug('Choosing {} in state {} for Q-Values: {}'.format(action,
                                                                          previous_state,
                                                                          agent.Q(previous_state)))
            if render:
                env.render()
            if done:
                break
        return total_reward

def run_epoch(env, agent, epsilon: float, epoch: int, episodes: int):
    # TODO: pass epsilon policy?
    rewards = []
    start = time.time()
    for i in range(episodes):
        rewards.append(run_episode(env, agent, epsilon))
    end = time.time()
    logger.info('({:.3}s)\t==> Epoch {}:\tepsilon = {:.2}\tAverage reward/episode = {:.2}'.format(end-start,
                                                                                                 epoch,
                                                                                    epsilon,
                                                                                    np.mean(rewards)
                                                                                    ))
