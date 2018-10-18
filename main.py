import gym
import numpy as np
from gym import wrappers, logger
from agent import Agent

logger.set_level(logger.INFO)
env = gym.make('FrozenLake-v0')
# env = wrappers.Monitor(env, directory='/tmp/rf', force=True)
env.seed(0)

agent = Agent(env.action_space.n)

epochs = 20
episodes = 500

### TODO: MOVE THESE INTO RUNNER
def decaying_epsilon (e, epochs):
    return 1-e/epochs

def run_episode(epsilon, train=True):
        state = env.reset()
        if not train:
            env.render()
        done = False
        steps = 0
        total_reward = 0
        while not done:
            action = agent.act(epsilon, state)
            previous_state = state
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if train:
                agent.store_observation(previous_state, action, reward)
            else:
                env.render()
        return total_reward
###

# train
for e in range(epochs):
    epsilon = decaying_epsilon(e, epochs)
    rewards = []
    for i in range(episodes):
        rewards.append(run_episode(epsilon))
    logger.info('Epoch {}:\tEpsilon = {:.2}\tAverage Reward = {:.2}'.format(e, epsilon, np.mean(rewards)))
    agent.train()

# demonstrate
for i in range(1):
    run_episode(epsilon, train=False)
