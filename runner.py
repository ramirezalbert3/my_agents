import random

def decaying_epsilon (e, epochs):
    return 1 - e / epochs

def run_episode(env, agent, epsilon, training=True, render=False):
        state = env.reset()
        if render:
            env.render()
        done = False
        steps = 0
        total_reward = 0
        for step in range(50):
            action = agent.act(state, training)
            if random.random() < epsilon:
                action = env.action_space.sample()
            previous_state = state
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if training:
                agent.store_observation(state, previous_state, action, reward, done)
            if render:
                env.render()
            if done:
                break
        return total_reward
