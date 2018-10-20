import numpy as np
from gym import logger
import pandas as pd

class TableAgent:
    def __init__(self, num_actions: int, num_states: int, gamma: float = 0.95, alpha = 0.1):
        self._gamma = gamma
        self._alpha = alpha
        self._num_actions = num_actions
        self._num_states = num_states
        self._q_table = {state: np.zeros(num_actions)
                         for state in range(num_states)}
    def act(self, state: tuple, training = True):
        '''
        Get either a greedy action if available
        '''
        return self._policy(state)
    
    def store_observation(self, next_state: tuple, state: tuple, action: int, reward: float, done: bool):
        new_q = reward
        if not done:
            new_q += self._gamma*self._v(next_state)
        self._q_table[state][action] += self._alpha * (new_q - self._q(state)[action])
    
    def train(self):
        pass
    
    def _q(self, state):
        return self._q_table[state]
    
    def _policy(self, state):
        return np.argmax(self._q(state))
    
    def _v(self, state):
        return np.max(self._q(state))
    
    def print_q_map(self):
        print(pd.DataFrame(self._q_table))
