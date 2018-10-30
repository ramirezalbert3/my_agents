import numpy as np
import pandas as pd

class TableAgent:
    def __init__(self, num_actions: int, num_states: int, gamma: float = 0.95, alpha = 0.1):
        self._gamma = gamma
        self._alpha = alpha
        self._num_actions = num_actions
        self._num_states = num_states
        self._q_impl = {state: np.zeros(num_actions)
                        for state in range(num_states)}
    
    def act(self, state: tuple):
        ''' Get either a greedy action '''
        return self.policy(state)
    
    def process_observation(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool):
        ''' Online training performed with observation '''
        new_q = reward
        if not done:
            # q = immediate_reward + discounted playing 'perfectly' from now on
            new_q += self._gamma*self.V(next_state)
        self._q_impl[state][action] += self._alpha * (new_q - self.Q(state)[action])
    
    def train(self, step_num: int):
        ''' Training is done on-line '''
        pass
    
    def Q(self, state):
        ''' value of any taken action in a given state and playing perfectly onwards '''
        return self._q_impl[state]
    
    def policy(self, state):
        ''' optimal greedy action for a state '''
        return np.argmax(self.Q(state))
    
    def V(self, state):
        ''' value of being in a given state (and playing perfectly onwards) '''
        return np.max(self.Q(state))
    
    def print_q_map(self):
        print(pd.DataFrame(self._q_impl))
