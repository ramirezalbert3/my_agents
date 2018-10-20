import time
import numpy as np
from gym import logger
from sklearn.linear_model import PassiveAggressiveRegressor

# TODO: de-couple and abstract the one_hot encoding
# should not be 'just part of the agent'
def one_hot(size, idx):
    res = np.zeros(size)
    res[idx] = 1
    return res

def one_hot_states_and_actions(num_states, state, num_actions, action):
    return np.hstack(
        (one_hot(num_states, state), one_hot(num_actions, action))
        )

class Agent:
    '''
    Attempt to write an agent that uses
    - gather observations during an epoch
    - at the end of each epoch, 'partial_fit' state/action pairs with rewards
    '''
    # TODO: change num_states for state_size
    def __init__(self, num_actions: int, num_states: int, gamma: float = 0.99):
        self._gamma = gamma
        self._num_actions = num_actions
        self._num_states = num_states
        X = np.array([one_hot_states_and_actions(num_states, s, num_actions, a)
             for s in range(num_states)
             for a in range(num_actions)])
        y = np.full(len(X), 0.0)
        self._q_impl = PassiveAggressiveRegressor().partial_fit(X, y)
        self._observations = ([], [])
    
    def act(self, state: tuple):
        ''' Get either a greedy action '''
        return self.policy(state)
    
    def process_observation(self, next_state: tuple, state: tuple, action: int, reward: float, done: bool):
        ''' Store observation to train at the end of epoch '''
        X, y = self._observations
        
        X.append(one_hot_states_and_actions(self._num_states, state, self._num_actions, action))
        
        next_q = reward
        if not done:
            # q = immediate_reward + discounted playing 'perfectly' from now on
            next_q += self._gamma * self.V(next_state)
        y.append(next_q)
        
        assert len(X) == len(y)
        self._observations = (X, y)
    
    def train(self):
        ''' after an epoch 're-fit' Q with observations '''
        X, y = self._observations # [(state, action)], [reward]
        start = time.time()
        self._q_impl.partial_fit(X, y)
        end = time.time()
        logger.debug('\t# Fitting with {} samples took {:.3} seconds'.format(len(X), end-start))
        self._observations = ([], []) # reset observations
    
    def Q(self, state):
        ''' value of any taken action in a given state and playing perfectly onwards '''
        return [self._q_impl.predict(one_hot_states_and_actions(self._num_states, state,
                                                                 self._num_actions, action).reshape(1, -1) # single sample reshape
                                )[0]
                for action in range(self._num_actions)]
    
    def policy(self, state):
        ''' optimal greedy action for a state '''
        return np.argmax(self.Q(state))
    
    def V(self, state):
        ''' value of being in a given state (and playing perfectly onwards) '''
        return np.max(self.Q(state))
    
    def print_q_map(self):
        for s in range(self._num_states):
            print('State {} Q-values: {}'.format(s, self.Q(s)))
