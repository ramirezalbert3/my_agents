import time
import numpy as np
import pandas as pd
from gym import logger
import tensorflow as tf
from tensorflow import keras

def build_dense_network(num_actions: int, num_states: int):
    model = keras.models.Sequential([
        keras.layers.Dense(64,
                           activation='relu',
                           input_shape=(num_states + num_actions,)), # one-hot encoding
        keras.layers.Dense(64,
                           activation='relu'),
        keras.layers.Dense(32),
        ])
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  etrics=['mae'])  # mean absolute error
    return model

def one_hot(list_of_pairs):
    res = None
    for size, idx in list_of_pairs:
        if idx >= size:
            raise RuntimeError('Number of options must be higher
                than the index for one-hot encoding')
        e = np.zeros(size)
        e[idx] = 1
        if res is None:
            res = e
        else:
            res = np.hstack((res, e))
    return res

class Agent:
    '''
    Attempt to write an agent with keras tensorflow API
    '''
    # TODO: change num_states for state_size/shape
    def __init__(self, num_actions: int, num_states: int, gamma: float = 0.99):
        self._gamma = gamma
        self._num_actions = num_actions
        self._num_states = num_states
        
        self._q_impl = build_dense_network(num_actions, num_states)
    
    def act(self, state: tuple):
        ''' Get either a greedy action '''
        return self.policy(state)
    
    def process_observation(self, next_state: tuple, state: tuple, action: int, reward: float, done: bool):
        ''' Store observation to train at the end of epoch '''
        X, y = self._observations
        
        X.append(one_hot([(self._num_states, state), (self._num_actions, action)]))
        
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
        self._q_impl.fit(X, y, epochs=5, batch_size=32)
        self._observations = ([], []) # reset observations
    
    def Q(self, state):
        ''' value of any taken action in a given state and playing perfectly onwards '''
        one_hot_states = [one_hot([(self._num_states, state), (self._num_actions, a)])
                          for a in self._num_actions]
        return model.predict(x, batch_size=self._num_actions)
    
    def policy(self, state):
        ''' optimal greedy action for a state '''
        return np.argmax(self.Q(state))
    
    def V(self, state):
        ''' value of being in a given state (and playing perfectly onwards) '''
        return np.max(self.Q(state))
    
    def print_q_map(self):
        q_table = pd.DataFrame([self.Q(s) for s in range(self._num_states)]).transpose()
        print(q_table)
