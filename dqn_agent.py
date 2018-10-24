from collections import deque
import random
import numpy as np
import pandas as pd
from gym import logger
import tensorflow as tf
from tensorflow import keras

'''
# References
# 0. https://www.tensorflow.org/guide/keras
# 1. https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# 2. https://keon.io/deep-q-learning/
#
# [1] and [2] basically implement the same thing
# [1] one uses/explains 2 models for Q target_Q & Q for stability
# [2] implements it in github, but does not explain it in the article
'''

def one_hot(size, idx):
    res = np.zeros(size)
    res[idx] = 1
    return res

def build_dense_network(num_actions: int, num_states: int):
    '''
    # TODO: hidden_layers and learning rate as arguments
    '''
    model = keras.models.Sequential([
        keras.layers.Dense(24,
                           activation='relu',
                           input_shape=(num_states,), # one-hot encoding
                           name='input'),
        keras.layers.Dense(24,
                           activation='relu'),
        keras.layers.Dense(num_actions,
                           name='output'),
        ])
    # Using Keras optimizers and not tf because of warnings when saving
    # tf optimizers apparently need to be recompiled upon loading, theyre not as convenient
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae']) # mean absolute error

    return model

class DQNAgent:
    '''
    Attempt to write an agent with keras tensorflow API
    '''
    # TODO: change num_states for state_size/shape
    def __init__(self, num_actions: int, num_states: int, gamma: float = 0.9, pretrained_model: keras.models.Sequential = None):
        
        if pretrained_model is not None:
            self._q_impl = pretrained_model
            # pairs are returned with (batch_size=None, size)
            _, num_actions = pretrained_model.get_layer(name='output').output_shape
            _, num_states = pretrained_model.get_layer(name='input').input_shape
        else:
            self._q_impl = build_dense_network(num_actions, num_states)
        
        self._gamma = gamma
        self._num_actions = num_actions
        self._num_states = num_states
        self._memory = deque(maxlen=2000)

    def act(self, state: int):
        ''' Get either a greedy action '''
        return self.policy(state)

    def process_observation(self, state: int, action: int, reward: float, next_state: int, done: bool):
        ''' Store observation to train later in batches '''
        self._memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        ''' 're-fit' Q replaying random samples from memory '''
        if len(self._memory) <= batch_size:
            logger.debug('Should only happen a few times in the beggining')
            return
        minibatch = random.sample(self._memory, batch_size)
        X = []
        y = []
        # TODO: try to remove this for-loop
        # we probably want to keep _memory as the records of states/actions/rewards though
        for state, action, reward, next_state, done in minibatch:
            state, target_q = self._observation_to_train_data(state,
                                                              action,
                                                              reward,
                                                              next_state,
                                                              done)
            X.append(state)
            y.append(target_q)
        self._q_impl.fit(np.array(X), np.array(y), batch_size=batch_size, epochs=3,verbose=0)

    def _observation_to_train_data(self, state: int, action: int, reward: float, next_state: int, done: bool):
        ''' get states observations, rewards and action and return X, y for training '''
        target = reward
        if not done:
            target += self._gamma * self.V(next_state)
        target_q = self.Q(state)
        target_q[action] = target
        one_hot_state = one_hot(self._num_states, state)
        return one_hot_state, target_q
    
    def Q(self, state: int):
        ''' value of any taken action in a given state and playing perfectly onwards '''
        one_hot_state = one_hot(self._num_states, state)
        return self._q_impl.predict(one_hot_state[np.newaxis])[0]
    
    def policy(self, state: int):
        ''' optimal greedy action for a state '''
        return np.argmax(self.Q(state))
    
    def V(self, state: int):
        ''' value of being in a given state (and playing perfectly onwards) '''
        return np.max(self.Q(state))
    
    def print_q_map(self):
        q_table = pd.DataFrame([self.Q(s) for s in range(self._num_states)]).transpose()
        print(q_table)
    
    def save(self, file_path: str = 'dqn_agent.h5'):
        ''' Save trained agent/model '''
        if not file_path.endswith('.h5'):
            file_path += '.h5'
        logger.info('Saving agent to: ' + file_path)
        self._q_impl.save(file_path)
    
    @staticmethod
    def from_h5(file_path: str = 'dqn_agent.h5', gamma: float = 0.9):
        logger.info('Loading agent from: ' + file_path)
        model = keras.models.load_model(file_path)
        agent = DQNAgent(None, None, gamma=gamma, pretrained_model=model)
        return agent

