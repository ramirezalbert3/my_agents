from collections import deque
from typing import Tuple
import random
import numpy as np
from gym import logger
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

def build_dense_network(num_actions: int, state_shape: tuple, hidden_layers: list = [24, 24]):
    '''
    # TODO: optimizer and losses as arguments
    '''
    
    model = keras.models.Sequential()
    
    for idx, val in enumerate(hidden_layers):
        if idx == 0:
            model.add(keras.layers.Dense(val,
                                         activation='relu',
                                         input_shape=state_shape,
                                         name='input'))
        else:
            model.add(keras.layers.Dense(val,
                                         activation='relu',
                                         name='hidden_layer_' + str(idx)))
    
    model.add(keras.layers.Dense(num_actions,
                           name='output'))

    # Using Keras optimizers and not tf because of warnings when saving
    # tf optimizers apparently need to be recompiled upon loading, theyre not as convenient
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae']) # mean absolute error

    return model

class DQNAgent:
    '''
    Attempt to write an agent with keras tensorflow API
    states need to be properly conditioned for the agent before being used
    '''
    def __init__(self, num_actions: int, state_shape: tuple, gamma: float = 0.9, pretrained_model: keras.models.Sequential = None) -> None:
        if pretrained_model is not None:
            self._q_impl = pretrained_model
        else:
            self._q_impl = build_dense_network(num_actions, state_shape)
        
        self._gamma = gamma
        self._memory = deque(maxlen=2000)

    def act(self, state: np.ndarray) -> int:
        ''' Get either a greedy action '''
        return self.policy(state)

    def process_observation(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        ''' Store observation to train later in batches '''
        self._memory.append((state, action, reward, next_state, done))

    def train(self, batch_size: int = 64, epochs: int = 3) -> None:
        ''' 're-fit' Q replaying random samples from memory '''
        if len(self._memory) <= batch_size:
            logger.debug('Should only happen a few times in the beggining')
            return
        minibatch = random.sample(self._memory, batch_size)
        X = []
        y = []
        # Tried to remove the for loop with zip(*minibatch) and
        # transforming observations when appending them, it was slower
        for state, action, reward, next_state, done in minibatch:
            state, target_q = self._observation_to_train_data(state,
                                                              action,
                                                              reward,
                                                              next_state,
                                                              done)
            X.append(state)
            y.append(target_q)
        self._q_impl.fit(np.array(X), np.array(y), batch_size=batch_size, epochs=epochs,verbose=0)

    def _observation_to_train_data(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Tuple[np.ndarray, list]:
        ''' get states observations, rewards and action and return X, y for training '''
        target = reward
        if not done:
            target += self._gamma * self.V(next_state)
        target_q = self.Q(state)
        target_q[action] = target
        return state, target_q
    
    def Q(self, state: np.ndarray) -> np.ndarray:
        ''' value of any taken action in a given state and playing perfectly onwards '''
        # TODO: after state-decoupling, get Q and others to work with batches too,
        #       then, review all these [np.newaxis] and [0]
        return self._q_impl.predict(state[np.newaxis])[0]
    
    def policy(self, state: np.ndarray) -> int:
        ''' optimal greedy action for a state '''
        return np.argmax(self.Q(state))
    
    def V(self, state: np.ndarray) -> float:
        ''' value of being in a given state (and playing perfectly onwards) '''
        return np.max(self.Q(state))
    
    def save(self, file_path: str = 'dqn_agent.h5') -> None:
        ''' Save trained model to .h5 file'''
        if not file_path.endswith('.h5'):
            file_path += '.h5'
        logger.info('Saving agent to: ' + file_path)
        self._q_impl.save(file_path)
    
    @staticmethod
    def from_h5(file_path: str = 'dqn_agent.h5', gamma: float = 0.9) -> 'DQNAgent':
        ''' Load trained model from .h5 file'''
        logger.info('Loading agent from: ' + file_path)
        model = keras.models.load_model(file_path)
        agent = DQNAgent(None, None, gamma=gamma, pretrained_model=model)
        return agent

