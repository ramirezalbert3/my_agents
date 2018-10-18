import random
import numpy as np
from sklearn.neural_network import MLPRegressor


class Agent:
    '''
    Attempt to write an agent that uses
    - gather observations during an epoch
    - at the end of each epoch, 'partial_fit' state/action pairs with rewards
    '''
    def __init__(self, num_actions: int):
        self._num_actions = num_actions
        self._q = None
        self._observations = ([], [])
    
    def act(self, epsilon: float, state: tuple):
        '''
        Get either an exploration or greedy action
        '''
        if random.random() <= epsilon or self._q is None:
            return random.randint(0, self._num_actions - 1)
        
        values = [self._q.predict([(state, action)])
                  for action in range(self._num_actions)]
        return np.argmax(values)
    
    def store_observation(self, state: tuple, action: int, reward: float):
        X, y = self._observations
        X.append((state, action))
        y.append(reward)
        assert len(X) == len(y)
        self._observations = (X, y)
    
    def train(self):
        '''
        after an epoch 're-fit' Q with observations
        '''
        # TODO: maybe add a round of initial values to all new states that havent seen an action?
        X, y = self._observations # [(state, action)], [reward]
        if self._q is None:
            self._q = MLPRegressor(hidden_layer_sizes=[20, 20])
        self._q.partial_fit(X, y)
        self._observations = ([], []) # reset observations
