import numpy as np

def one_hot(size, idx):
    res = np.zeros(size)
    res[idx] = 1
    return res

class StateSerializer:
    '''
    Class that handles different types of states for the agent
    # Default:
        For multi-dimensional states, states are fed and handled as-is
    # Non-default use cases (Factory method provided):
        For discrete states, we use one_hot_encoding and num_states
    '''
    def __init__(self, state_shape: tuple):
        self._state_shape = state_shape
        self._one_hot = False
    
    def serialize(self, state):
        if self._one_hot:
            return one_hot(self._state_shape[0], state)
        return state
    
    def deserialize(self, state):
        if self._one_hot:
            return np.argmax(state)
        return state
    
    @property
    def shape(self):
        return self._state_shape
    
    @staticmethod
    def from_num_states(num_states: int):
        handler = StateSerializer(state_shape=(num_states,))
        handler._one_hot = True
        return handler
    
