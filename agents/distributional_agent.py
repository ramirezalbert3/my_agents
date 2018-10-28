from collections import deque
from typing import Tuple
import random
import numpy as np
from gym import logger
from tensorflow import keras

'''
# References
# 0.  https://keras.io/getting-started/functional-api-guide/
# 1.  https://arxiv.org/abs/1707.06887
# 2.  https://flyyufelix.github.io/2017/10/24/distributional-bellman.html
# 2b. https://github.com/flyyufelix/C51-DDQN-Keras
'''

def build_distributional_network(num_actions: int, state_shape: tuple, num_atoms: int = 10, hidden_layers: list = [24, 24]):
    inputs = keras.layers.Input(shape=state_shape, name='input') 
    
    for idx, val in enumerate(hidden_layers):
        if idx == 0:
            hidden = keras.layers.Dense(val, activation='relu', name='hidden_layer_' + str(idx))(inputs)
        else:
            hidden = keras.layers.Dense(val, activation='relu', name='hidden_layer_' + str(idx))(hidden)

    distributions = [keras.layers.Dense(num_atoms, activation='softmax', name='output_' + str(idx))(hidden)
                     for idx in range(num_actions)]

    model = keras.Model(inputs=inputs, outputs=distributions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy')

    return model

class DistributionalAgent:
    '''
    Attempt to write Distributional agent with keras tensorflow API
    states need to be properly conditioned for the agent before being used
    this does not implement either the target_Q vs Q or DDQN
    '''
    
    class Distribution:
        def __init__(self, v_min: float, v_max: float, num_atoms: int = 10):
            # Distributional parameters
            self.v_min = v_min          # env specific: this is for regular cartpole
            self.v_max = v_max         # env specific: this is for regular cartpole
            self.num_atoms = num_atoms # hyperparameter
            self.delta_z = (v_max - v_min) / float(num_atoms-1)
            self.z = np.array([v_min + i * self.delta_z for i in range(num_atoms)])
        
        def project_to_distribution(self, values):
            ''' locate values in distribution Vmin and Vmax'''
            Tz = np.clip(values, self.v_min, self.v_max)
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = np.floor(bj).astype(int), np.ceil(bj).astype(int)
            return bj, m_l, m_u
        

    def __init__(self, num_actions: int, state_shape: tuple, v_min: float, v_max: float, num_atoms: int = 10,
                 gamma: float = 0.9, pretrained_model: keras.models.Model = None) -> None:
        if pretrained_model is not None:
            self._z_impl = pretrained_model
        else:
            self._z_impl = build_distributional_network(num_actions, state_shape)
        
        self._gamma = gamma
        self._memory = deque(maxlen=2000)
        
        self._distribution = DistributionalAgent.Distribution(v_min=0, v_max=200, num_atoms=10)

    def act(self, state: np.ndarray) -> int:
        ''' Get either a greedy action '''
        return self.policy(state)[0]

    def process_observation(self, state: np.ndarray, action: int, reward: float,
                            next_state: np.ndarray, done: bool) -> None:
        ''' Store observation to train later in batches '''
        self._memory.append((state, action, reward, next_state, done))

    def train(self, batch_size: int = 64, epochs: int = 3) -> None:
        ''' 're-fit' Q replaying random samples from memory '''
        if len(self._memory) <= batch_size:
            logger.debug('Should only happen a few times in the beggining')
            return
        
        minibatch = random.sample(self._memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states, target_zs = self._observations_to_train_data(np.array(states),
                                                             np.array(actions),
                                                             np.array(rewards),
                                                             np.array(next_states),
                                                             np.array(dones))
        
        self._z_impl.fit(states, target_zs, batch_size=batch_size, epochs=epochs, verbose=0)

    def _observations_to_train_data(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                                    next_states: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' get states observations, rewards and action and return X, y for training '''
        assert(states.shape == next_states.shape)
        assert(actions.shape == rewards.shape == dones.shape)
        assert(len(states) == len(actions))
        
        batch_size = len(actions)
        values = rewards + np.logical_not(dones) * self._gamma * self.V(next_states)
        bj, m_l, m_u = self._distribution.project_to_distribution(values)
        target_zs = self.Z(states)
        
        # BUG: second target_zs should not use m_l or m_u, but EVERY atom
        target_zs[actions, np.arange(batch_size), m_l] = target_zs[actions, np.arange(batch_size), m_l] * np.logical_not(dones) * (m_u - bj) + dones * (m_u - bj)
        target_zs[actions, np.arange(batch_size), m_u] = target_zs[actions, np.arange(batch_size), m_u] * np.logical_not(dones) * (bj - m_l) + dones * (bj - m_l)
        #
        target_zs = np.vsplit(target_zs, len(target_zs)) # split into n_actions-long list
        target_zs = [np.squeeze(z) for z in target_zs] # remove 1-dims leftovers, keep as list
        return states, target_zs
    
    def Z(self, states: np.ndarray) -> np.ndarray:
        ''' distributions for actions in a batch of states '''
        if len(states.shape) ==  1:
            # we're evaluating a single example -> make batch_size = 1
            states = states[np.newaxis]
        return np.array(self._z_impl.predict(states)) # (num_actions, batch_size, num_atoms)
    
    def Q(self, states: np.ndarray) -> np.ndarray:
        ''' value of any taken action in a batch of states and playing perfectly onwards '''
        z = self.Z(states)
        if len(states.shape) ==  1:
            # TODO: should not be necessary to run twice
            # we're evaluating a single example -> make batch_size = 1
            states = states[np.newaxis]
        q_shape = len(states), len(z) # (batch_size, num_actions)
        z = np.vstack(z) # (batch_size * num_actions, num_atoms)
        q = np.sum(np.multiply(z, self._distribution.z), axis=1).reshape(q_shape, order='F')
        return q
    
    def policy(self, states: np.ndarray) -> int:
        ''' optimal greedy action for a batch of states '''
        return np.argmax(self.Q(states), axis=1) # axis=0 is batch axis
    
    def V(self, states: np.ndarray) -> float:
        ''' value of being in a batch of states (and playing perfectly onwards) '''
        return np.max(self.Q(states), axis=1) # axis=0 is batch axis
    
    def save(self, file_path: str = 'dqn_agent.h5') -> None:
        ''' Save trained model to .h5 file'''
        if not file_path.endswith('.h5'):
            file_path += '.h5'
        logger.info('Saving agent to: ' + file_path)
        self._z_impl.save(file_path)
    
    @staticmethod
    def from_h5(file_path: str = 'dqn_agent.h5', gamma: float = 0.9) -> 'DQNAgent':
        ''' Load trained model from .h5 file'''
        logger.info('Loading agent from: ' + file_path)
        model = keras.models.load_model(file_path)
        agent = DQNAgent(None, None, gamma=gamma, pretrained_model=model)
        return agent

