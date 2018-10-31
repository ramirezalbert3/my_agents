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
# 1b. https://www.youtube.com/watch?v=ba_l8IKoMvU
# 2.  https://flyyufelix.github.io/2017/10/24/distributional-bellman.html
# 2b. https://github.com/flyyufelix/C51-DDQN-Keras
# 3. https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
'''

def build_distributional_network(num_actions: int, state_shape: tuple, num_atoms: int, hidden_layers: list = [24, 24]):
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
    this implements the target_Q vs online_Q as per [1] and [3]
    '''
    
    class Distribution:
        def __init__(self, v_min: float, v_max: float, num_atoms: int):
            self.v_min = v_min         # env specific: this is for regular cartpole
            self.v_max = v_max         # env specific: this is for regular cartpole
            self.num_atoms = num_atoms # hyperparameter
            self.delta_z = (v_max - v_min) / float(num_atoms-1)
            self.z = np.array([v_min + i * self.delta_z for i in range(num_atoms)])
        
        def project_to_distribution(self, values):
            ''' project values to distribution (Vmin, Vmax, num_atoms)'''
            Tz = np.clip(values, self.v_min, self.v_max)
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = np.floor(bj).astype(int), np.ceil(bj).astype(int)
            return bj, m_l, m_u
        

    def __init__(self, num_actions: int, state_shape: tuple, v_min: float, v_max: float, num_atoms: int = 21,
                 gamma: float = 0.9, target_update_freq: int = 200,
                 pretrained_model: keras.models.Model = None) -> None:
        if pretrained_model is not None:
            self._z_impl = pretrained_model
        else:
            self._z_impl = build_distributional_network(num_actions, state_shape, num_atoms)
        
        # Start target network = to online network
        self._target_z_impl = keras.models.Model.from_config(self._q_impl.get_config())
        self._update_target_model()
        
        self._target_update_freq = target_update_freq
        self._gamma = gamma
        self._memory = deque(maxlen=2000)
        
        self._distribution = DistributionalAgent.Distribution(v_min=v_min, v_max=v_max, num_atoms=num_atoms)

    def act(self, state: np.ndarray) -> int:
        ''' Get greedy action '''
        return self.policy(state)[0]

    def process_observation(self, state: np.ndarray, action: int, reward: float,
                            next_state: np.ndarray, done: bool) -> None:
        ''' Store observation to train later in batches '''
        self._memory.append((state, action, reward, next_state, done))

    def train(self, step_num: int, batch_size: int = 64, epochs: int = 3) -> None:
        ''' 're-fit' Q replaying random samples from memory '''
        if len(self._memory) <= batch_size:
            logger.warning('Cant train on an empty memory, warm-up the agent!')
            return
        
        minibatch = random.sample(self._memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states, target_zs = self._observations_to_train_data(np.array(states),
                                                             np.array(actions),
                                                             np.array(rewards),
                                                             np.array(next_states),
                                                             np.array(dones))
        
        result = self._z_impl.fit(states, target_zs, batch_size=batch_size, epochs=epochs, verbose=0)
        
        if step_num % self._target_update_freq == 0:
            self._update_target_model()
        
        return result
    
    def _update_target_model(self):
        self._target_z_impl.set_weights(self._z_impl.get_weights())
        
    def _observations_to_train_data(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                                    next_states: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' get states observations, rewards and action and return X, y for training '''
        assert(states.shape == next_states.shape)
        assert(actions.shape == rewards.shape == dones.shape)
        assert(len(states) == len(actions))
        
        next_zs = self.Z(next_states)
        batch_size, num_actions, num_atoms = len(actions), len(next_zs), self._distribution.num_atoms
        
        # Vectorization
        rew_mat = np.repeat(rewards, num_atoms).reshape((batch_size, num_atoms), order='C')
        done_mat = np.repeat(dones, num_atoms).reshape((batch_size, num_atoms), order='C')
        z_mat = np.repeat(self._distribution.z, batch_size).reshape((batch_size, num_atoms), order='F')
        targets = rew_mat + np.logical_not(done_mat) * self._gamma * z_mat # (batch_size, num_atoms)
        bj, m_l, m_u = self._distribution.project_to_distribution(targets)
        
        m_prob = np.zeros((num_actions, batch_size, num_atoms))
        
        # TODO: not sure if its possible to vectorize a little bit
        #       maybe we can vectorize a bit if there are no 'repeated' X (if len(set(X))==len(X))
        #       X being actions, m_l and m_u --> we can do these computations in a matrix
        for i in range(batch_size):
            for j in range(num_atoms):
                m_prob[actions[i], i, m_l[i, j]] += ((m_u - bj) * (done_mat + np.logical_not(done_mat) * next_zs[actions, np.arange(batch_size)]))[i, j]
                m_prob[actions[i], i, m_u[i, j]] += ((bj - m_l) * (done_mat + np.logical_not(done_mat) * next_zs[actions, np.arange(batch_size)]))[i, j]
        m_prob = np.vsplit(m_prob, num_actions)  # split into n_actions-long list
        m_prob = [np.squeeze(x) for x in m_prob] # remove 1-dims leftovers, keep as list
        return states, m_prob
    
    def Z(self, states: np.ndarray) -> np.ndarray:
        ''' distributions for actions in a batch of states '''
        if len(states.shape) ==  1:
            # we're evaluating a single example -> make batch_size = 1
            states = states[np.newaxis]
        return np.array(self._target_z_impl.predict(states)) # (num_actions, batch_size, num_atoms)
    
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
        ''' Save online trained model to .h5 file'''
        if not file_path.endswith('.h5'):
            file_path += '.h5'
        logger.info('Saving agent to: ' + file_path)
        self._z_impl.save(file_path)
    
    @staticmethod
    def from_h5(file_path: str = 'dqn_agent.h5', gamma: float = 0.9,
                target_update_freq: int = 200) -> 'DQNAgent':
        ''' Load trained model from .h5 file'''
        logger.info('Loading agent from: ' + file_path)
        model = keras.models.load_model(file_path)
        agent = DQNAgent(None, None, gamma=gamma,
                         target_update_freq=target_update_freq, pretrained_model=model)
        return agent

