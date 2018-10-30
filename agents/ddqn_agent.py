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
# 3. https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
# 4. https://arxiv.org/pdf/1509.06461.pdf
#
# [1] and [2] basically implement the same thing
# [1] one uses/explains 2 models for Q target_Q & Q for stability
# [2] implements it in github, but does not explain it in the article
# [3] Minh 2015 is baseline DQN
# [4] Van Hasselt 2015, proposes using online model for greedy policy and target model for evaluation
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

class DDQNAgent:
    '''
    Attempt to write a basic DQN agent with keras tensorflow API
    states need to be properly conditioned for the agent before being used
    this agent implements DDQN as per [4. Van Hasselt 2015] which is an improvement
    of [3. Minh 2015] 'Algorithm 1: deep Q-learning with experience replay'
    by using the online network for greedy policy, but target network to evaluate its value
    '''
    def __init__(self, num_actions: int, state_shape: tuple, gamma: float = 0.9,
                 target_update_freq: int = 200,
                 pretrained_model: keras.models.Sequential = None) -> None:
        if pretrained_model is not None:
            self._q_impl = pretrained_model
        else:
            self._q_impl = build_dense_network(num_actions, state_shape)
        
        # Start target network = to online network
        self._target_q_impl = keras.models.Sequential.from_config(self._q_impl.get_config())
        self._update_target_model()
        
        self._target_update_freq = target_update_freq
        self._gamma = gamma
        self._memory = deque(maxlen=2000)

    def act(self, state: np.ndarray) -> int:
        ''' Get greedy action '''
        return self.policy(state)[0]

    def process_observation(self, state: np.ndarray, action: int, reward: float,
                            next_state: np.ndarray, done: bool) -> None:
        ''' Store observation to train later in batches '''
        self._memory.append((state, action, reward, next_state, done))

    def train(self, step_num: int, batch_size: int = 64, epochs: int = 1) -> None:
        ''' 're-fit' Q replaying random samples from memory '''
        if len(self._memory) <= batch_size:
            logger.debug('Should only happen a few times in the beggining')
            return
        
        minibatch = random.sample(self._memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states, target_qs = self._observations_to_train_data(np.array(states),
                                                             np.array(actions),
                                                             np.array(rewards),
                                                             np.array(next_states),
                                                             np.array(dones))
        
        result = self._q_impl.fit(states, target_qs, batch_size=batch_size, epochs=epochs, verbose=0)
        
        if step_num % self._target_update_freq == 0:
            self._update_target_model()
        
        return result
    
    def _update_target_model(self):
        self._target_q_impl.set_weights(self._q_impl.get_weights())

    def _observations_to_train_data(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                                    next_states: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' get states observations, rewards and action and return X, y for training '''
        assert(states.shape == next_states.shape)
        assert(actions.shape == rewards.shape == dones.shape)
        assert(len(states) == len(actions))
        
        targets = rewards + np.logical_not(dones) * self._gamma * self.V(next_states, use_target=True)
        target_qs = self.Q(states, use_target=False)
        target_qs[np.arange(len(target_qs)), actions] = targets
        return states, target_qs
    
    def Q(self, states: np.ndarray, use_target: bool = False) -> np.ndarray:
        ''' value of any taken action in a batch states and playing perfectly onwards '''
        if len(states.shape) ==  1:
            # we're evaluating a single example -> make batch_size = 1
            states = states[np.newaxis]
        
        if use_target:
            # This happens during training/value evaluation according to [4]
            self._target_q_impl.predict(states)
        
        # This happens during greedy policy evaluation according to [4]
        return self._q_impl.predict(states)
    
    def policy(self, states: np.ndarray, use_target: bool = False) -> int:
        ''' optimal greedy action for a batch of states '''
        return np.argmax(self.Q(states, use_target), axis=1) # axis=0 is batch axis
    
    def V(self, states: np.ndarray, use_target: bool = False) -> float:
        ''' value of being in a batch of states (and playing perfectly onwards) '''
        return np.max(self.Q(states, use_target), axis=1) # axis=0 is batch axis
    
    def save(self, file_path: str = 'dqn_agent.h5') -> None:
        ''' Save online trained model to .h5 file'''
        if not file_path.endswith('.h5'):
            file_path += '.h5'
        logger.info('Saving agent to: ' + file_path)
        self._q_impl.save(file_path)
    
    @staticmethod
    def from_h5(file_path: str = 'dqn_agent.h5', gamma: float = 0.9) -> 'DDQNAgent':
        ''' Load trained model from .h5 file'''
        logger.info('Loading agent from: ' + file_path)
        model = keras.models.load_model(file_path)
        agent = DDQNAgent(None, None, gamma=gamma, pretrained_model=model)
        return agent

