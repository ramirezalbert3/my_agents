import numpy as np
'''
# This is basically a clean up of the implementations explained below
# TODO: actually read through to understand the code
1.  https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
1b. https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb
2.  https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
3.  https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py
4.  https://arxiv.org/pdf/1511.05952.pdf
'''

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
    
    def add(self, priority, data):
        ''' Add priority score in the sumtree leaf and add the experience in data'''
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity: 
            # If we're above the capacity, you go back to first index (we overwrite)
            # TODO: double check this, we overwrite old values right?
            self.data_pointer = 0
        
    def update(self, tree_index: int, priority):
        ''' Update the leaf priority score and propagate the change through tree '''
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, v):
        '''
        Get the leaf_index and priority value of a leaf and experience
        associated with an index
        '''
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):
                # If we reach bottom, end the search
                leaf_index = parent_index
                break
            else:
                # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]
    
    @property
    def max_priority(self):
        return np.max(self.tree[-self.capacity:])
    
    @property
    def min_priority(self):
        return np.min(self.tree[-self.capacity:])

class PrioritizedMemory:
    '''
    Prioritized replay memory implemented with a sum-tree according to [4]
    TODO: priorities are clipped with 'max_error'
    hyperparameters defaulted and kept constant according to [3] (Dopamine by Google)
    '''
    def __init__(self, capacity: int, alpha: float = 0.5,
                 beta: float = 0.5, max_error: float = 1.):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_error = max_error  # clipped abs error
    
    def store(self, experience):
        ''' Store a new experience in our tree initially with a score of max_prority '''
        max_priority = self.tree.max_priority
        if max_priority == 0:
            max_priority = self.max_error
        self.tree.add(max_priority, experience)
    
    def sample(self, batch_size: int):
        '''
        1. Sample a minibatch of k size
        2. Divide the range [0, priority_total] k ranges.
        3. Uniformly sample a value from each range
        4. Search in the sumtree, retrieve the experience where priority score corresponds to sample values
        5. IS (importance-sampling) weights for each minibatch element
        '''
        batch = []
        
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_ISWeights = np.empty((batch_size, 1), dtype=np.float32)
        
        priority_segment = self.tree.total_priority / batch_size
        
        p_min = self.tree.min_priority / self.tree.total_priority
        max_weight = (p_min * batch_size) ** (-self.beta)
        
        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            
            sampling_probabilities = priority / self.tree.total_priority
            
            b_ISWeights[i, 0] = np.power(batch_size * sampling_probabilities, -self.beta) / max_weight
            b_idx[i]= index
            batch.append([data])
        # TODO: double-check (and rework if needed) the returns and their types
        return b_idx, batch, b_ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        ''' Update the priorities on the tree '''
        abs_errors = np.absolute(abs_errors) + 1e-5 # avoid zero probabilities
        clipped_errors = np.minimum(abs_errors, self.max_error)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
