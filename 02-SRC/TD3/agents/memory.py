"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import numpy as np


class SumTree:
    """
    Sum tree data structure for efficient priority-based sampling.
    Supports O(log n) sampling and O(log n) priority updates.

    Used by PrioritizedReplayBuffer for Prioritized Experience Replay (PER).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample on leaf node given a value s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return total priority sum."""
        return self.tree[0]

    def add(self, priority, data):
        """Add data with given priority."""
        idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """Update priority of existing node."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get data, priority, and tree index for a value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedMemory:
    """
    Prioritized Experience Replay (PER) buffer.

    Samples transitions with probability proportional to TD-error priority.
    Uses importance sampling weights to correct for bias.

    Key hyperparameters:
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling correction (annealed from beta_start to 1.0)
        epsilon: Small constant to ensure non-zero priority

    Reference: Schaul et al., "Prioritized Experience Replay" (2015)
    """
    def __init__(self, max_size=500000, alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=1e-5):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames  # Frames to anneal beta to 1.0
        self.epsilon = epsilon  # Small constant for non-zero priority
        self.frame = 0
        self.max_priority = 1.0  # Track max priority for new transitions

    def _get_beta(self):
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        return beta

    def add_transition(self, transition):
        """Add transition with max priority (will be corrected on first sample)."""
        # New transitions get max priority to ensure they're sampled at least once
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch=1):
        """
        Sample batch of transitions with priority-based probability.

        Returns:
            transitions: numpy array of sampled transitions
            indices: tree indices for priority updates
            is_weights: importance sampling weights for loss correction
        """
        if batch > self.tree.n_entries:
            batch = self.tree.n_entries

        indices = np.zeros(batch, dtype=np.int32)
        priorities = np.zeros(batch, dtype=np.float64)
        transitions = []

        # Divide priority range into segments for stratified sampling
        segment = self.tree.total() / batch

        beta = self._get_beta()
        self.frame += 1

        for i in range(batch):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            # Handle edge case where priority is 0 (shouldn't happen but be safe)
            if priority == 0:
                priority = self.epsilon

            priorities[i] = priority
            indices[i] = idx

            # Convert stored data back to list format
            if isinstance(data, tuple):
                transitions.append(list(data))
            else:
                transitions.append(data)

        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta) / max_w
        sampling_probs = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -beta)
        is_weights /= is_weights.max()  # Normalize by max weight

        return np.asarray(transitions, dtype=object), indices, is_weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.

        Args:
            indices: Tree indices from sample()
            td_errors: Absolute TD errors from training
        """
        for idx, td_error in zip(indices, td_errors):
            # Priority = (|TD_error| + epsilon)^alpha
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, np.abs(td_error) + self.epsilon)

    def __len__(self):
        return self.tree.n_entries

    def get_stats(self):
        """
        Get PER buffer statistics for monitoring.

        Returns:
            dict: Statistics including beta, max_priority, frame count
        """
        return {
            'beta': self._get_beta(),
            'max_priority': self.max_priority,
            'frame': self.frame,
            'total_priority': self.tree.total(),
            'n_entries': self.tree.n_entries,
            'alpha': self.alpha,
        }


# experience replay buffer class
# we only need add, sample and getall methods
#
class Memory():
    def __init__(self, max_size=500000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        #########################################################
        # This is the main method to add new transitions to the buffer
        #########################################################
        # If the buffer is empty, we create a blank buffer
        # If the buffer is not empty, we update the current index
        # and the size of the buffer
        #########################################################
        # If the buffer is full, we overwrite the oldest transition
        #########################################################
        # If the buffer is not full, we add the new transition
        # to the current index, simple as that.
        #########################################################
        if self.size == 0:
            blank_buffer = []
            for i in range(self.max_size):
                blank_buffer.append(np.asarray(transitions_new, dtype=object))
            self.transitions = np.asarray(blank_buffer)

        updated_transition = np.asarray(transitions_new, dtype=object)
        for i in range(updated_transition.shape[0]):
            self.transitions[self.current_idx, i] = updated_transition[i]
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = self.current_idx + 1
        if self.current_idx >= self.max_size:
            self.current_idx = 0

    def sample(self, batch=1):
        #########################################################
        # This one is used to sample a batch of transitions from the buffer
        #########################################################
        # If the batch size is greater than the size of the buffer, we sample all the transitions
        #########################################################
        # Then if the batch size is less than the size of the buffer, we sample a batch of transition
        if batch > self.size:
            batch = self.size

        indices_list = []
        for i in range(batch):
            while True:
                rand_ind = np.random.randint(0, self.size)
                if rand_ind not in indices_list:
                    indices_list.append(rand_ind)
                    break

        sampled_transitions = []
        for idx in indices_list:
            sampled_transition = []
            for col in range(self.transitions.shape[1]):
                sampled_transition.append(self.transitions[idx, col])
            sampled_transitions.append(sampled_transition)
        return np.asarray(sampled_transitions, dtype=object)

    def get_all_transitions(self):
        all_transitions = []
        for i in range(self.size):
            row_transitions = []
            for j in range(self.transitions.shape[1]):
                row_transitions.append(self.transitions[i, j])
            all_transitions.append(row_transitions)
        return np.asarray(all_transitions, dtype=object)

    def __len__(self):
        #dont remove
        return self.size
