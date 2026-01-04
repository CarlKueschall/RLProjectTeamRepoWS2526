import numpy as np

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
