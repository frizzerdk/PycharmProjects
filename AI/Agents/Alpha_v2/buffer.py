import numpy as np
import random
from collections import deque
import pickle
import os
class MemoryBuffer:
    """
    MemoryBuffer is a class designed to store and manage a memory buffer for reinforcement learning algorithms,
    specifically for those that utilize experience replay.
    """

    def __init__(self, size):
        """
        Initialize the MemoryBuffer with a given size.

        :param size: Maximum size of the memory buffer.
        """
        assert isinstance(size, int) and size > 0, "Size should be a positive integer."
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self._len = 0

    def sample(self, batch_size):
        """
        Samples a random batch from the replay memory buffer.

        :param batch_size: Size of the batch to sample.
        :return: Tuple containing state, action, reward, next state, done, and truncated arrays.
        """
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size should be a positive integer."
        batch = []
        count = min(batch_size, self._len)
        batch = random.sample(self.buffer, count)

        return self._convert_to_arrays(batch)

    def sample_recent_bias(self, batch_size, recent_proportion=0.05):
        """
        Samples a random batch from the replay memory buffer, with a bias towards the recent samples.

        :param batch_size: Size of the batch to sample.
        :param recent_proportion: Proportion of recent samples vs random samples in the final batch.
        :return: Tuple containing state, action, reward, next state, done, and truncated arrays.
        """
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size should be a positive integer."
        assert isinstance(recent_proportion, (int, float)) and 0 <= recent_proportion <= 1, "Recent proportion should be a number between 0 and 1."
        batch = []
        count = min(batch_size, self._len)
        recent_count = int(count * recent_proportion)
        recent_count = min(recent_count, self._len)
        random_count = count - recent_count
        if recent_count > 0:
            batch = [self.buffer[-i] for i in range(1, recent_count + 1)]
        if random_count > 0:
            random_batch = random.sample(self.buffer, random_count)
            batch += random_batch

        return self._convert_to_arrays(batch)

    def _convert_to_arrays(self, batch):
        """
        Converts a batch of experiences into numpy arrays.

        :param batch: List of experiences.
        :return: Tuple containing state, action, reward, next state, done, and truncated arrays.
        """
        s1_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s2_arr = np.float32([arr[3] for arr in batch])
        done_arr = np.float32([arr[4] for arr in batch])
        truncated_arr = np.float32([arr[5] for arr in batch])

        return s1_arr, a_arr, r_arr, s2_arr, done_arr, truncated_arr

    def len(self):
        """
        Returns the current length of the memory buffer.

        :return: Integer representing the current length of the buffer.
        """
        return self._len

    def add(self, s1, a, r, s2, done=False, truncated=False, priority=None):
        """
        Adds a particular transaction in the memory buffer.

        :param s1: Current state.
        :param a: Action taken.
        :param r: Reward received.
        :param s2: Next state.
        :param done: Whether the episode is done.
        :param truncated: Whether the episode is truncated.
        :param priority: Priority of the transition (for priority-based replay memory).
        """
        if priority is not None:
            transition = (s1, a, r, s2, done, truncated, priority)
        else:
            transition = (s1, a, r, s2, done, truncated)

        self._len += 1
        if self._len > self.maxSize:
            self._len = self.maxSize
        self.buffer.append(transition)

    def save_to_file(self, filename, directory='./Experience/'):
        """
        Saves the memory buffer to a file.

        :param filename: Name of the file to save the buffer.
        :param directory: Directory to save the file. Defaults to './Experience/'.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = open(directory + filename, 'wb')
        pickle.dump(self.buffer, file)
        file.close()

    def load_from_file(self, filename, directory='./Experience/'):
        """
        Loads a memory buffer from a file.

        :param filename: Name of the file to load the buffer from.
        :param directory: Directory to load the file from. Defaults to './Experience/'.
        """
        try:
            file = open(directory + filename, 'rb')
            buffer = pickle.load(file)
            self.buffer.extend(buffer)
            self._len = min(self._len + len(buffer), self.maxSize)
            file.close()
        except FileNotFoundError:
            print("File not found.")

class PrioritizedMemoryBuffer(MemoryBuffer):
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_increment=0.001):
        """
        Initialize a prioritized experience replay memory buffer.

        :param size: Maximum size of the buffer.
        :param alpha: Controls the impact of priorities on the sampling process (0 <= alpha <= 1).
        :param beta_start: Initial value of the importance sampling weight annealing factor (0 <= beta_start <= 1).
        :param beta_increment: Increment value for the beta parameter after each update.
        """
        super().__init__(size)
        self.priorities = deque(maxlen=size)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

    def add(self, s1, a, r, s2, done=False, truncated=False, td_error=None):
        """
        Adds a particular transaction in the memory buffer.

        :param s1: Current state.
        :param a: Action taken.
        :param r: Reward received.
        :param s2: Next state.
        :param done: Whether the episode is done.
        :param truncated: Whether the episode is truncated.
        :param td_error: Temporal-difference error for the transition.
        """
        transition = (s1, a, r, s2, done, truncated)
        priority = self._calculate_priority(td_error) if td_error is not None else max(self.priorities, default=1)
        super().add(*transition, priority)

    def _calculate_priority(self, td_error):
        """
        Calculate the priority for a given TD error.

        :param td_error: Temporal-difference error.
        :return: Priority value.
        """
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        return priority

    def sample(self, batch_size, beta=0.6):
        if self._len == 0:
            raise RuntimeError("The replay memory is empty. Add some transitions before sampling.")

        if self._len <= batch_size:
            print("The replay memory size is less than the requested batch size. "
                  "Returning all available samples.")
            batch_size = self._len

        # Extract priorities directly from the buffer
        priorities = np.array([transition[-1] for transition in self.buffer])

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(np.arange(len(probs)), size=batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]
        is_weights = self._calculate_importance_sampling_weights(probs, indices)

        # Remove the priority values from the sampled transitions
        batch_no_priority = [(s1, a, r, s2, done, truncated) for s1, a, r, s2, done, truncated, _ in batch]

        return self._convert_to_arrays(batch_no_priority), is_weights, indices

    def _calculate_importance_sampling_weights(self, probs, indices):
        """
        Calculate the importance sampling weights for the sampled experiences.

        :param probs: Probability distribution of the experiences.
        :param indices: Indices of the sampled experiences.
        :return: Importance sampling weights.
        """
        is_weights = (self._len * probs[indices]) ** -self.beta
        is_weights /= is_weights.max()
        return is_weights

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of the experiences using new TD errors.

        :param indices: Indices of the experiences to update.
        :param td_errors: New TD errors for the experiences.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = self._calculate_priority(td_error)
            s1, a, r, s2, done, truncated = self.buffer[idx][:6]  # Unpack the transition tuple without the priority
            self.buffer[idx] = (s1, a, r, s2, done, truncated, priority)  # Update the tuple with the new priority

    def increase_beta(self):
            """
        Increase the beta parameter by the specified beta_increment value.
        The beta parameter is used for importance sampling weights annealing.
        """
            self.beta = min(1.0, self.beta + self.beta_increment)

    # All other methods from the MemoryBuffer class are inherited and don't need to be redefined.
