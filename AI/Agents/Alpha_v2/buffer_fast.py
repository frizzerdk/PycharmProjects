import torch
import pickle
import os
import numpy as np

class MemoryBuffer:
    def __init__(self, size, state_shape, action_shape, device="cpu"):
        """
        Initialize a memory buffer.

        :param size: Maximum size of the buffer.
        :param state_shape: Shape of the state tensor.
        :param action_shape: Shape of the action tensor.
        :param device: Device to store the tensors on ("cpu" or "cuda").
        """
        self.maxSize = size
        self._len = 0
        self.device = device
        # Convert state_shape and action_shape to tuples if they are integers
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)

        self.s1 = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
        self.a = torch.zeros((size, *action_shape), dtype=torch.float32, device=device)
        self.r = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.s2 = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
        self.done = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.truncated = torch.zeros((size, 1), dtype=torch.float32, device=device)

    def add(self, s1, a, r, s2, done=False, truncated=False):
        """
        Adds a particular transaction in the memory buffer.

        :param s1: Current state.
        :param a: Action taken.
        :param r: Reward received.
        :param s2: Next state.
        :param done: Whether the episode is done.
        :param truncated: Whether the episode is truncated.
        """
        idx = self._len % self.maxSize

        self.s1[idx] = torch.tensor(s1, dtype=torch.float32, device=self.device)
        self.a[idx] = torch.tensor(a, dtype=torch.float32, device=self.device)
        self.r[idx] = torch.tensor(r, dtype=torch.float32, device=self.device)
        self.s2[idx] = torch.tensor(s2, dtype=torch.float32, device=self.device)
        self.done[idx] = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.truncated[idx] = torch.tensor(truncated, dtype=torch.float32, device=self.device)

        self._len += 1
        if self._len > self.maxSize:
            self._len = self.maxSize

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the memory buffer.

        :param batch_size: Size of the batch to sample.
        :return: A batch of transitions.
        """
        if self._len <= batch_size:
           # print("The replay memory size is less than the requested batch size. "
            #      "Returning all available samples.")
            batch_size = self._len

        indices = torch.randint(0, self._len, (batch_size,), device=self.device)

        s1_batch = self.s1[indices]
        a_batch = self.a[indices]
        r_batch = self.r[indices]
        s2_batch = self.s2[indices]
        done_batch = self.done[indices]
        truncated_batch = self.truncated[indices]

        return s1_batch, a_batch, r_batch, s2_batch, done_batch, truncated_batch

    def __len__(self):
        return self._len

    def save_to_file(self, filename, directory='./Experience/'):
        """
        Saves the memory buffer to a file.

        :param filename: Name of the file to save the buffer.
        :param directory: Directory to save the file. Defaults to './Experience/'.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, filename)

        with open(file_path, 'wb') as file:
            pickle.dump({
                's1': self.s1.cpu().numpy(),
                'a': self.a.cpu().numpy(),
                'r': self.r.cpu().numpy(),
                's2': self.s2.cpu().numpy(),
                'done': self.done.cpu().numpy(),
                'truncated': self.truncated.cpu().numpy(),
                '_len': self._len,
            }, file)

    def load_from_file(self, filename, directory='./Experience/'):
        """
        Loads a memory buffer from a file.

        :param filename: Name of the file to load the buffer from.
        :param directory: Directory to load the file from. Defaults to './Experience/'.
        """
        file_path = os.path.join(directory, filename)

        try:
            with open(file_path, 'rb') as file:
                checkpoint = pickle.load(file)

            self.s1[:checkpoint['_len']] = torch.tensor(checkpoint['s1'], dtype=torch.float32, device=self.device)
            self.a[:checkpoint['_len']] = torch.tensor(checkpoint['a'], dtype=torch.float32, device=self.device)
            self.r[:checkpoint['_len']] = torch.tensor(checkpoint['r'], dtype=torch.float32, device=self.device)
            self.s2[:checkpoint['_len']] = torch.tensor(checkpoint['s2'], dtype=torch.float32, device=self.device)
            self.done[:checkpoint['_len']] = torch.tensor(checkpoint['done'], dtype=torch.float32, device=self.device)
            self.truncated[:checkpoint['_len']] = torch.tensor(checkpoint['truncated'], dtype=torch.float32,
                                                               device=self.device)
            self._len = checkpoint['_len']
        except FileNotFoundError:
            print("File not found.")

class PrioritizedMemoryBuffer(MemoryBuffer):
    def __init__(self, size, state_shape, action_shape, alpha=0.6, beta=0.4, device="cpu"):
        """
        Initialize a prioritized memory buffer.

        :param size: Maximum size of the buffer.
        :param state_shape: Shape of the state tensor.
        :param action_shape: Shape of the action tensor.
        :param alpha: Prioritization exponent (0 <= alpha <= 1).
        :param beta: Importance-sampling exponent (0 <= beta <= 1).
        :param device: Device to store the tensors on ("cpu" or "cuda").
        """
        super().__init__(size, state_shape, action_shape, device=device)
        self.td_errors = None
        self.weights = None
        self.probabilities = None
        self.indicies = None
        self.alpha = alpha
        self.beta = beta
        self.priorities = torch.zeros(size, dtype=torch.float32, device=device)
        self.max_priority = 1.0

    def add(self, s1, a, r, s2, done=False, truncated=False):
        """
        Adds a particular transaction in the memory buffer with maximum priority.

        :param s1: Current state.
        :param a: Action taken.
        :param r: Reward received.
        :param s2: Next state.
        :param done: Whether the episode is done.
        :param truncated: Whether the episode is truncated.
        """
        idx = self._len % self.maxSize
        super().add(s1, a, r, s2, done, truncated)
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the memory buffer with prioritized sampling.

        :param batch_size: Size of the batch to sample.
        :return: A batch of transitions and importance sampling weights.
        """
        if self._len <= batch_size:
            # print("The replay memory size is less than the requested batch size. "
            #       "Returning all available samples.")
            batch_size = self._len

        priorities = self.priorities[:self._len] ** self.alpha
        probabilities = priorities / torch.sum(priorities)
        self.probabilities = probabilities
        self.indicies = torch.multinomial(probabilities, batch_size, replacement=True)

        weights = (self._len * probabilities[self.indicies]) ** (-self.beta)
        weights /= torch.max(weights)
        weights = weights.to(self.device)
        self.weights = weights
        
        s1_batch = self.s1[self.indicies]
        a_batch = self.a[self.indicies]
        r_batch = self.r[self.indicies]
        s2_batch = self.s2[self.indicies]
        done_batch = self.done[self.indicies]
        truncated_batch = self.truncated[self.indicies]

        batch = (s1_batch, a_batch, r_batch, s2_batch, done_batch, truncated_batch)

        return batch, weights, self.indicies


    def update_priorities(self, indices, td_errors):
        self.td_errors = td_errors.detach()
        priorities = torch.abs(td_errors) + 1e-5
        self.priorities[indices] = priorities.squeeze()
        self.max_priority = max(self.max_priority, torch.max(priorities).item())

    def set_beta(self, beta):
        """
        Set the importance-sampling exponent beta.

        :param beta: New value for beta (0 <= beta <= 1).
        """
        self.beta = beta
