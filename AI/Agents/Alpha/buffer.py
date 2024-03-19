import numpy as np
import random
from collections import deque
import pickle
import os

class MemoryBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0


	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr

	def sample_recent_bias(self, count, recent_proportion=0.05):
		"""
        samples a random batch from the replay memory buffer, with a bias towards the recent samples
        :param count: batch size
        :param recent_proportion: proportion of recent samples vs random samples in the final batch
        :return: batch (numpy array)
        """
		batch = []
		count = min(count, self.len)
		recent_count = int(count * recent_proportion)
		recent_count = min(recent_count, self.len)
		random_count = count - recent_count
		if recent_count > 0:
			batch = [self.buffer[-i] for i in range(1, recent_count + 1)]
		if random_count > 0:
			random_batch = random.sample(self.buffer, random_count)
			batch += random_batch

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr

	def len(self):
		return self.len

	def add(self, s, a, r, s1):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""
		transition = (s,a,r,s1)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)

	def save_to_file(self, filename):
		if not os.path.exists('./Experience/'):
			os.makedirs('./Experience/')
		file = open('./Experience/' + filename, 'wb')
		pickle.dump(self.buffer, file)
		file.close()

	def load_from_file(self, filename):
		try:
			file = open('./Experience/' + filename, 'rb')
			buffer = pickle.load(file)
			self.buffer.extend(buffer)
			self.len = min(self.len + len(buffer), self.maxSize)
			file.close()
		except FileNotFoundError:
			print("File not found.")
