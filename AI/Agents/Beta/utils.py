import time

import numpy as np
import torch
import shutil
import itertools
import torch.autograd as Variable


def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)
def generate_combinations(M_CONFIG,do_all=False):
	combinations = []
	if do_all:

		config_values = [v if isinstance(v, list) else [v] for v in M_CONFIG.values()]
		config_combinations = list(itertools.product(*config_values))
		count = 0
		for combination in config_combinations:
			config = dict(zip(M_CONFIG.keys(), combination))
			combinations.append(config)
	else:
		# Get the default values
		default_values = [v[0] if isinstance(v, list) else v for v in M_CONFIG.values()]
		combinations.append(dict(zip(M_CONFIG.keys(), default_values)))
		# Iterate over each option and generate combinations where it doesn't use the default value
		for i, key in enumerate(M_CONFIG.keys()):
			if isinstance(M_CONFIG[key], list):
				for val in M_CONFIG[key][1:]:
					combo = list(default_values)
					combo[i] = val
					combinations.append(dict(zip(M_CONFIG.keys(), combo)))

		# Add the default combination


	return combinations



def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')

def tic():
	global tic_start_time
	tic_start_time = time.time()

def toc():
	global tic_start_time
	return time.time() - tic_start_time

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
	"""
    A class for generating Ornstein-Uhlenbeck noise to add exploration
    in the action space of a Reinforcement Learning agent.
    """

	def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
		"""
        Constructor for the Ornstein-Uhlenbeck noise class.

        Args:
            action_dim (int): Dimensionality of the action space.
            mu (float): Mean of the noise process. Default is 0.
            theta (float): A parameter that governs the rate of mean reversion.
                Default is 0.15.
            sigma (float): The standard deviation of the noise process.
                Default is 0.2.
        """
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		"""
        Reset the Ornstein-Uhlenbeck noise process.
        """
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		"""
        Sample from the Ornstein-Uhlenbeck noise process.

        Returns:
            ndarray: An array of noise values for the action space.
        """
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
	ou = OrnsteinUhlenbeckActionNoise(10)
	states = []
	for i in range(1000):
		states.append(ou.sample())
	import matplotlib
	import matplotlib.pyplot as plt

	matplotlib.use('TkAgg')

	plt.plot(states)
	plt.show()
	plt.pause(10)
