from __future__ import division

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

import numpy as np
import math

import buffer_fast as buffer
import utils as utils
import model as model

class Agent:

    def __init__(self, state_dim, action_dim, action_lim, ram,
                 batch_size=128,
                 batch_multi=16,
                 learning_rate=0.001,
                 gamma=0.99,
                 tau=0.001,
                 weight_decay=0,
                 theta=0.15,
                 sigma=0.3,
                 actor_reg_coeff=1,
                 critic_reg_coeff=0.0001):

        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :param batch_size: Batch size for training (int)
        :param batch_multi: How many batches to train on per iteration (int)
        :param learning_rate: Learning rate for the actor and critic models (float)
        :param gamma: Discount factor for calculating future rewards (float)
        :param tau: Parameter for soft update of target actor and critic models (float)
        :param weight_decay: Weight decay for the optimizer (float)
        :param theta: Ornstein-Uhlenbeck process parameter (float)
        :param sigma: Ornstein-Uhlenbeck process parameter (float)
        :param actor_reg_coeff: Regularization coefficient for actor (float)
        :param critic_reg_coeff: Regularization coefficient for critic (float)
        :return:
        """
        self.is_weights = None
        self.sampled_indices = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim, theta=theta, sigma=sigma)
        self.batch_size = batch_size
        self.batch_multi = batch_multi
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay

        self.actor_reg_coeff = actor_reg_coeff
        self.critic_reg_coeff = critic_reg_coeff

        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        self.critic = model.Critic(self.state_dim, self.action_dim)
        self.target_critic = model.Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('device: ', self.device)

        self.actor.to(self.device)
        self.target_actor.to(self.device)

        self.critic.to(self.device)
        self.target_critic.to(self.device)

        self.index = 0
        self.s1 = None
        self.a1 = None
        self.r1 = None
        self.s2 = None
        self.done = None
        self.truncated = None

    def step_done(self):
        self.optimize()

    def episode_done(self):
        self.noise.reset()

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        self.target_actor.eval()
        state = Variable(torch.from_numpy(state)).to(self.device)
        action = self.target_actor.forward(state).detach()
        self.target_actor.train()
        return action.data.cpu().numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state)).to(self.device)
        action = self.actor.forward(state).detach()
        #	print(action)
        new_action = action.cpu().data.numpy() + (self.noise.sample() * self.action_lim)
        return new_action

    def optimize(self):
        """
                Samples a random batch from replay memory and performs optimization
                :return:
                """
        if isinstance(self.ram, buffer.PrioritizedMemoryBuffer):
            batch, is_weights, sampled_indices = self.ram.sample(self.batch_size)
        else:
            batch = self.ram.sample(self.batch_size)
            is_weights = None
            sampled_indices = None

        s1, a1, r1, s2, done, truncated = batch

        s1 = s1.to(self.device)
        a1 = a1.to(self.device)
        r1 = r1.to(self.device)
        s2 = s2.to(self.device)
        done = done.to(self.device)
        truncated = truncated.to(self.device)
        if is_weights is not None:
            is_weights = is_weights.float().to(self.device)

        if sampled_indices is not None:
            sampled_indices = sampled_indices.to(self.device)
        else:
            sampled_indices = torch.tensor([]).to(self.device)  # Empty tensor for sampled_indices if None

        def safe_insert(A, B, x):
            bx = B.size()[0]
            if A.size()[0] < bx + x:
                x = 0
            A[x:(x + bx)] = B

        if self.s1 is None:
            self.s1 = s1
            self.a1 = a1
            self.r1 = r1
            self.s2 = s2
            self.done = done
            self.truncated = truncated
            self.sampled_indices = sampled_indices
            self.is_weights = is_weights
        elif (self.s1.size()[0] + s1.size()[0]) < self.batch_multi*self.batch_size:
            self.s1=torch.cat((self.s1, s1))
            self.a1=torch.cat((self.a1, a1))
            self.r1=torch.cat((self.r1, r1))
            self.s2=torch.cat((self.s2, s2))
            self.done=torch.cat((self.done, done))
            self.truncated=torch.cat((self.truncated, truncated))
            self.sampled_indices = torch.cat((self.sampled_indices, sampled_indices))
            self.is_weights = torch.cat((self.is_weights, is_weights))
        else:
            bis = s1.size()[0]
            self.index += bis
            self.index = self.index if self.index > self.batch_multi *self.batch_size else 0
            safe_insert(self.s1, s1, self.index)
            safe_insert(self.a1, a1, self.index)
            safe_insert(self.r1, r1, self.index)
            safe_insert(self.s2, s2, self.index)
            safe_insert(self.done, done, self.index)
            safe_insert(self.truncated, truncated, self.index)
            safe_insert(self.sampled_indices, sampled_indices, self.index)
            safe_insert(self.is_weights, is_weights, self.index)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        # self.target_critic.train()
        # self.target_actor.train()
        # self.critic.train()
        # self.actor.train()

        loss_critic = self.critic.loss(self.s1, self.a1, self.r1, self.s2,self.done,self.truncated,
                                       self.target_actor, self.target_critic,
                                       self.gamma,
                                       self.device,
                                       l2_reg_coeff=self.critic_reg_coeff,
                                       is_weights=self.is_weights)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        loss_actor= self.actor.loss(self.s1,self.done, self.critic, self.device, l2_reg_coeff=self.actor_reg_coeff)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        utils.soft_update(self.target_actor, self.actor, self.tau)
        utils.soft_update(self.target_critic, self.critic, self.tau)

        # Calculate TD errors and update priorities
        td_errors = self.critic.td_error + 1e-6
        # Update the priorities using the calculated TD errors
        if len(self.sampled_indices) > 1:
            self.ram.update_priorities(self.sampled_indices, td_errors)
        else:
            print("Not enough samples in the buffer to update priorities.")

    # if self.iter % 100 == 0:
    # 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
    # 		' Loss_critic :- ', loss_critic.data.numpy()
    # self.iter += 1


    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        models_dir = './Models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        actor_file = f'{models_dir}/{episode_count}_actor.pt'
        critic_file = f'{models_dir}/{episode_count}_critic.pt'
        torch.save(self.target_actor.state_dict(), actor_file)
        torch.save(self.target_critic.state_dict(), critic_file)
        print('Models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        actor_file = f'./Models/{episode}_actor.pt'
        critic_file = f'./Models/{episode}_critic.pt'

        if os.path.exists(actor_file) and os.path.exists(critic_file):
            self.actor.load_state_dict(torch.load(actor_file))
            self.critic.load_state_dict(torch.load(critic_file))
            utils.hard_update(self.target_actor, self.actor)
            utils.hard_update(self.target_critic, self.critic)
            print('Models loaded successfully')
        else:
            print(f"Error: One or both model files do not exist. Could not load models.")

    def set_noise(self, theta, sigma):
        """
        Sets the theta and sigma parameters of the Ornstein-Uhlenbeck noise generator used for exploration.

        :param theta: A float value specifying the value of the theta parameter. The theta parameter controls the rate at which the
        noise reverts to the mean value.
        :param sigma: A float value specifying the value of the sigma parameter. The sigma parameter controls the variance of the noise.
        :return: None
        """
        # Set the value of theta to the specified value
        self.noise.theta = theta

        # Set the value of sigma to the specified value
        self.noise.sigma = sigma

    def randomize_noise(self, theta_range=(0.05, 0.4), sigma_range=(0.01, 0.5)):
        """
        Randomizes the theta and sigma parameters of the Ornstein-Uhlenbeck noise generator used for exploration.

        :param theta_range: A tuple containing the minimum and maximum values for the theta parameter. The theta parameter controls
        the rate at which the noise reverts to the mean value.
        :param sigma_range: A tuple containing the minimum and maximum values for the sigma parameter. The sigma parameter controls
        the variance of the noise.
        :return: None
        """
        # Choose a random value for theta within the specified range
        self.noise.theta = random.uniform(*theta_range)

        # Choose a random value for sigma within the specified range
        self.noise.sigma = random.uniform(*sigma_range)


