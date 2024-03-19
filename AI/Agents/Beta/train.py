from __future__ import division

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

import numpy as np
import math

import utils as utils
import model as model

class Trainer:

    def __init__(self, state_dim, action_dim, action_lim, ram, ram_val,
                 batch_size=128,
                 batch_multi=16,
                 learning_rate=0.001,
                 gamma=0.99,
                 tau=0.001,
                 weight_decay=0,
                 theta=0.15,
                 sigma=0.3,
                 val_split=0.2):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :param batch_size: Batch size for training (int)
        :param batches_max: Maximum size of replay memory buffer (int)
        :param learning_rate: Learning rate for the actor and critic models (float)
        :param gamma: Discount factor for calculating future rewards (float)
        :param tau: Parameter for soft update of target actor and critic models (float)
        :param weight_decay: Weight decay for the optimizer (float)
        :param theta: Ornstein-Uhlenbeck process parameter (float)
        :param sigma: Ornstein-Uhlenbeck process parameter (float)
        :return:
        """
        self.l2_reg_coeff = 0.01
        self.val_split = val_split
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.ram_val = ram_val
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim, theta=theta, sigma=sigma)
        self.batch_size = batch_size
        self.batch_multi = batch_multi
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay
        self.best_critic_val_loss=float('inf')
        self.best_actor_val_loss=float('inf')

        self.actor_int = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate*1, weight_decay=self.weight_decay)

        self.critic_init = model.Critic(self.state_dim, self.action_dim)
        self.critic = model.Critic(self.state_dim, self.action_dim)
        self.target_critic = model.Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        self.reset_models()

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

    def reset_models(self):
        utils.hard_update(self.target_actor, self.actor_int)
        utils.hard_update(self.target_critic, self.critic_init)
        utils.hard_update(self.actor, self.actor_int)
        utils.hard_update(self.critic, self.critic_init)

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
        Optimizes the critic and actor using samples from the memory buffer.
        :return:
        """
        epoch_num = 10
        sub_epoch_num = 4
        patience = 1
        critic_val_loss = []
        actor_val_loss = []
        best_crit = self.critic
        best_actor = self.actor
        patience_count = 0
        final_epoch = 0
        # get entire memory buffer and convert to torch tensors then move to GPU
        val_size=self.batch_size * self.batch_multi * self.val_split
        val_size=val_size - val_size % self.batch_size
        s1, a1, r1, s2 = self.ram.sample_recent_bias(self.batch_size * self.batch_multi,recent_proportion=0.7)
        s1_val, a1_val, r1_val, s2_val = self.ram_val.sample_recent_bias(val_size,recent_proportion=0.7)
        if len(s1) < self.batch_size:
            return self.best_critic_val_loss, 0, critic_val_loss, actor_val_loss
        print('buffer size: ', len(s1),' val buffer size: ', len(s1_val))
        # Move to GPU
        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        r1 = torch.from_numpy(r1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)
        s1_val = torch.from_numpy(s1_val).to(self.device)
        a1_val = torch.from_numpy(a1_val).to(self.device)
        r1_val = torch.from_numpy(r1_val).to(self.device)
        s2_val = torch.from_numpy(s2_val).to(self.device)
        self.s1 = s1_val
        self.a1 = a1_val
        self.r1 = r1_val
        self.s2 = s2_val

        # make itterable for batches
        s1_test = torch.split(s1, self.batch_size)
        a1_test = torch.split(a1, self.batch_size)
        r1_test = torch.split(r1, self.batch_size)
        s2_test = torch.split(s2, self.batch_size)

        s1_val = torch.split(s1_val, self.batch_size)
        a1_val = torch.split(a1_val, self.batch_size)
        r1_val = torch.split(r1_val, self.batch_size)
        s2_val = torch.split(s2_val, self.batch_size)

        #training loop
        for epoch in range(epoch_num):
            improved = False
            sub_patience = 1
            sub_patience_count = 0
            self.best_critic_val_loss = self.validate_critic(a1_val, r1_val, s1_val, s2_val)
            for sub_epoch in range(sub_epoch_num):
                # ---------------------- optimize critic ----------------------
                test_batches = zip(s1_test, a1_test, r1_test, s2_test)
                for s1_batch, a1_batch, r1_batch, s2_batch in test_batches:
                  # Use target actor exploitation policy here for loss evaluation
                    a2 = self.target_actor.forward(s2_batch).detach()
                    next_val = torch.squeeze(self.target_critic.forward(s2_batch, a2).detach())
                    # y_exp = r + gamma*Q'( s2, pi'(s2))
                    y_expected = r1_batch + self.gamma * next_val
                    # y_pred = Q( s1, a1)
                    y_predicted = torch.squeeze(self.critic.forward(s1_batch, a1_batch))

                    # Add L2 regularization term to the loss function
                    critic_reg = 0
                    for param in self.critic.parameters():
                        critic_reg += torch.norm(param, 2)
                    critic_reg = self.l2_reg_coeff*0.01 * critic_reg

                    # compute critic loss, and update the critic
                    loss_critic = F.smooth_l1_loss(y_predicted, y_expected)+ critic_reg
                    self.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    self.critic_optimizer.step()

                average_critic_loss = self.validate_critic(a1_val, r1_val, s1_val, s2_val)
                critic_val_loss.append(average_critic_loss)
                if average_critic_loss < self.best_critic_val_loss:
                    self.best_critic_val_loss = average_critic_loss
                    best_crit = self.critic
                    print("New best critic validation loss: ", self.best_critic_val_loss,
                          " at epoch ",epoch,
                          " sub epoch ", sub_epoch,
                          " patience count: ", sub_patience_count)
                    sub_patience_count = 0
                    improved = True
                else:
                    sub_patience_count += 1

                if sub_patience_count > sub_patience:
                    self.critic = best_crit
                    self.actor = best_actor
                    break
            sub_patience_count=0
            self.best_actor_val_loss = self.validate_actor(a1_val, r1_val, s1_val, s2_val)
            for sub_epoch in range(sub_epoch_num):
                # ---------------------- optimize actor ----------------------
                test_batches = zip(s1_test, a1_test, r1_test, s2_test)
                for s1_batch, a1_batch, r1_batch, s2_batch in test_batches:
                    pred_a1, layer_activations= self.actor.get_activations(s1_batch)
                    pred_a1.to(self.device)
                    actor_reg = 0
                    for activation in layer_activations:
                        actor_reg += torch.norm(activation, 2)
                    actor_reg = self.l2_reg_coeff * actor_reg

                    loss_actor = -1 * torch.sum(best_crit.forward(s1_batch, pred_a1).to(self.device)) + actor_reg
                    self.actor_optimizer.zero_grad()
                    loss_actor.backward()
                    self.actor_optimizer.step()

                average_actor_loss = self.validate_actor(a1_val, r1_val, s1_val, s2_val)
                actor_val_loss.append(average_actor_loss)
                if average_actor_loss < self.best_actor_val_loss:
                    self.best_actor_val_loss = average_actor_loss
                    best_actor = self.actor
                    print("New best actor validation loss: ", self.best_actor_val_loss,
                          " at epoch ",epoch,
                          " sub epoch ", sub_epoch,
                          " patience count: ", sub_patience_count)
                    sub_patience_count = 0
                    improved = True
                else:
                    sub_patience_count += 1

                if sub_patience_count > sub_patience:
                    self.critic = best_crit
                    self.actor = best_actor
                    break


            if improved:
                print("New best validation loss: ", self.best_critic_val_loss,
                      " and actor loss: ", self.best_actor_val_loss,
                      " at epoch ", epoch,
                      " patience count: ", patience_count)
                patience_count = 0
            else:
                patience_count += 1

            if patience_count > patience:
                self.critic = best_crit
                self.actor = best_actor
                break



        utils.hard_update(self.target_actor, self.actor)#, self.tau)
        utils.hard_update(self.target_critic, self.critic)#, self.tau)
        return self.best_critic_val_loss, final_epoch, critic_val_loss,actor_val_loss



        # if self.iter % 100 == 0:

    def validate_critic(self, a1_val, r1_val, s1_val, s2_val):
        # ---------------------- check validation ----------------------
        average_loss = 0
        batch_num = 0
        val_batches = zip(s1_val, a1_val, r1_val, s2_val)
        with torch.no_grad():
            for s1_batch, a1_batch, r1_batch, s2_batch in val_batches:
                a2 = self.target_actor.forward(s2_batch).detach()
                next_val = torch.squeeze(self.target_critic.forward(s2_batch, a2).detach())
                # y_exp = r + gamma*Q'( s2, pi'(s2))
                y_expected = r1_batch + self.gamma * next_val
                # y_pred = Q( s1, a1)
                y_predicted = torch.squeeze(self.critic.forward(s1_batch, a1_batch))

                # compute critic loss, and update the critic
                loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
                average_loss += loss_critic.item()
                batch_num += 1
        average_loss = average_loss / (batch_num*self.batch_size)
        return average_loss
    def validate_actor(self, a1_val, r1_val, s1_val, s2_val):
        # ---------------------- check validation ----------------------
        average_loss = 0
        batch_num = 0
        val_batches = zip(s1_val, a1_val, r1_val, s2_val)
        with torch.no_grad():
            for s1_batch, a1_batch, r1_batch, s2_batch in val_batches:
                pred_a1 = self.actor.forward(s1_batch).to(self.device)
                loss_actor = -1 * torch.sum(self.critic.forward(s1_batch, pred_a1).to(self.device))
                average_loss += loss_actor.item()
                batch_num += 1
        average_loss = average_loss / (batch_num*self.batch_size)
        return average_loss

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


