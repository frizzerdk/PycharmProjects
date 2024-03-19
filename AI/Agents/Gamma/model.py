import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
    # The "fanin" parameter is either provided as an argument or set to the number of rows in "size" if not provided
    fanin = fanin or size[0]
    # Calculates the scaling factor "v" by taking the reciprocal square root of "fanin"
    v = 1. / np.sqrt(fanin)
    # Prints the calculated value of "v"
    print(v)
    # Returns a tensor of specified "size" with randomly initialized values uniformly distributed between -v and v
    return torch.Tensor(size).uniform_(-v, v)

class Reward(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Reward, self).__init__()

        # Set the dimensions of the state and action
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Layer 1
        # Layer s1 state
        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs1.activation = nn.LeakyReLU()
        # Layer s2 state
        self.fcs2 = nn.Linear(256,128)
        self.fcs2.activation = nn.LeakyReLU()
        # Layer a1 action
        self.fca1 = nn.Linear(action_dim,128)
        self.fca1.activation = nn.LeakyReLU()

        # Layer 2 from state and action
        self.fc2 = nn.Linear(256,128)
        self.fc2.activation = nn.LeakyReLU()

        # Output layer
        self.fc3 = nn.Linear(128,1)
        self.fc3.weight.data.uniform_(-EPS,EPS)

        # Other layers
        self.dropout = nn.Dropout(0.05)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """

        s1 = self.fcs1(state)
        s1 = self.fcs1.activation(s1)
        do_bn = s1.size(0) > 1 and False
        if do_bn:
            s1 = self.bn1(s1)

        s2 = self.fcs2(s1)
        s2 = self.fcs2.activation(s2)
        if do_bn:
            s2 = self.bn2(s2)

        a1 = self.fca1(action)
        a1 = self.fca1.activation(a1)
        if do_bn:
            a1 = self.bn2(a1)

        x = torch.cat((s2, a1), dim=1)

        x = self.fc2(x)
        x = self.fc2.activation(x)
        x = self.fc3(x)
        reward = x
        return reward

    def loss(self, state, action, reward):
        """
        returns loss of critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :param reward: Target reward (Torch Variable : [n,1] )
        :return: loss (Torch Variable : [1] )
        """
        # Calculate the loss
        r_predicted = torch.squeeze(self.forward(state, action))
        # print max absoulute predicted and target reward
        #print("max absoulute predicted reward: ", torch.max(torch.abs(r_predicted)), "max absoulute target reward: ", torch.max(torch.abs(reward)))
        loss = F.mse_loss(r_predicted, reward)
        return loss

    def validate(self, state, action, reward):
        """
        returns average loss of critic network over validation set
        :param state: Input state (Torch Variable : [n,batch_size,state_dim])
        :param action: Input Action (Torch Variable : [n,batch_size,action_dim])
        :param reward: Target reward (Torch Variable : [n,batch_size,1])
        :return: average loss (float)
        """
        # Calculate the average loss over the validation set
        average_loss = 0
        samples = 0
        val_batches = zip(state, action, reward)
        with torch.no_grad():
            for s1_batch, a1_batch, r1_batch in val_batches:
                loss_critic = self.loss(s1_batch, a1_batch, r1_batch)
                average_loss += loss_critic.item()
                samples = s1_batch.size(0)
        average_loss = average_loss / samples
        return average_loss



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim,pred_steps=10):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc2.activation = nn.LeakyReLU()
        self.fc3 = nn.Linear(128, 64)
        self.fc3.activation = nn.LeakyReLU()
        self.fc4 = nn.Linear(64, action_dim)
        self.fc4.activation = nn.Tanh()
        self.fc4.weight.data.uniform_(-EPS,EPS)
        self.l2_reg_coeff = 0.1
        self.pred_steps = pred_steps

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.activations = []



    def forward(self, state,save_activations=False):
        do_bn = state.dim() > 1 and False

        # layer 1
        x = self.fc1(state)
        x = self.fc1.activation(x)
        if do_bn:
            x = self.bn1(x)
        if save_activations:
            self.activations.append(x.clone())

        # layer 2
        x = self.fc2(x)
        x = self.fc2.activation(x)
        if do_bn:
            x = self.bn2(x)
        if save_activations:
            self.activations.append(x.clone())

        # layer 3
        x = self.fc3(x)
        x = self.fc3.activation(x)
        if do_bn:
            x = self.bn3(x)
        if save_activations:
            self.activations.append(x.clone())

        # layer 4
        x = self.fc4(x)
        action = self.fc4.activation(x)

        action = action * self.action_lim

        return action

    def get_activations(self, state):
        self.activations = []
        action = self.forward(state, save_activations=True)
        return action, self.activations

    def loss(self, state, critic,predictor):
        """
        returns loss of actor network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :param critic: Critic network
        :param predictor: Predictor network
        """
        # Calculate the loss
        action, layer_activations= self.get_activations(state)
        total_reward = critic.forward(state, action).sum()
        gamma = 0.95
        for i in range(self.pred_steps):
            state = predictor.forward(state, action)
            action = self.forward(state)
            total_reward += critic.forward(state, action).sum()*gamma**i

        loss = -total_reward
        actor_reg = 0
        for activation in layer_activations:
            F.relu(activation.abs()-3, inplace=True)
            actor_reg += torch.norm(activation*10, 2)
        actor_reg = self.l2_reg_coeff * actor_reg
        return loss+actor_reg

    def validate(self, state, critic,predictor):
        """
        returns average loss of actor network over validation set
        :param state: Input state (Torch Variable : [n,batch_size,state_dim])
        :param action: Input Action (Torch Variable : [n,batch_size,action_dim])
        :param critic: Critic network
        :param predictor: Predictor network
        :return: average loss (float)
        """
        # Calculate the average loss over the validation set
        average_loss = 0
        samples = 0
        val_batches = zip(state)
        with torch.no_grad():
            for s1_batch, in val_batches:
                loss_actor = self.loss(s1_batch, critic,predictor)
                average_loss += loss_actor.sum().item()
                samples = s1_batch.size(0)
        average_loss = average_loss / samples
        return average_loss

class Predictor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
         :param state_dim: Dimension of input state (int)
         :param action_dim: Dimension of input action (int)
         :return:
         """
        super(Predictor, self).__init__()

        # Set the dimensions of the state and action
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Layer 1
        # Layer s1 state
        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs1.activation = nn.LeakyReLU()
        # Layer s2 state
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.activation = nn.LeakyReLU()
        # Layer a1 action
        self.fca1 = nn.Linear(action_dim, 128)
        self.fca1.activation = nn.LeakyReLU()

        # Layer 2 from state and action
        self.fc2 = nn.Linear(256, 128)
        self.fc2.activation = nn.LeakyReLU()

        # Output layer
        self.fc3 = nn.Linear(128, state_dim)

        # Other layers
        #self.dropout = nn.Dropout(0.05)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)


    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: state: predicted state (Torch Variable : [n,state_dim] )
        """

        s1 = self.fcs1(state)
        s1 = self.fcs1.activation(s1)
        do_bn = s1.size(0) > 1 and False
        if do_bn:
            s1 = self.bn1(s1)

        s2 = self.fcs2(s1)
        s2 = self.fcs2.activation(s2)
        if do_bn:
            s2 = self.bn2(s2)

        a1 = self.fca1(action)
        a1 = self.fca1.activation(a1)
        if do_bn:
            a1 = self.bn2(a1)

        x = torch.cat((s2, a1), dim=1)

        x = self.fc2(x)
        x = self.fc2.activation(x)
        x = self.fc3(x)
        state = x
        return state

    def loss(self, state, action, next_state):
        """
        returns loss of predictor network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :param next_state: Input next state (Torch Variable : [n,state_dim] )
        """
        # Calculate the loss
        predicted_state = self.forward(state, action)
        loss = F.smooth_l1_loss(predicted_state, next_state)

        return loss

    def validate(self, state, action, next_state):
        """
        returns average loss of predictor network over validation set
        :param state: Input state (Torch Variable : [n,batch_size,state_dim])
        :param action: Input Action (Torch Variable : [n,batch_size,action_dim])
        :param next_state: Input next state (Torch Variable : [n,batch_size,state_dim])
        :return: average loss (float)
        """
        # Calculate the average loss over the validation set
        average_loss = 0
        samples = 0
        val_batches = zip(state, action, next_state)
        with torch.no_grad():
            for s1_batch, a_batch, s2_batch in val_batches:
                loss_predictor = self.loss(s1_batch, a_batch, s2_batch)
                average_loss += loss_predictor.item()
                samples = s1_batch.size(0)
        average_loss = average_loss / samples
        return average_loss




