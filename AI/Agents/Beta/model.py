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

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs1.activation = nn.ReLU()
        self.fcs2 = nn.Linear(256,128)
        self.fcs2.activation = nn.ReLU()

        self.fca1 = nn.Linear(action_dim,128)
        self.fca1.activation = nn.ReLU()

        self.fc2 = nn.Linear(256,128)
        self.fc2.activation = nn.ReLU()

        self.fc3 = nn.Linear(128,1)
        self.fc3.weight.data.uniform_(-EPS,EPS)

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
        do_bn = s1.size(0) > 1 and True
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

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.activation = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc2.activation = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.fc3.activation = nn.ReLU()
        self.fc4 = nn.Linear(64, action_dim)
        self.fc4.activation = nn.Tanh()
        self.fc4.weight.data.uniform_(-EPS,EPS)

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




