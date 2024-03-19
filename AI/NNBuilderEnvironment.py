import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# My imports
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as Funct

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import Image
import enum
import json


class NNBuilderEnvironment():
    def __init__(self,dataset_train,dataset_val):
        self.dataset_val=dataset_val
        self.dataset_train=dataset_train
        self.set_dataloader(shuffle=True,batch_size=100)
        self.input_size=dataset_val[0][0].shape[0]
        self.output_size=dataset_val[:][1].unique().shape[0]
        self.current_nn=nn.Sequential()

    def add_layer(self, layer_type, **kwargs):
        if layer_type == 'linear':
            if 'in_features' not in kwargs:
                kwargs['in_features'] = self.current_nn[-1].out_features
            layer = nn.Linear(kwargs['in_features'], kwargs['out_features'])
        elif layer_type == 'relu':
            layer = nn.ReLU()
        elif layer_type == 'conv2d':
            if 'in_channels' not in kwargs:
                kwargs['in_channels'] = self.current_nn[-1].out_channels
            layer = nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], kwargs['kernel_size'],
                              stride=kwargs.get('stride', 1), padding=kwargs.get('padding', 0))
        else:
            raise ValueError("Invalid layer type. Choose from 'linear', 'relu', 'conv2d'")
        self.current_nn.add_module(layer_type, layer)

    def set_dataloader(self, shuffle, batch_size):
        self.train_loader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle)
        self.val_loader = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)
