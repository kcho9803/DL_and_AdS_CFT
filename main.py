# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:32:57 2021

@author: Kyungchan Cho
"""

import data_generator
import nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as NN

# Generate dataset
dataset = data_generator.GeneratedDataset()
dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)

# Set neural network
model = nn.Net(generate = False)

# Set optimizer
criterion = NN.L1Loss()
optimizer = optim.Adam()