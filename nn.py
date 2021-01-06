# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:24:27 2021

@author: Kyungchan Cho
"""

# Constructing Neural Network

import torch
import torch.nn as nn
import numpy as np

# Customized output layer
class outLayer(nn.Module):
    
    def __init__(self):
        super(outLayer, self).__init__()
    
    def forward(self, x):
        F = 2*np.pi/0.1 + x[0] - x[0]**3
        return torch.Tensor([[np.tanh(np.abs(F))]])

# Customized output layer for data generation    
class genLayer(nn.Module):
    
    def __init__(self):
        super(genLayer, self).__init__()
        
    def forward(self, x):
        F = 2*np.pi/0.1 + x[0] - x[0]**3
        return torch.Tensor([[np.abs(F)]])

# Neural network, eta_ini = 1, eta_fin = 0.1, stepsize = 0.1, N = 10
class Net(nn.Module):
  
    def __init__(self, N = 10, generate = False):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(N):
            tempLayer = nn.Linear(2, 2, bias = False)
            if generate:
                eta = 1 - 0.1*k
                tempLayer.weight.data = torch.Tensor([[1, 0.1], [-0.1, 1-0.1*3/np.tanh(3*eta)]])
            self.layers.append(tempLayer)
        if generate:
            self.out = genLayer()
        else:
            self.out = outLayer()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x