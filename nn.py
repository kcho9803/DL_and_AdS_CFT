# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:24:27 2021

@author: Kyungchan Cho
"""

# Constructing Neural Network

import torch
import torch.nn as nn
import numpy as np

# Customized activation function
def activ(x):
    x = x.clone()
    x.data[:,1] -= 0.1 * torch.pow(x[:,0], 3)
    return x

# Customized output layer
class outLayer(nn.Module):
    
    def __init__(self):
        super(outLayer, self).__init__()
    
    def forward(self, x):
        F = 20*x[:,1] + x[:,0] - torch.pow(x[:,0], 3)
        return (torch.tanh(100*(F-0.1))-torch.tanh(100*(F+0.1))+2)/2
        #return torch.abs(F)

# Customized output layer for data generation    
class genLayer(nn.Module):
    
    def __init__(self):
        super(genLayer, self).__init__()
        
    def forward(self, x):
        F = 20*x[0,1] + x[0,0] - x[0,0]**3
        return torch.abs(F)

# Neural network, eta_ini = 1, eta_fin = 0.1, delta_eta = -0.1, N = 10
class Net(nn.Module):
  
    def __init__(self, N = 10, generate = False):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(N):
            tempLayer = nn.Linear(2, 2, bias = False)
            eta = 1 - 0.1*k
            #tempLayer.weight = nn.Parameter(torch.Tensor([[1, -0.1], [0.1, 1+0.1*3/np.tanh(3*eta)]]))
            if generate:
                tempLayer.weight = nn.Parameter(torch.Tensor([[1, -0.1], [0.1, 1+0.1*3/np.tanh(3*eta)]]))
            else:
                tempLayer.weight = nn.Parameter(torch.Tensor([[1, -0.1],[0.1, 1+0.1*np.random.normal(loc = 1/eta ,scale = 1)]]))
                #tempLayer.weight = nn.Parameter(torch.Tensor([[1, -0.1], [0.1, 1+0.1*3/np.tanh(3*eta)]]))
            self.layers.append(tempLayer)
        if generate:
            self.out = genLayer()
        else:
            self.out = outLayer()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = activ(x)
        x = self.out(x)
        return x

    def extractMetric(self):
        # Number of layers = 10
        metric = np.zeros((10,2))
        for k in range(10):
            metric[k,0] = 1 - 0.1*k
            metric[k,1] = (self.layers[k].weight[1,1].item() - 1) * 10
        return metric