import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, N_layers = 10, eta_i = 1, eta_f = 0.1, m2 = -1, generate = False):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(N):
            tempLayer = nn.Linear(1, 1, bias = False)
            eta = 1 - 0.1*k
            if generate:
                tempLayer.weight = nn.Parameter(torch.Tensor([1 + 0.1*3/np.tanh(3*eta)]))
            else:
                tempLayer.weight = nn.Parameter(torch.Tensor([1 + 0.1*np.rnadom.normal(loc = 1/eta, scale = 1)]))
            self.layers.append(tempLayer)
        if generate:
            self.out = genLayer()
        else:
            self.out = outLayer()

    def forward(self):
        phi = x[0, 0]
        pi = x[0, 1]
        for layer in self.layers:
