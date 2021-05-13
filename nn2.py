import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, N_layers = 10, eta_i = 1, eta_f = 0.1, m2 = -1, generate = False):
        super(Net, self).__init__()

        self.N_layers = N_layers
        self.m2 = m2
        self.layers = nn.ModuleList()
        self.etaArray = torch.linspace(eta_i, eta_f, N_layers)
        self.stepSize = self.etaArray[1] - self.etaArray[0]
        for k in range(N_layers):
            tempLayer = nn.Linear(1, 1, bias = False)
            # eta = 1 - 0.1*k
            eta = self.etaArray[k]
            if generate:
                tempLayer.weight = nn.Parameter(torch.Tensor([1 + 0.1*3/np.tanh(3*eta)]))
            else:
                tempLayer.weight = nn.Parameter(torch.Tensor([1 + 0.1*np.random.normal(loc = 1/eta, scale = 1)]))
            self.layers.append(tempLayer)
        if generate:
            self.out = genLayer()
        else:
            self.out = outLayer()

    def forward(self, x):
        phi_cur = x[:, 0].reshape(-1, 1)
        pi_cur = x[:, 1].reshape(-1, 1)
        for layer in self.layers:
            phi_nxt = phi_cur + self.stepSize * pi_cur
            pi_nxt = self.stepSize * (self.m2 * phi_cur) + layer(pi_cur).reshape(-1, 1)
            pi_nxt = pi_nxt + self.stepSize * (phi_cur ** 3)
            phi_cur = phi_nxt
            pi_cur = pi_nxt
        pred = self.out(phi_nxt, pi_nxt)
        return pred

    def extractMetric(self):
        metric = np.zeros((self.N_layers, 2))
        for k in range(self.N_layers):
            metric[k,0] = self.etaArray[k]
            metric[k,1] = (self.layers[k].weight[0].item() - 1) / (-self.stepSize)
        return metric

class genLayer(nn.Module):

    pi_only = True
    stepSize = -0.1
    m2 = -1

    def __init__(self):
        super(genLayer, self).__init__()
    
    def forward(self, phi, pi):
        if self.pi_only:
            F = pi
        else:
            F = 20 * pi - self.m2 * phi - torch.pow(phi, 3)
        return torch.abs(F)

class outLayer(nn.Module):

    pi_only = True
    stepSize = -0.1
    m2 = -1

    def __init__(self):
        super(outLayer, self).__init__()

    def forward(self, phi, pi):
        if self.pi_only:
            F = pi
        else:
            F = 20 * pi - self.m2 * phi - torch.pow(phi, 3)
        return (torch.tanh(100*(F-0.1))-torch.tanh(100*(F+0.1))+2)/2