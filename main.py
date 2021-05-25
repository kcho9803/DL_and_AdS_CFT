# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:32:57 2021

@author: Kyungchan Cho
"""

import data_generator
import nn2 as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as NN
import numpy as np

def L1(y_bar, y):
    return torch.sum(torch.abs(y_bar - y))

path = 'E:\\Github\\DL_and_AdS_CFT\\'

load = True
if load:
    dataset = data_generator.LoadedDataset()
else:
    # Generate dataset
    dataset = data_generator.GeneratedDataset()
dataloader = DataLoader(dataset, batch_size = 500, shuffle = True)

# Set neural network
model = nn.Net(generate = False)
print('Neural network set')

# Extract initialized metric
initMetric = model.extractMetric()
print('Initial metric extracted')

# Set loss function & optimizer
criterion = NN.L1Loss(reduction = 'sum')
optimizer = optim.Adam(model.parameters(), lr = 0.00005)

epochs = 150000
losses = []
norms = []
regs = []
c_reg = 0.15

test = False

# Training
print('Beginning training sequence')
for curEpoch in range(epochs):
    batch_loss = 0.0
    batch_norm = 0.0
    batch_reg = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        if test:
            print(y_pred)
            print(y)
            test = False
        regularizer = torch.zeros(1)
        for i in range(8):
            # regularizer += ((1-0.1*i)**4)*((model.layers[i+1].weight[1,1]-model.layers[i].weight[1,1])**2)
            # regularizer += ((1-0.1*i)**4)*((model.layers[i+1].weight[0]-model.layers[i].weight[0])**2)
            # regularizer += ((model.layers[i+2].weight[1,1]-2*model.layers[i+1].weight[1,1]+model.layers[i].weight[1,1]))**2
            regularizer += ((model.layers[i+2].weight[0]-2*model.layers[i+1].weight[0]+model.layers[i].weight[0]))**2
        loss = criterion(y_pred.view_as(y), y)**2 + c_reg * regularizer
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        batch_norm += criterion(y_pred.view_as(y), y).item()
        batch_reg += regularizer.item() * c_reg
        # Apply constraint
        # for layer in model.layers:
        #    layer.weight = NN.Parameter(torch.Tensor([[1, -0.1],[0.1, layer.weight[1,1].item()]]))
    losses.append(batch_loss)
    norms.append(batch_norm)
    regs.append(batch_reg)
    print("Epoch {0}: Loss = {1}, L1 = {2}, reg = {3}".format(curEpoch+1, batch_loss, batch_norm, batch_reg))
print("Training complete")

# Plot loss
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set variables for plotting
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8.0
mpl.rcParams.update({
    'xtick.major.size': 2,
    'xtick.minor.size': 1.5,
    'xtick.major.width': 0.75,
    'xtick.minor.width': 0.75,
    'xtick.labelsize': 8.0,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.major.size': 2,
    'ytick.minor.size': 1.5,
    'ytick.major.width': 0.75,
    'ytick.minor.width': 0.75,
    'ytick.labelsize': 8.0,
    'ytick.direction': 'in',
    'xtick.major.pad': 2,
    'xtick.minor.pad': 2,
    'ytick.major.pad': 2,
    'ytick.minor.pad': 2,
    'ytick.right': True,
    'savefig.dpi': 600,
    'savefig.transparent': True,
    'axes.linewidth': 0.75,
    'lines.linewidth': 1.0
})
width = 3.4
height = width * 0.9

# Plot loss
fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.set_yscale('log')
ax1.plot(np.arange(epochs)+1, losses, label = 'Loss')
ax1.legend(loc = 'upper right')
fig1.savefig(path+'TrainingLoss.png')

# Plot trained metric
trainedMetric = model.extractMetric()
fig2, ax2 = plt.subplots(figsize = (width, height))
ax2.plot(trainedMetric[:,0], trainedMetric[:,1], label = 'Emergent Metric')
ax2.plot(trainedMetric[:,0], 3*np.ones_like(trainedMetric[:,0])/np.tanh(3*trainedMetric[:,0]), label = 'True Metric')
ax2.legend(loc = 'upper right')
fig2.savefig(path+'TrainedMetric.png')

# Plot initial metric
fig3, ax3 = plt.subplots(figsize = (width, height))
ax3.plot(initMetric[:,0], initMetric[:,1], label = 'Emergent Metric')
ax3.plot(initMetric[:,0], 3*np.ones_like(initMetric[:,0])/np.tanh(3*initMetric[:,0]), label = 'True Metric')
ax3.legend(loc = 'upper right')
fig3.savefig(path+'InitialMetric.png')

# Plot L1 norm
fig4, ax4 = plt.subplots(figsize = (width, height))
ax4.set_yscale('log')
ax4.plot(np.arange(epochs)+1, norms, label = 'L1')
ax4.legend(loc = 'upper right')
fig4.savefig(path+'TrainingLoss_L1.png')

# Plot regularizer
fig5, ax5 = plt.subplots(figsize = (width, height))
ax5.plot(np.arange(epochs)+1, regs, label = 'Reg')
ax5.legend(loc = 'upper right')
fig5.savefig(path+'TrainingLoss_reg.png')