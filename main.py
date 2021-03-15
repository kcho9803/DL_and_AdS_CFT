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
import numpy as np

# Generate dataset
dataset = data_generator.GeneratedDataset()
dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)

# Set neural network
model = nn.Net(generate = False)

# Extract initialized metric
initMetric = model.extractMetric()

# Set loss function & optimizer
criterion = NN.L1Loss(reduction = 'sum')
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 100
losses = []
c_reg = 0.001

# Training
for curEpoch in range(epochs):
    batch_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        tempMetric = model.extractMetric()
        loss = criterion(y_pred.view_as(y), y) + c_reg * np.sum(np.multiply(np.power(tempMetric[:9][0],4), np.power(tempMetric[1:][1]-tempMetric[:9][1], 2)))
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        # Apply constraint
        for layer in model.layers:
            layer.weight = NN.Parameter(torch.Tensor([[1, -0.1],[0.1, layer.weight[1][1].item()]]))
    losses.append(batch_loss)
    print("Epoch {0}: Loss = {1}".format(curEpoch+1, batch_loss))
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
ax1.plot(np.arange(100)+1, losses, label = 'Loss')
ax1.legend(loc = 'upper right')
        
plt.show()
fig1.savefig('TrainingLoss.pdf')

# Plot trained metric
trainedMetric = model.extractMetric()
fig2, ax2 = plt.subplots(figsize = (width, height))
ax2.plot(trainedMetric[:][0], trainedMetric[:][1], label = 'Emergent Metric')
ax2.plot(trainedMetric[:][0], 3*np.ones_like(trainedMetric[:][0])/np.tanh(3*trainedMetric[:][0]), label = 'True Metric')
ax2.legend(loc = 'upper right')

plt.show()
fig2.savefig('TrainedMetric.pdf')

# Plot initial metric
fig3, ax3 = plt.subplots(figsize = (width, height))
ax3.plot(initialMetric[:][0], initialMetric[:][1], label = 'Emergent Metric')
ax3.plot(initialMetric[:][0], 3*np.ones_like(initialMetric[:][0])/np.tanh(3*initialMetric[:][0]), label = 'True Metric')
ax3.legend(loc = 'upper right')

plt.show()
fig3.savefig('InitialMetric.pdf')