# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:19:06 2021

@author: Kyungchan Cho
"""

import numpy as np
import torch
import nn
from torch.utils.data import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

class GeneratedDataset(Dataset):
    # Data generation. d = 3, L = 1. metric: h(eta) = 3coth(3eta)
    # Generate positive and negative data, 1000 for each class.
    def __init__(self):
        print('Generating dataset...')
        generator = nn.Net(generate = True)
        eps = 0.1
        n_pos = 0
        n_neg = 0
        pos = torch.zeros(1000,2)
        neg = torch.zeros(1000,2)
        itr = 0
        while (n_pos < 1000)or(n_neg < 1000):
            phi_ini = np.random.uniform(low = 0.0, high = 1.5)
            pi_ini = np.random.uniform(low = -0.2, high = 0.2)
            rand_input = torch.Tensor([[phi_ini, pi_ini]])
            out = generator.forward(rand_input)
            if out < eps:
                if n_pos < 1000:
                    if (n_pos+1)%50 == 0:
                        print('Data Generation: {}%'.format((n_pos+1)/10))
                    pos[n_pos] = rand_input
                    n_pos += 1
            if out > eps:
                if n_neg < 1000:
                    neg[n_neg] = rand_input
                    n_neg += 1
            itr += 1
        self.y_data = torch.cat((torch.zeros(1000), torch.ones(1000)), dim = 0)
        self.x_data = torch.cat((pos, neg), dim = 0)
        print('Dataset generated')
        # Save generated dataset
        df = pd.DataFrame(torch.cat((self.x_data, torch.transpose(self.y_data.unsqueeze(0),0,1)), dim = 1).numpy(), columns = ['phi','pi','label'])
        # Specify path
        path = 'D:\\Github\\DL_and_AdS_CFT\\'
        df.to_csv(path+'generatedDataset.csv', sep = ',')
        print('Dataset saved')
        
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
        
        fig, ax = plt.subplots(figsize = (width, height))
        temp1 = pos.numpy()
        temp2 = neg.numpy()
        ax.scatter(temp1[:, 0], temp1[:, 1], label = 'Positive', marker = '.')
        ax.scatter(temp2[:, 0], temp2[:, 1], label = 'Negative', marker = '.')
        ax.legend(loc = 'upper right')
        fig.savefig(path+'Dataset.pdf')
        
    def __len__(self):
        return list(self.x_data.size())[0]
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

class LoadedDataset(Dataset):
    def __init__(self):
        # Specify path
        path = 'D:\\Github\\DL_and_AdS_CFT\\'
        savedDataset = pd.read_csv(path+'generatedDataset.csv', header = 0).to_numpy()
        print(np.shape(savedDataset))
        self.x_data = torch.Tensor(savedDataset[:,1:3])
        print(np.shape(savedDataset[:,1:3]))
        print(self.x_data.shape)
        self.y_data = torch.Tensor(np.squeeze(np.transpose(savedDataset[:,3])))
        print(np.shape(savedDataset[:,3]))
        print(self.y_data.shape)
        print('Dataset loaded')

    def __len__(self):
        return list(self.x_data.size())[0]
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y