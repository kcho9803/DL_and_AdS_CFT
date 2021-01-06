# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:32:57 2021

@author: Kyungchan Cho
"""

import data_generator
import torch
from torch.utils.data import DataLoader

# Generate dataset
dataset = data_generator.GeneratedDataset()
dataloader = DataLoader(dataset, batch_size = 2000, shuffle = True)