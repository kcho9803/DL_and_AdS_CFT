import torch
import torch.nn as NN
import nn
import data_generator
from torch.utils.data import DataLoader

dataset = data_generator.LoadedDataset()
dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)

generator = nn.Net(generate = True)
network = nn.Net(generate = False)

print('Output test for 2000 data points:')
i = 0
L1 = 0
for x, y in dataloader:
    gen_y = generator(x)
    net_y = network(x)
    #print('Data {}: Generator = {}, Network = {}'.format(i+1,gen_y.item(),net_y.item()))
    i += 1
    if torch.abs(net_y - y) > 10e-3:
        print('Data {}: Label = {}, Pred = {}'.format(i, y.item(), net_y.item()))
    L1 += torch.abs(net_y - y)
    if i > 1999:
        break
print(L1.item())
