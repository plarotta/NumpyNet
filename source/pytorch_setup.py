import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

class MyNet(nn.Module):
  def __init__(self, in_size, out_size, n_hidden_layers, n_hidden_units):
    super(MyNet, self).__init__()
    self.in_size = in_size
    self.out_size = out_size
    self.n_hidden_layers = n_hidden_layers
    self.n_hidden_units = n_hidden_units

    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(self.in_size, self.n_hidden_units))
    for _ in range(self.n_hidden_layers-1):
      self.layers.append(nn.Linear(self.n_hidden_units,self.n_hidden_units))

    self.layers.append(nn.Linear(self.n_hidden_units,self.out_size))
    self.softmax = nn.Softmax(dim=0)
    self.relu = nn.ReLU()


  def forward(self, x):
    for layer_idx in range(len(self.layers)-1):
      x = self.layers[layer_idx](x)
      x = self.relu(x)
    x = self.layers[-1](x)
    return x

  def inference(self, x):
    for layer_idx in range(len(self.layers)-1):
      x = self.layers[layer_idx](x)
      x = self.relu(x)
    x = self.layers[-1](x)

    x = self.softmax(x)
    return x

class CustomDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_x, data_y):
    'Initialization'
    self.obs = data_x
    self.label = data_y

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.label)

  def __getitem__(self, index):
    'Generates one sample of data'
    X = self.obs[index,:]
    y = self.label[index]

    return X, y


    