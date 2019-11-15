import torch
from torch.utils import data


class Dataset(data.Dataset):
  '''
  Characterizes a dataset for PyTorch
  '''

  def __init__(self, data):
    'Initialization'
    self.data = data

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    sent1 = self.data[index]['sent1']
    sent2 = self.data[index]['sent2']
    boundary = self.data[index]['boundary']

    return sent1, sent2, boundary
