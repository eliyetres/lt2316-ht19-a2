import torch
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, sentences, boundaries):
        'Initialization'
        self.boundaries = boundaries
        self.sentences = sentences

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.sentences)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.sentences[index]
        y = self.boundaries[index]

        return X, y
