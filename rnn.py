import torch
import torch.nn as nn
import torch.optim as optim


class SpeakerRNN(nn.Module):
    def __init__(self, device, emb_size, hidden_size, num_classes, batch_size, num_layers, bidirectionality=False):
        super().__init__()
        self.device = device
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = num_classes
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectionality

        self.gru = nn.GRU(emb_size, hidden_size, num_layers=self.num_layers,
                          bidirectional=bidirectionality, batch_first=True)

    def forward(self, sequence):
        #hidden_layer = self.init_hidden(self.batch_size)
        hidden_layer = self.init_hidden(len(sequence))  # should be 32 or 1
        hidden_layer = hidden_layer.to(self.device)
        self.gru.flatten_parameters()
        output, hidden = self.gru(sequence, hidden_layer)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).float()


class SpeakerClassifier(nn.Module):
    def __init__(self, device, input_size, output_size):
        super().__init__()
        self.device = device
        self.output = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        output = self.output(inputs)
        output = self.sigmoid(output)
        output = output.to(self.device)
        return output
