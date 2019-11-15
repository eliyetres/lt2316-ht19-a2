import torch
import torch.nn as nn
import torch.optim as optim

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, vocab_size):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size
#         self.device = torch.device(config.DEVICE)

#         self.emb = nn.Embedding(vocab_size, input_size)

#         self.lstm = nn.LSTM(input_size, hidden_size) #the only required parameters, should we add more? What about numer of layers? Bidirectionality?
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         self.softmax = nn.LogSoftmax(dim=1) #what dimension here?

#     def setup_model(self, input_size, hidden_size, output_size, vocab_size):
#         model = RNN(input_size, hidden_size, output_size, vocab_size)
#         # Loss function and optimizer
#         self.criterion = nn.BCELoss()
#         self.optimizer = torch.optim.Adam(model.parameters())

#         return model

#     def setup(self):
#         # Loss function and optimizer
#         self.criterion = nn.BCELoss()
#         self.optimizer = torch.optim.Adam(self.parameters())

#     def forward(self, sentence):

#         #Initializing the hidden layer
#         initHidden = initHidden()

#         #Send sentence to embedding layer
#         output = self.emb(sentence)

#         #Send the embedding output into the LSTM
#         output, hidden = self.lstm(output, init_hidden)

#         #Send the LSTM output into the linear layer
#         output = self.linear(output)

#         #Calling softmax
#         output = self.softmax(output)

#         return output, hidden

#     def initHidden(self):
#         #How do we initiate the hidden?
#         return torch.zeros(1, self.hidden_size)

#     def train(self, sentence, boundary):

#         #Model is initiated when calling this function
#         #The epoch and batch runs outside this function
#         #The input here is a number of sentences and their labels
#         #It is already sent to device

#         # set optimizer to zero
#         self.optimizer.zero_grad()

#         # forward pass
#         result = self.forward(sentence) #or simply self?

#         # add loss
#         loss = self.criterion(result, boundary) #What is result, should there be an index here?

#         # compute gradients
#         loss.backward()

#         # backprop
#         self.optimizer.step()


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

        self.gru = nn.GRU(emb_size, hidden_size, num_layers=self.num_layers, bidirectional=bidirectionality, batch_first=True)

    def forward(self, sequence):
        hidden_layer = self.init_hidden(self.batch_size)
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