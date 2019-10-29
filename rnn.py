import torch.nn as nn
import torch
import torch.optim as optim
import config

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.device = torch.device(config.DEVICE)

        self.emb = nn.Embedding(vocab_size, input_size)
        
        self.lstm = nn.LSTM(input_size, hidden_size) #the only required parameters, should we add more? What about numer of layers? Bidirectionality?
        self.linear = nn.Linear(hidden_size, vocab_size)        
        self.softmax = nn.LogSoftmax(dim=1) #what dimension here?

    def setup_model(self, input_size, hidden_size, output_size, vocab_size):
        model = RNN(input_size, hidden_size, output_size, vocab_size)
        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters())

        return model

    def setup(self):
        # Loss function and optimizer                                                                                                                                               
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    
    def forward(self, sentence):

        #Initializing the hidden layer
        initHidden = initHidden()

        #Send sentence to embedding layer    
        output = self.emb(sentence)

        #Send the embedding output into the LSTM
        output, hidden = self.lstm(output, init_hidden)

        #Send the LSTM output into the linear layer
        output = self.linear(output)

        #Calling softmax
        output = self.softmax(output)

        return output, hidden
        
    def initHidden(self):
        #How do we initiate the hidden?
        return torch.zeros(1, self.hidden_size)

    def train(self, sentence, boundary):

        #Model is initiated when calling this function
        #The epoch and batch runs outside this function
        #The input here is a number of sentences and their labels
        #It is already sent to device
                            
        # set optimizer to zero
        self.optimizer.zero_grad()
        
        # forward pass
        result = self.forward(sentence) #or simply self?
        
        # add loss
        loss = self.criterion(result, boundary) #What is result, should there be an index here?
        
        # compute gradients
        loss.backward()
        
        # backprop
        self.optimizer.step()
