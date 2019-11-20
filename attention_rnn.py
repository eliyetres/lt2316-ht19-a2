import torch
import torch.nn as nn


class SpeakerAttentionRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, dev):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dev = dev

        self.gru = nn.GRU(emb_size, hidden_size,
                          num_layers=num_layers, batch_first=True)

    def forward(self, seq):
        hidden_layer = self.init_hidden(len(seq))
        hidden_layer = hidden_layer.to(self.dev)
        output, hidden = self.gru(seq, hidden_layer)
        return output, hidden

    def init_hidden(self, seq_len):
        return torch.zeros(self.num_layers, seq_len, self.hidden_size)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        energies = self.fc(rnn_output)
        energies = energies.squeeze(dim=2)
        attention_weights = torch.softmax(energies, dim=1)
        rnn_output = rnn_output.permute(2, 0, 1)
        weighted_output = attention_weights * rnn_output
        output = weighted_output.sum(dim=2)
        output = output.t()
        return output


class SpeakerAttentionClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, device=None):
        super().__init__()
        self.attn_layer = Attention(hidden_size)
        self.fc_output = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, first_seq, second_seq):
        seq = torch.cat((first_seq, second_seq), dim=2)
        output = self.attn_layer(seq)
        output = self.fc_output(output)
        output = self.sigmoid(output)
        return output
