import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import Trainer

class SequenceEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
                 bidirectional=True, rnn_type='GRU'):
        super(SequenceEncoder, self).__init__()
        self.bidirectional = bidirectional
        rnn_cell = getattr(nn, rnn_type)
        self.rnn = rnn_cell(int(embedding_dim), hidden_dim, nlayers,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, hidden=None):
        outputs, hidden = self.rnn(input, hidden)
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state
        if self.rnn.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
        return hidden



