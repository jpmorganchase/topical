import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import Trainer


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=192, out_features=420
        )
        self.encoder_output_layer = nn.Linear(
            in_features=420, out_features=192
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=192, out_features=420
        )
        self.decoder_output_layer = nn.Linear(
            in_features=420, out_features=192
        )

    def forward(self, features,hidden=None):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code


class Encoder(nn.Module):
    """
    RNN Sequence Encoder
    """
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
                 bidirectional=True, rnn_type='GRU',use_pca=True,num_pca_components = 192):
        super(Encoder, self).__init__()
        self.use_pca = use_pca
        self.bidirectional = bidirectional
        self.rnn_type=rnn_type
        #assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, 'LSTM') # fetch constructor from torch.nn, cleaner than if
        gru = rnn_cell(int(embedding_dim), hidden_dim, nlayers,
                            dropout=dropout, bidirectional=bidirectional)
        print("{} Params: {}".format(rnn_type,sum(p.numel() for p in gru.parameters())))
        print("AE Params: {}".format(sum(p.numel() for p in AE().parameters())))
        if rnn_type in ['GRU','RNN','LSTM']:
            rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
            self.cell = rnn_cell(int(embedding_dim), hidden_dim, nlayers,
                                dropout=dropout, bidirectional=bidirectional)
            print("Using RNN encoder")
        else:
            self.cell = AE()
            print("Using MLP Autoencoder")
        if not self.use_pca: # Add a linear layer to reduce dimensionality of the embedding vectors
            print(embedding_dim,'embedding_dim')
            self.linear_layer = nn.Linear(embedding_dim,num_pca_components)

    def forward(self, input, hidden=None):
        if not self.use_pca:
            out = self.linear_layer(input)
        out = self.cell(input, hidden)
        return out


class Attention(nn.Module):
    """
    Vanilla Attention Mechanism
    """
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values, attention_masks):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

    """def forward(self, query, keys, values, attention_masks):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        attention_masks = torch.diag_embed(attention_masks, offset=0, dim1=-2, dim2=-1)
        keys = torch.moveaxis(keys, 0, -1)
        keys = torch.bmm(keys, attention_masks)
        values = torch.moveaxis(values, 0, -1)
        values = torch.bmm(values, attention_masks)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        #keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
        values = values.moveaxis(1,-1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination"""

class Classifier(nn.Module):
    """
    Model Classification Head
    """
    def __init__(self, encoder, attention, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, num_classes)
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input):
        input['embeddings'] = torch.transpose(input['embeddings'], 0, 1)
        outputs, hidden = self.encoder(input['embeddings'])
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state
        if self.encoder.bidirectional and self.encoder.rnn_type in ['GRU','LSTM']: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
        energy, linear_combination = self.attention(hidden, outputs, outputs, input['attention_mask']) # hidden, outputs, outputs, attention_mask
        logits = self.decoder(linear_combination)
        return logits, energy, linear_combination

class ClassifierTrainer(Trainer):
    """
    Classification Task Trainer - inheriting from HuggingFace Trainer class (see HG documentation)
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(inputs)
        logits = outputs[0]
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits,
                        labels)
        return (loss, outputs) if return_outputs else loss

