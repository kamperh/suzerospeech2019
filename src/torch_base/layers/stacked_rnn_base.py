# imports
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
 Stacked RNN Base base code
    
    Args:
        input_size    (int)    : input size
        hidden_sizes  (list)   : list of layer hidden sizes
        mode          (string) : type of cell to use ['LSTM', 'RNN', 'RNN']
        bias          (bool)   : include bias terms, default True
        bidirectional (bool)   : set to True for bidirectional RNN layers
        batch_first   (bool)   : (t, B, d) else (B, t, d)

"""


class StackedRnnBase(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_sizes, mode="LSTM", bias=True,
                 bidirectional=False, batch_first=True):

        super(StackedRnnBase, self).__init__()

        self.num_layers = len(hidden_sizes)

        self.batch_first = batch_first

        # layer input sizes
        input_sizes = [input_size].append(hidden_sizes[:-1])

        self.multilayer_rnn = nn.ModuleList([
            nn.RNNBase(
                mode=mode,
                input_size=input_sizes[i],
                hidden_size=hidden_sizes[i],
                bias=bias,
                bidirectional=bidirectional
            ) for i in range(self.num_layers)
        ])

    def forward(self, x, x_len=None):

        if x_len is not None:
            # pack sequences
            x = pack_padded_sequence(
                x,
                x_len,
                batch_first=self.batch_first
            )

        for i in range(self.num_layers):
            # layer output and states
            x, h = self.multilayer_rnn[i](x)

        if x_len is not None:
            # pad packed seq
            x, _ = pad_packed_sequence(
                x,
                padding_value=0.0,
                batch_first=self.batch_first
            )

        # final output and state
        return x, h
