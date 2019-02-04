# imports
import torch.nn as nn

"""
 Stacked RNN Base base code
    
    Args:
        input_sizes   (list)   : list of layer input sizes
        output_sizes  (list)   : list of layer output sizes
        mode          (string) : type of cell to use ['LSTM', 'RNN', 'RNN']
        bias          (bool)   : include bias terms, default True
        bidirectional (bool)   : set to True for bidirectional RNN layers
        batch_first   (bool)   : (t, B, d) else (B, t, d)

"""


class StackedRnnBase(nn.Module):

    def __init__(self,
                 input_sizes,
                 hidden_sizes, mode="LSTM", bias=True,
                 bidirectional=False, batch_first=False):

        super(StackedRnnBase, self).__init__()

        if len(input_sizes) == len(hidden_sizes):
            self.num_layers = len(input_sizes)
        else:
            raise ValueError('len(input_sizes) != len(hidden_sizes)')

        self.batch_first = batch_first

        self.multilayer_rnn = nn.ModuleList([
            nn.RNNBase(
                mode=mode,
                input_size=input_sizes[i],
                hidden_size=hidden_sizes[i],
                bias=bias,
                bidirectional=bidirectional
            ) for i in range(self.num_layers)
        ])

    def forward(self, x):

        for i in range(self.num_layers):
            # layer output and states
            x, h = self.multilayer_rnn[i](x)

        # final output and state
        return x, h
