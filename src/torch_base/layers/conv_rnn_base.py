import torch
import numbers
import warnings
import torch.nn as nn
from .conv_rnn_cell import Conv1DRnnCell


class Conv1DRnn(nn.Module):

    def __init__(self,
                 mode,
                 input_dim,
                 hidden_dim,
                 kernel_i, stride_i, padding_i, dilation_i, groups_i,
                 kernel_h, stride_h, padding_h, dilation_h, groups_h,
                 num_layers=1,
                 bias=False,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False):

        super(Conv1DRnn, self).__init__()

        # rnn parameters
        self.mode = mode
        self.bias = bias
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # sanity check dropout value
        self.dropout = dropout
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError("Dropout should be a number in range [0, 1]!")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        # def number of directions
        self.bidirectional = bidirectional

        # def Convolutional RNN cells
        self.rnn_cells_forward = nn.ModuleList([
            Conv1DRnnCell(
                mode,
                input_dim[n],
                self.hidden_dim[n],
                *(kernel_i[n], stride_i[n], padding_i[n], dilation_i[n], groups_i[n]),
                *(kernel_h[n], stride_h[n], padding_h[n], dilation_h[n], groups_h[n]),
                bias=bias
            ) for n in self.num_layers
        ])

        if bidirectional:
            self.rnn_cells_reverse = nn.ModuleList([
                Conv1DRnnCell(
                    input_dim[n],
                    self.hidden_dim[n],
                    *(kernel_i[n], stride_i[n], padding_i[n], dilation_i[n], groups_i[n]),
                    *(kernel_h[n], stride_h[n], padding_h[n], dilation_h[n], groups_h[n]),
                    bias=bias
                ) for n in self.num_layers
            ])

    def forward(self, x):

        if not self.batch_first:
            # (T, B, C, L) -> (B, T, C, L)
            x = input.permute(1, 0, 2, 3)

        # def seq length
        t_seq = x.size(1)

        for n in range(self.num_layers):

            forward_seq = []
            reverse_seq = []

            # init state None
            r_ht = ht = None

            for t in range(t_seq):

                ht = self.rnn_cells_forward[n](
                    x[:, t, :, :], ht
                )
                forward_seq.append(ht)

                if self.bidirectional:
                    r_ht = self.rnn_cells_reverse[n](
                        x[:, (t_seq - t - 1), :, :, :], r_ht
                    )
                    reverse_seq.append(r_ht)

            x = torch.stack(
                forward_seq,
                dim=1
            )

            if self.bidirectional:
                reverse_seq = torch.stack(
                    reverse_seq,
                    dim=1
                )

                # concat reverse and forward outputs
                x = torch.cat(
                    [x, reverse_seq],
                    dim=2
                )

            if n < self.num_layers - 1 and self.dropout != 0:
                # add dropout to all but last layer
                x = nn.Dropout(p=self.dropout)

        return x




