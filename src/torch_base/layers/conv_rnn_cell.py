# imports
import torch
import torch.nn as nn

"""
Class ConvRnnCell

    Creates a single Convolutional Gated Recurrent Unit cell

        Args:
            mode       (string)  : LSTM, GRU or RNN
            input_dim  (int)     : number of channels in input
            hidden_dim (int)     : dimension of cell's hidden state 
            kernel_i   (int)     : size of filter applied to input to cell
            stride_i   (int)     : stride applied to input convolution
            kernel_h   (int)     : size of filter applied to the previous hidden state
            stride_h   (int)     : stride applied to hidden state convolution
            padding_i  (int)     : padding applied around input
            padding_h  (int)     : padding applied to previous hidden state
            bias       (boolean) : True to include bias terms in convolution
"""


class Conv1DRnnCell(nn.Module):

    def __init__(self,
                 mode,
                 input_dim,
                 hidden_dim,
                 kernel_i, stride_i, kernel_h, stride_h,
                 padding_i=0, dilation_i=1, groups_i=1,
                 padding_h=0, dilation_h=1, groups_h=1,
                 device=None, bias=True):

        super(Conv1DRnnCell, self).__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim

        if mode == "GRU":
            self.gate_size = 3 * hidden_dim
        elif mode == "LSTM":
            self.gate_size = 4 * hidden_dim
        elif mode == "RNN":
            self.gate_size = hidden_dim
        else:
            err_msg = "Cell type {} is not supported, try GRU, LSTM or RNN"
            raise ValueError(err_msg.format(mode))

        self.conv_in = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.gate_size,
            kernel_size=kernel_i,
            stride=stride_i,
            padding=padding_i,
            dilation=dilation_i,
            groups=groups_i,
            bias=bias
        )

        self.conv_h = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=self.gate_size,
            kernel_size=kernel_h,
            stride=stride_h,
            padding=padding_h,
            dilation=dilation_h,
            groups=groups_h,
            bias=bias
        )

        # GPU || CPU
        if not device:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

    def forward(self, x, h_prev):

        if self.mode == "GRU":
            x = self._gru_forward(x, h_prev)

        elif self.mode == "LSTM":
            x = self._lstm_forward(x, h_prev)

        elif self.mode == "RNN":
            x = self._rnn_forward(x, h_prev)

        return x

    def _gru_forward(self, x, h_prev):

        x = self.conv_in(x)
        h = self.conv_h(h_prev)
        x_z, x_r, x_n = torch.split(x, self.hidden_dim, dim=1)
        h_z, h_r, h_n = torch.split(h, self.hidden_dim, dim=1)

        # GRU logic
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        n = torch.tanh(x_n + r * h_n)
        h_t = (1 - z) * n + z * h_prev

        return h_t

    def _lstm_forward(self, x, h_prev):

        h_prev, c_prev = h_prev

        x = self.conv_in(x)
        h = self.conv_h(h_prev)
        x_i, x_f, x_g, x_o = torch.split(x, self.hidden_dim, dim=1)
        h_i, h_f, h_g, h_o = torch.split(h, self.hidden_dim, dim=1)

        # LSTM logic
        i = torch.sigmoid(x_i + h_i)
        f = torch.sigmoid(x_f + h_f)
        g = torch.tanh(x_g + h_g)
        o = torch.sigmoid(x_o + h_o)

        # cell & hidden state
        c_t = f*c_prev + i*g
        h_t = o*torch.tanh(c_t)

        return h_t, c_t

    def _rnn_forward(self, x, h_prev):

        x = self.conv_in(x)
        h = self.conv_h(h_prev)

        # RNN logic
        h_t = torch.tanh(x + h)

        return h_t

    def reset_parameters(self):
        self.conv_in.reset_parameters()
        self.conv_h.reset_parameters()
        return

    def init_hidden(self, batch_size, l):
        # init hidden state to zero
        h_0 = torch.zeros(batch_size, self.hidden_dim, l)
        return h_0.to(self.device)
