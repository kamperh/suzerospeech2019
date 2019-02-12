# imports
import os
import torch
import torch.nn as nn
from layers import Binarizer
from layers import StackedRnnBase


"""
MFCC GRU Autoencoder
    
"""


class MfccAuto(nn.Module):

    def __init__(self, bnd, input_size=13):
        super(MfccAuto, self).__init__()

        self.name = "MfccAuto"

        # bottle-neck depth
        self.bnd = bnd

        # Encoder Network
        self.encoder = StackedRnnBase(
            input_size=input_size,
            hidden_sizes=[
                64, 256, 512
            ],
            mode="GRU"

        )

        self.linear_encoder = nn.Sequential(
                nn.Linear(
                    in_features=512,
                    out_features=self.bnd
                ),
                nn.Tanh()
        )

        # Binarization Layer
        self.binarizer = Binarizer()

        self.linear_decoder = nn.Sequential(
                nn.Linear(
                    in_features=self.bnd,
                    out_features=512
                ),
                nn.Tanh()
        )

        # Decoder Network
        self.decoder = StackedRnnBase(
            input_size=512,
            hidden_sizes=[
                512, 256, 64, input_size
            ],
            mode="GRU"
        )

    def forward(self, x, x_len=None):

        # RNN encoder
        e, h_e = self.encoder(x, x_len)

        # Linear Encoding
        e_b = torch.stack(
            [
                self.linear_encoder(
                    e[:, t]
                ) for t in range(e.size(1))
            ],
            dim=1
        )

        # Binarization
        b = self.binarizer(e_b)

        # Linear Decoding
        d_b = torch.stack(
            [
                self.linear_decoder(
                    b[:, t]
                ) for t in range(b.size(1))
            ],
            dim=1
        )

        # RNN decoder
        d, h_d = self.decoder(d_b, x_len)

        # ret decoding & bits
        return d, b[b != 0]

    def load(self, save_file):

        save_file = os.path.expanduser(
            save_file
        )

        if not os.path.isfile(save_file):
            err_msg = "{} is not a valid file location"
            raise ValueError(
                err_msg.format(save_file)
            )
        else:
            self.load_state_dict(
                torch.load(save_file)
            )
        return self
