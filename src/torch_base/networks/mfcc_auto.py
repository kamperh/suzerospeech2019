# imports
import os
import torch
import torch.nn as nn
from torch_base.layers import Binarizer
from torch_base.layers import StackedRnnBase


"""
MFCC GRU Autoencoder
    
"""


class MfccAuto(nn.Module):

    def __init__(self, bnd, input_size=39):
        super(MfccAuto, self).__init__()

        self.name = "MfccAuto"

        # bottle-neck depth
        self.bnd = bnd

        # Encoder Network
        self.encoder = StackedRnnBase(
            input_sizes=[
                input_size, 20, 60
            ],
            hidden_sizes=[
                20, 60, self.bnd
            ],
            mode='GRU'

        )

        # Binarization Network
        self.binarizer = Binarizer()

        # Decoder Network
        self.decoder = StackedRnnBase(
            input_sizes=[
                self.bnd, 120, 60, 20,
            ],
            hidden_sizes=[
                120, 60, 20, input_size
            ],
            mode='GRU'
        )

    def forward(self, x, x_len):

        # encode & decode using RNN
        e, h_e = self.encoder(x, x_len)
        b = self.binarizer(e)
        d, h_d = self.decoder(b, x_len)

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
        return
