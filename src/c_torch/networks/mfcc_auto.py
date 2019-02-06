# imports
import torch.nn as nn
from layers import Binarizer
from layers import StackedRnnBase


"""
MFCC GRU Autoencoder
    
"""


class MfccAuto(nn.Module):

    def __init__(self, bnd, input_size):
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
