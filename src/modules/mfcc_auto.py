# imports
import torch.nn as nn
from base_torch.layers import Binarizer
from base_torch.layers import StackedRnnBase

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
                input_size, 20, 60, 120
            ],
            hidden_sizes=[
                20, 60, 120, self.bnd
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

    def forward(self, x):

        # encode & decode
        e = self.encoder(x)
        b = self.binarizer(e)
        d = self.decoder(b)

        # ret decoding & bits
        return d, b

