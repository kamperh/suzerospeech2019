# imports
import os
import torch
import torch.nn as nn
from layers import Binarizer
from layers import StackedRnnBase

"""
Convolutional Speech Binarizer

    input  : Mfcc's / Filter-Banks
    output : Mfcc's / Filter-Banks
    
    Rnn type : GRU
    Supports Speaker Conditioning

"""


class ConvSpeechAuto(nn.Module):

    def __init__(self,
                 name, bnd,
                 input_size, cond_speakers=None):

        super(ConvSpeechAuto, self).__init__()

        # def model name
        self.name = name

        # bottle-neck depth
        self.bnd = bnd

        self.speaker_cond = False

        # Encoder Network
        self.encoder = StackedRnnBase(
            input_sizes=[
                input_size, 64, 256
            ],
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

        if cond_speakers:

            # apply speaker conditioning
            self.name = "CondMfccAuto"
            self.speaker_cond = True

            # embedding dimension
            embed_dim = 100

            self.speaker_embed = nn.Embedding(
                num_embeddings=cond_speakers,
                embedding_dim=embed_dim
            )

            # Decoder Network
            self.decoder = StackedRnnBase(
                input_sizes=[
                    512 + embed_dim, 512, 256, 64
                ],
                hidden_sizes=[
                    512, 256, 64, input_size
                ],
                mode="GRU"
            )
        else:
            self.decoder = StackedRnnBase(
                input_sizes=[
                    512, 512, 256, 64
                ],
                hidden_sizes=[
                    512, 256, 64, input_size
                ],
                mode="GRU"
            )

    def forward(self, x, x_len=None, speaker_ids=None):

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

        if self.speaker_cond:
            # Concat Speaker Embedding
            d_b = torch.stack(
                [
                    torch.cat(
                        [
                            d_b[:, t],
                            self.speaker_embed(speaker_ids).squeeze(1)
                        ],
                        dim=1
                    ) for t in range(d_b.size(1))
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
