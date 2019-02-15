# imports
import os
import torch
import torch.nn as nn
from layers import Binarizer
from layers import LinearRnnBase


"""
MFCC GRU Autoencoder
    
"""


class SpeechAuto(nn.Module):

    def __init__(self, name,
                 bnd,
                 input_size,
                 target_size, speaker_cond=None):

        super(SpeechAuto, self).__init__()

        self.name = name

        # bottle-neck depth
        self.bnd = bnd

        # Encoder Network
        self.encoder = LinearRnnBase(
            input_sizes=[input_size],
            hidden_sizes=[64],
            mode="GRU"

        )

        self.linear_encoder = nn.Sequential(
                nn.Linear(
                    in_features=64,
                    out_features=self.bnd,
                    bias=True
                ),
                nn.Tanh()
        )

        # Binarization Layer
        self.binarizer = Binarizer()

        self.linear_decoder = nn.Sequential(
                nn.Linear(
                    in_features=self.bnd,
                    out_features=64,
                    bias=True
                ),
                nn.Tanh()
        )

        if speaker_cond is not None:

            # apply speaker conditioning
            self.condition = True
            embed_dim, num_speakers = speaker_cond

            self.speaker_embed = nn.Embedding(
                embedding_dim=embed_dim,
                num_embeddings=num_speakers,
            )

            # Decoder Network
            self.decoder = LinearRnnBase(
                input_sizes=[64 + embed_dim],
                hidden_sizes=[target_size],
                mode="GRU"
            )
        else:
            self.condition = False

            self.decoder = LinearRnnBase(
                input_sizes=[64],
                hidden_sizes=[target_size],
                mode="GRU"
            )

    def forward(self, x, x_len=None, speaker_id=None):

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

        if self.condition:
            # Concat Speaker Embedding
            d_b = torch.stack(
                [
                    torch.cat(
                        [
                            d_b[:, t],
                            self.speaker_embed(speaker_id).squeeze(1)
                        ],
                        dim=1
                    ) for t in range(d_b.size(1))
                ],
                dim=1
            )

        # RNN decoder
        d, h_d = self.decoder(d_b, x_len)

        # ret decoding & bits
        return d, b

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
