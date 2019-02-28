# imports
import os
import torch
import torch.nn as nn
from layers import PixelShuffle1D, Binarizer

"""
Convolutional Speech Autoencoder

    Args: 
        name         (string) : model name
        bnd          (int)    : bottleneck depth
        input_size   (int)    : input feature dimension
        target_size  (int)    : output feature size
        speaker_cond (tuple)  : apply speaker conditioning at decoder by supplying (embed_dim, num_speakers) 
           
"""


class ConvSpeechAuto(nn.Module):

    def __init__(self, name,
                 bnd, input_size,
                 target_size, speaker_cond=None):

        super(ConvSpeechAuto, self).__init__()

        # def model name
        self.name = name

        # bottle-neck depth
        self.bnd = bnd

        # decay factor
        self.df = 8

        # Encoder Network

        # (B, F, T) -> (B, F, T/8)
        self.encoder = nn.Sequential(

            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=512,
                out_channels=self.bnd,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.Tanh()

        )

        # Binarization Layer
        self.binarizer = Binarizer()

        # Post-ConvBin Layer
        self.dec_conv_bin = nn.Sequential(

            nn.Conv1d(
                in_channels=self.bnd,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.Tanh()

        )

        if speaker_cond is None:
            # don't condition
            self.condition = False
            embed_dim = 0

        else:
            # create speaker embedding
            self.condition = True
            embed_dim, n_speakers = speaker_cond

            # Speaker Embedding Table
            self.speaker_embed = nn.Embedding(
                embedding_dim=embed_dim,
                num_embeddings=n_speakers
            )

        self.decoder = nn.Sequential(
            # (B, F, T/8) -> (B, F, T)
            nn.Conv1d(
                in_channels=512 + embed_dim,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.ReLU(),

            PixelShuffle1D(
                upscale_factor=2
            ),

            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.ReLU(),

            PixelShuffle1D(
                upscale_factor=2
            ),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.ReLU(),

            PixelShuffle1D(
                upscale_factor=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=target_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.ReLU()

        )

    def forward(self, x, x_len=None, speaker_id=None):

        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        t_seq = x.size(2)

        # Dynamic pad
        x = self._dynamic_pad(x)

        # Conv1D Rnn Encoder
        x = self.encoder(x)

        # Binarize [-1, 1] -> {1, -1}
        b = self.binarizer(x)

        # Post-Binarization Conv1D
        x = self.dec_conv_bin(b)

        if self.condition:
            # Speaker Conditioning
            x = x.permute(0, 2, 1)

            x = torch.stack([
                torch.cat(
                    [x[:, t], self.speaker_embed(speaker_id).squeeze(1)],
                    dim=1
                ) for t in range(x.size(1))],
                dim=1
            ).permute(0, 2, 1)

        # Target Synthesis with PixelShuffle
        x = self.decoder(x)

        # Crop dynamic padding
        x = x[:, :, : t_seq]

        # (B, F, T) -> (B, T, F)
        x = x.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        return x, b

    def _dynamic_pad(self, x):

        if x.size(2) % self.df != 0:
            pad_size = (x.size(2) // self.df + 1) * self.df - x.size(2)

            # apply padding to right
            x = nn.ConstantPad1d(
                padding=(0, pad_size),
                value=0
            )(x)

        return x

    def load(self, save_file):

        save_file = os.path.expanduser(
            save_file
        )

        if not os.path.isfile(save_file):
            err_msg = "{} file location D.N.E"
            raise FileNotFoundError(
                err_msg.format(save_file)
            )
        else:
            self.load_state_dict(
                torch.load(save_file)
            )
        return self

