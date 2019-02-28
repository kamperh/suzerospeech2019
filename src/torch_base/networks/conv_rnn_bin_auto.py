# imports
import os
import torch
import torch.nn as nn
from functions import GOF2Feat, Feat2GOF
from layers import Conv1DRnn, PixelShuffle1D, Binarizer


"""
Convolutional RNN Speech Autoencoder

    Args: 
        name         (string) : model name
        bnd          (int)    : bottleneck depth
        input_size   (int)    : input feature dimension
        target_size  (int)    : output feature size
        gof          (int)    : Group Of Features size
        rnn_mode     (string) : type of rnn to use, choices: GRU, LSTM, RNN
        speaker_cond (tuple)  : apply speaker conditioning at decoder by supplying (embed_dim, num_speakers) 
           
"""


class ConvRnnSpeechAuto(nn.Module):

    def __init__(self,
                 name, bnd,
                 input_size, target_size,
                 gof, rnn_mode="GRU", speaker_cond=None):

        super(ConvRnnSpeechAuto, self).__init__()

        # def model name
        self.name = name

        # bottle-neck depth
        self.bnd = bnd

        # Group of Features GOF
        self.gof = gof

        # Decay Factor
        self.df = 8

        # Encoder Network

        # (B, F, T) -> (B, F, T/8)
        self.encoder = nn.Sequential(

            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            Feat2GOF(
                gof_size=self.gof
            ),

            Conv1DRnn(
                mode=rnn_mode,
                input_dim=[64, 128, 256],
                hidden_dim=[128, 256, 512],
                kernel_i=[3, 3, 3], stride_i=[2, 2, 2],
                kernel_h=[1, 1, 1], stride_h=[1, 1, 1],
                padding_i=[1, 1, 1], dilation_i=[1, 1, 1], groups_i=[1, 1, 1],
                padding_h=[0, 0, 0], dilation_h=[1, 1, 1], groups_h=[1, 1, 1],
                bias=True,
                num_layers=3
            ),

            GOF2Feat(),

            nn.Conv1d(
                in_channels=512,
                out_channels=self.bnd,
                kernel_size=1,
                stride=1,
                padding=0,
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

            Feat2GOF(
                gof_size=self.gof // 8,
                time_first=True
            ),

            Conv1DRnn(
                mode=rnn_mode,
                input_dim=[512 + embed_dim],
                hidden_dim=[512],
                kernel_i=[3], stride_i=[1],
                kernel_h=[1], stride_h=[1],
                padding_i=[1], dilation_i=[1], groups_i=[1],
                padding_h=[0], dilation_h=[1], groups_h=[1],
                bias=True,
                num_layers=1
            ),

            GOF2Feat(),

            PixelShuffle1D(
                upscale_factor=2
            ),

            Feat2GOF(
                gof_size=self.gof // 4
            ),

            Conv1DRnn(
                mode=rnn_mode,
                input_dim=[256],
                hidden_dim=[512],
                kernel_i=[3], stride_i=[1],
                kernel_h=[1], stride_h=[1],
                padding_i=[1], dilation_i=[1], groups_i=[1],
                padding_h=[0], dilation_h=[1], groups_h=[1],
                bias=True,
                num_layers=1
            ),

            GOF2Feat(),

            PixelShuffle1D(
                upscale_factor=2
            ),

            Feat2GOF(
                gof_size=self.gof // 2
            ),

            Conv1DRnn(
                mode=rnn_mode,
                input_dim=[256],
                hidden_dim=[128],
                kernel_i=[3], stride_i=[1],
                kernel_h=[1], stride_h=[1],
                padding_i=[1], dilation_i=[1], groups_i=[1],
                padding_h=[0], dilation_h=[1], groups_h=[1],
                bias=True,
                num_layers=1
            ),

            GOF2Feat(),

            PixelShuffle1D(
                upscale_factor=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=target_size,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True
            ),

            nn.ReLU()

        )

    def forward(self, x, x_len=None, speaker_id=None):

        t_seq = x.size(1)

        # Dynamic pad
        x = self._dynamic_pad(x)

        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)

        # Conv1D Rnn Encoder
        x = self.encoder(x)

        # Binarize [-1, 1] -> {1, -1}
        b = self.binarizer(x)

        # Post-Binarization Conv1D
        x = self.dec_conv_bin(b)

        if self.condition:

            x = x.permute(0, 2, 1)

            x = torch.stack([
                torch.cat(
                    [x[:, t], self.speaker_embed(speaker_id).squeeze(1)],
                    dim=1
                ) for t in range(x.size(1))],
                dim=1
            )
        else:
            x = x.permute(0, 2, 1)

        # Target Synthesis Conv1D Rnn Decoder with PixelShuffle
        x = self.decoder(x)

        # Crop out any dynamic padding
        x = x[:, :, : t_seq]

        # (B, F, T) -> (B, T, F)
        x = x.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        return x, b

    def _dynamic_pad(self, x):

        if x.size(1) % self.gof != 0:

            pad_size = (x.size(1) // self.gof + 1) * self.gof - x.size(1)

            # apply padding to right
            x = nn.ConstantPad1d(
                padding=(0, pad_size),
                value=0
            )(x.permute(0, 2, 1))

            # (B, F, T) -> (B, T, F)
            x = x.permute(0, 2, 1)

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
