# imports
import os
import torch
import torch.nn as nn
from layers import Binarizer
from layers import Conv1DRnn

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
                 input_size, target_size,
                 gof=10, speaker_cond=None):

        super(ConvSpeechAuto, self).__init__()

        # def model name
        self.name = name

        # bottle-neck depth
        self.bnd = bnd

        # Group of Features GOF
        self.gof = gof

        # Encoder Network

        # (B, F, T) -> (B, F, T/2)
        self.encoder = Conv1DRnn(
            mode="GRU",
            input_dim=[input_size],
            hidden_dim=[64],
            kernel_i=[3], stride_i=[2],
            kernel_h=[1], stride_h=[1],
            padding_i=[1], dilation_i=[1], groups_i=[1],
            padding_h=[0], dilation_h=[1], groups_h=[1],
            bias=True,
            num_layers=1
        )

        # Pre-ConvBin Layer
        self.enc_conv_bin = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
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
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.Tanh()

        )

        self.depth_to_space = nn.Sequential(

            # (B, F, T/2) -> (B, F, T)
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True
            ),

            nn.ReLU()
        )

        if speaker_cond is None:
            self.condition = False

            # Conv1D Decoder No Speaker Conditioning
            self.decoder = Conv1DRnn(
                mode="GRU",
                input_dim=[self.bnd],
                hidden_dim=[64],
                kernel_i=[1], stride_i=[1],
                kernel_h=[1], stride_h=[1],
                padding_i=[1], dilation_i=[1], groups_i=[1],
                padding_h=[0], dilation_h=[1], groups_h=[1],
                bias=True,
                num_layers=1
            )

        else:

            self.condition = True
            embed_dim, n_speakers = speaker_cond

            # Speaker Embedding Table
            self.speaker_embed = nn.Embedding(
                embedding_dim=embed_dim,
                num_embeddings=n_speakers
            )

            # Conv1D Decoder No Speaker Conditioning
            self.decoder = Conv1DRnn(
                mode="GRU",
                input_dim=[self.bnd + embed_dim],
                hidden_dim=[64],
                kernel_i=[1], stride_i=[1],
                kernel_h=[1], stride_h=[1],
                padding_i=[1], dilation_i=[1], groups_i=[1],
                padding_h=[0], dilation_h=[1], groups_h=[1],
                bias=True,
                num_layers=1
            )

        # Output Conv1D
        self.conv_out = nn.Sequential(

            nn.Conv1d(
                in_channels=64,
                out_channels=target_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.Tanh()

        )

    def forward(self, x, speaker_id=None):

        # extract dimensions
        batch_size, t, f = x.size()

        # split into GOF
        x = torch.stack(
            torch.split(
                x,
                split_size_or_sections=self.gof,
                dim=1
            ),
            dim=1
        )

        # (B, S, T, F) -> (B, S, F, T)
        x = x.permute(0, 1, 3, 2)

        # Conv1D Rnn Encoder
        x = self.encoder(x)

        # (B, S, F, T) -> (B, T, F) -> (B, F, T)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(
            batch_size, -1, x.size(3)
        )
        x = x.permute(0, 2, 1)

        # Pre-Binarization Conv1D [-1, 1]
        x = self.enc_conv_bin(x)

        # Binarize [-1, 1] -> {1, -1}
        b = self.binarizer(x)

        # Post-Binarization Conv1D
        x = self.dec_conv_bin(b)

        # Depth-to-Space Unit
        x = self.depth_to_space(x)

        #
        x = x.permute(0, 2, 1)

        # split into GOF
        x = torch.stack(
            torch.split(
                x,
                split_size_or_sections=self.gof,
                dim=1
            ),
            dim=1
        )

        # Target Synthesis Conv1D Rnn Decoder
        x = self.decoder(x)

        # (B, S, F, T) -> (B, T, F) -> (B, F, T)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(
            batch_size, -1, x.size(3)
        )
        x = x.permute(0, 2, 1)

        # Conv Out
        x = self.conv_out(x)

        return x, b

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

x = torch.randn(2, 10, 13).cuda()
net = ConvSpeechAuto("", 10, 13, 15, gof=10, speaker_cond=None).cuda()
y = net(x)