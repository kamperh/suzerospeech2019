# imports
import os
import torch
import torch.nn as nn
from layers import Conv1DRnn, PixelShuffle1D, Binarizer


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
            input_dim=[input_size, 64, 256, 512],
            hidden_dim=[64, 256, 512, 512],
            kernel_i=[3], stride_i=[2],
            kernel_h=[1], stride_h=[1],
            padding_i=[1], dilation_i=[1], groups_i=[1],
            padding_h=[0], dilation_h=[1], groups_h=[1],
            bias=True,
            num_layers=4
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
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.Tanh()

        )

        if speaker_cond is None:
            self.condition = False

            # Conv1D Decoder No Speaker Conditioning
            self.decoder = Conv1DRnn(
                mode="GRU",
                input_dim=[64],
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
                input_dim=[64 + embed_dim],
                hidden_dim=[64],
                kernel_i=[1], stride_i=[1],
                kernel_h=[1], stride_h=[1],
                padding_i=[1], dilation_i=[1], groups_i=[1],
                padding_h=[0], dilation_h=[1], groups_h=[1],
                bias=True,
                num_layers=1
            )

        self.depth_to_space = PixelShuffle1D(
            upscale_factor=2
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

            nn.ReLU()

        )

    def forward(self, x, x_len=None, speaker_id=None):

        t_seq = x.size(1)

        # Dynamic pad
        x = self._dynamic_pad(x)

        # Feat -> GOF
        x = self._feat2gof(x, self.gof)

        # Conv1D Rnn Encoder
        x = self.encoder(x)

        # GOF to single Feature
        x = self._gof2feat(x)

        # Pre-Binarization Conv1D [-1, 1]
        x = self.enc_conv_bin(x)

        # Binarize [-1, 1] -> {1, -1}
        b = self.binarizer(x)

        # Post-Binarization Conv1D
        x = self.dec_conv_bin(b)

        # Depth-to-Space Unit
        x = self.depth_to_space(x)

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

        # Feat_hat -> GOF_hat
        x = self._feat2gof(x, self.gof)

        # Target Synthesis Conv1D Rnn Decoder
        x = self.decoder(x)

        # GOF to single Feature
        x = self._gof2feat(x)

        # Conv Out
        x = self.conv_out(x)

        # (B, F, T) -> (B, T, F)
        x = x.permute(0, 2, 1)

        # Crop any dynamic padding
        x = x[:, :t_seq]

        return x, b.permute(0, 2, 1)

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

    @staticmethod
    def _feat2gof(feat, gof_size):

        # split into GOF (B, T, F) -> (B, GOF, F, T)
        gof = torch.stack(
            torch.split(
                feat,
                split_size_or_sections=gof_size,
                dim=1
            ),
            dim=1
        ).permute(0, 1, 3, 2)
        return gof

    @staticmethod
    def _gof2feat(gof):
        # (B, S, F, T) -> (B, T, F) -> (B, F, T)

        batch_size, _, _, _ = gof.size()

        feat = gof.permute(0, 1, 3, 2).contiguous()
        feat = feat.view(
            batch_size, -1, feat.size(3)
        )
        feat = feat.permute(0, 2, 1)

        return feat

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
