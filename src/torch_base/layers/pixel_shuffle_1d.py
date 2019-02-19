# imports
import torch.nn as nn

"""
Pixel Shuffle 1D

    increases Speech frame resolution (depth-to-space unit)

        Args:
            upscale_factor (int) : factor to increase spatial resolution and decrease channel depth by

        Ref:
            https://arxiv.org/abs/1609.05158
"""


class PixelShuffle1D(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, x):

        # extract dimension
        batch_size, channels, in_length = x.size()
        channels //= self.upscale_factor

        # new dimensions
        out_length = in_length * self.upscale_factor

        x = x.contiguous().view(
            batch_size, channels, self.upscale_factor, in_length
        )

        x = x.permute(0, 1, 3, 2).contiguous()

        x = x.view(batch_size, channels, out_length)

        return x

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
