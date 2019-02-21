import torch
import torch.nn as nn

"""
Features to Group Of Features
    
    subdivides a single speech feature into a GOF
        
        Args:
            gof_size   (int)  : size of feature grouping
            time_first (bool) : default False expects input feature to be (B, F, T)
"""


class Feat2GOF(nn.Module):

    def __init__(self, gof_size, time_first=False):
        super(Feat2GOF, self).__init__()

        self.gof_size = gof_size
        self.time_first = time_first

    def forward(self, x):

        if not self.time_first:
            # (B, F, T) -> (B, T, F)
            x = x.permute(0, 2, 1)

        x = torch.stack(
            torch.split(
                x,
                split_size_or_sections=self.gof_size,
                dim=1
            ),
            dim=1
        ).permute(0, 1, 3, 2)

        # (B, F, T)

        return x


"""
Group Of Features to Features

    combines a GOF into a single feature

"""


class GOF2Feat(nn.Module):

    def __init__(self):
        super(GOF2Feat, self).__init__()

    def forward(self, x):
        # (B, GOF, F, T) -> (B, F, T)

        batch_size, _, feat_size, _ = x.size()

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            batch_size, feat_size, -1
        )

        return x
