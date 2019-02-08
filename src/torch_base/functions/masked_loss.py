# imports
import torch.nn as nn

"""
Masked Loss

    used to mask padding of variable length inputs when calculating loss
    
    Ref:
        https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    
"""


class MaskedLoss(nn.Module):

    def __init__(self, criterion):
        super(MaskedLoss, self).__init__()

        self.criterion = criterion

    def forward(self, output, target):

        # select non-padded values
        output = output[output != 0.0]

        target = target[target != 0.0]

        # loss on masked values
        loss = self.criterion(
            output,
            target=target
        )

        return loss
