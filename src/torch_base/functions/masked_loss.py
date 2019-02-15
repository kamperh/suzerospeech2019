# imports
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

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

    def forward(self, model_output, target, lengths):

        output, _ = pack_padded_sequence(
            model_output,
            lengths,
            batch_first=True
        )

        target, _ = pack_padded_sequence(
            target,
            lengths,
            batch_first=True
        )

        # loss on masked values
        loss = self.criterion(
            output,
            target=target
        )

        return loss
