# imports
import torch
from random import shuffle
from torch.nn import ConstantPad1d

"""
Transforms applied per batch of MFCC's

"""

"""
Class Mfcc2Seq
    
    flatten mfcc tensor to seq
"""


class Mfcc2Seq(object):

    def __call__(self, mfcc):

        # flatten mfcc
        seq = mfcc.view(-1)

        return seq

    def __repr__(self):
        return self.__class__.__name__


"""
Class Numpy2Tensor

    convert a numpy array to torch.Tensor
"""


class Numpy2Tensor(object):

    def __call__(self, np_inpt):
        # Np -> Torch.Tensor
        ten = torch.from_numpy(np_inpt)
        return ten

    def __repr__(self):
        return self.__class__.__name__


"""
Class Numpy2Tensor

    convert a numpy array to torch.Tensor
"""


class Tensor2Numpy(object):

    def __call__(self, ten_inpt):
        # Np -> Torch.Tensor
        npy = ten_inpt.numpy()
        return npy

    def __repr__(self):
        return self.__class__.__name__


"""
Function mfcc_collate
    
    pads each mfcc in batch to longest seq in that batch

"""


def mfcc_collate(batch, pad_val=0):

    r"""Puts each data field into a tensor with outer dimension batch size"""

    if isinstance(batch[0], torch.Tensor):

        if batch[0].dim() > 2:
            # flatten mfcc's
            batch = [Mfcc2Seq()(b) for b in batch]

        # shuffle batch
        shuffle(batch)

        # max seq length
        seq_len = [b.size(0) for b in batch]
        max_seq = max(seq_len)

        # pad to max length
        batch = [
            ConstantPad1d((0, int(max_seq-b.size(0))), value=pad_val)(b) for b in batch
        ]

        # return tensor batch && original seq lengths

        return torch.stack(batch, 0), seq_len

    else:
        err_msg = "mfcc collate requires batch contain tensors, found {}"
        raise TypeError((
                err_msg.format(type(batch[0]))
            ))

    return
