# imports
import torch

"""
Transforms applied per batch of MFCC's

"""

"""
Class CropMfcc

    crops Mfcc tensor to desired (T x freq)
"""


class CropMfcc(object):

    def __init__(self, t, freq):

        self.t = t
        self.f = freq

    def __call__(self, mfcc):
        # Np -> Torch.Tensor
        mfcc = mfcc[0:self.t, 0:self.f]
        return mfcc

    def __repr__(self):
        return self.__class__.__name__


"""
Class Numpy2Tensor

    convert a numpy array to float torch.Tensor
"""


class Numpy2Tensor(object):

    def __call__(self, np_inpt):
        # Np -> Torch.Tensor
        ten = torch.from_numpy(np_inpt)
        return ten.float()

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
