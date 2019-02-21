# imports
import torch

"""
Transforms applied to SpeechData

"""

"""
Class CropSpeech

    crops Speech Feature Tensor to desired (T x F)
"""


class CropSpeech(object):

    def __init__(self, t, feat):

        self.t = t
        self.f = feat

    def __call__(self, speech):

        speech = speech[:self.t, 0:self.f]

        return speech

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
