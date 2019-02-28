# imports
import torch
from random import randint

"""
Transforms applied to SpeechData

"""

"""
Class CropSeqSpeech

    crops Speech Feature Tensor to desired (T x F)
"""


class CropSeqSpeech(object):

    def __init__(self, t):

        self.t = t

    def __call__(self, inpt, target):

        if inpt.size(0) > self.t:
            index = randint(0, inpt.size(0) - self.t)
            inpt = inpt[index:index+self.t, :]
            target = target[index:index+self.t, :]

        else:
            inpt = inpt[:self.t, :]
            target = target[:self.t, :]

        return inpt, target

    def __repr__(self):
        return self.__class__.__name__


"""
Class CropSpeech

    crops Speech Feature Tensor to desired (T x F)
"""


class CropSpeech(object):

    def __init__(self, t, feat):

        self.t = t
        self.f = feat

    def __call__(self, speech):

        speech = speech[: self.t, : self.f]

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
