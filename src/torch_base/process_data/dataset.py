# imports
import os
import numpy as np
from torch.utils.data.dataset import Dataset

"""
MfccDataset:

    Facilitates the creation of an iterable dataset from a folder of MFCC's.

        Args:
            mfcc_npz  (string)   : numpy archive (.npz) containing mfcc / filter bank data
            ret_keys  (bool)     : set to True to return array of utt_keys
            transform (callable) : optional transforms to be applied to mfcc data

"""


class MfccDataset(Dataset):

    def __init__(self, mfcc_npz, ret_keys=False, transform=None):

        mfcc_npz = os.path.expanduser(mfcc_npz)

        if not os.path.isfile(mfcc_npz):
            err_msg = "Specified file : {} does not exist!"
            raise NotADirectoryError(err_msg.format(mfcc_npz))
        else:
            self.mfcc_npz = mfcc_npz

        # load mfcc numpy file
        self.npz = np.load(
            self.mfcc_npz
        )

        self.keys = sorted(
            self.npz.keys()
        )

        self.transform = transform

        self.ret_keys = ret_keys

    def __getitem__(self, index):

        # extract mfcc
        utt_key = self.keys[index]

        utt_feat = self.npz[
            utt_key
        ]

        if self.transform is not None:
            # apply transform
            utt_feat = self.transform(utt_feat)

        if self.ret_keys:
            return utt_key, utt_feat

        return utt_feat

    def __len__(self):
        return len(self.keys)
