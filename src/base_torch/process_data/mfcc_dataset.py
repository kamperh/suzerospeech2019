# imports
import os
import glob
import numpy as np
from torch.utils.data.dataset import Dataset

"""
MfccDataset:

    Facilitates the creation of an iterable dataset from a folder of MFCC's.

        Args:
            mfcc_dir  (string)   : path to directory containing video clip files
            vid_ext   (string)   : video file extension, default is '.mp4'
            transform (callable) : optional transforms to be applied to mfcc data

"""


class MfccDataset(Dataset):

    def __init__(self, mfcc_npz, transform=None):

        mfcc_npz = os.path.expanduser(mfcc_npz)

        if not os.path.isfile(mfcc_npz):
            raise NotADirectoryError
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

    def __getitem__(self, index):

        # extract mfcc
        utt_key = self.keys[index]

        utt_feat = self.npz[
            utt_key
        ]

        if self.transform is not None:
            # apply transform
            utt_feat = self.transform(utt_feat)

        return utt_feat

    def __len__(self):
        return len(self.keys)
