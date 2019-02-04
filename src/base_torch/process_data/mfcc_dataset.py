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

    def __init__(self, mfcc_dir, transform=None):

        mfcc_dir = os.path.expanduser(mfcc_dir)

        if not os.path.isdir(mfcc_dir):
            raise NotADirectoryError
        else:
            self.mfcc_dir = mfcc_dir

        # load mfcc numpy file
        npz_fn = glob.glob(
            mfcc_dir + '/*.npz'
        )

        self.npz = np.load(npz_fn)

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

        return utt_key, utt_feat

    def __len__(self):
        return len(self.utt_keys)
