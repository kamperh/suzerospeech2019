# imports
import os
import numpy as np
from torch.utils.data.dataset import Dataset

"""
SpeechDataset:

    Facilitates the creation of an iterable dataset from speech MFCC's or Filter Banks.

        Args:
            speech_npz (string)   : numpy archive (.npz) containing MFCC or Filter bank data
            ret_keys   (bool)     : set to True to return array of utt_keys
            transform  (callable) : optional transforms to be applied to feature data

"""


class SpeechDataset(Dataset):

    def __init__(self,
                 speech_npz,
                 condition=False,
                 return_keys=False, transform=None):

        speech_npz = os.path.expanduser(speech_npz)

        if not os.path.isfile(speech_npz):
            err_msg = "Specified file : {} does not exist!"
            raise FileNotFoundError(err_msg.format(speech_npz))
        else:
            self.speech_npz = speech_npz

        # load speech feature .npz file
        self.npz = np.load(
            self.speech_npz
        )

        self.keys = sorted(
            self.npz.keys()
        )

        self.transform = transform
        self.return_keys = return_keys

        if condition:
            # Speaker Set
            self.speakers = set([])

            for utt_key in self.keys:

                # extract speaker i.d.
                speaker_id = utt_key[:utt_key.index("_")]

                # add speaker to set
                self.speakers.add(speaker_id)

    def __getitem__(self, index):

        # extract utterance key
        utt_key = self.keys[index]

        # utterance features
        utt_feat = self.npz[utt_key]

        if self.transform:
            # apply transforms
            utt_feat = self.transform(utt_feat)

        if self.return_keys:
            return utt_key, utt_feat

        return utt_feat

    def __len__(self):
        return len(self.keys)


"""
ConversionSpeechDataset:

    Iterable dataset for input -> target speech features (MFCC's to Filter Banks or vice versa).

        Args:
            inpt_npz    (string)   : numpy archive (.npz) containing MFCC or Filter bank data (input)
            target_npz  (string)   : numpy archive (.npz) containing MFCC or Filter bank data (target)
            ret_keys    (bool)     : set to True to return array of utt_keys
            transform   (callable) : optional transforms to be applied to feature data

"""


class ConversionSpeechDataset(Dataset):

    def __init__(self,
                 inpt_npz,
                 target_npz,
                 condition=False, return_keys=False,
                 inpt_transform=None, target_transform=None):

        # expand & check file locations
        inpt_npz = os.path.expanduser(
            inpt_npz
        )

        if not os.path.isfile(inpt_npz):
            err_msg = "Specified file : {} does not exist!"
            raise FileNotFoundError(err_msg.format(inpt_npz))
        else:
            self.inpt_npz = np.load(
                self.inpt_npz
            )

        target_npz = os.path.expanduser(
            target_npz
        )

        if not os.path.isfile(cond_npz):
            err_msg = "Specified file : {} does not exist!"
            raise FileNotFoundError(err_msg.format(target_npz))
        else:
            self.target_npz = np.load(
                target_npz
            )

        self.keys = sorted(
            self.inpt_npz.keys()
        )

        self.return_keys = return_keys
        self.inpt_transform = inpt_transform
        self.target_transform = target_transform

        # Dict of Speakers and their utterances
        if condition:
            # Speaker Set
            self.speakers = set([])

            for utt_key in self.keys:

                # extract speaker i.d.
                speaker_id = utt_key[:utt_key.index("_")]

                # add speaker to set
                self.speakers.add(speaker_id)

    def __getitem__(self, index):

        # extract utterance key
        utt_key = self.keys[index]

        # input utterance features
        inpt_utt = self.inpt_npz[
            utt_key
        ]

        # target utterance features
        target_utt = self.target_npz[
            utt_key
        ]

        if self.transform:
            # apply transforms
            inpt_utt = self.inpt_transform(
                inpt_utt
            )

            target_utt = self.target_transform(
                target_utt
            )

        if self.return_keys:
            return utt_key, inpt_utt, target_utt

        return inpt_utt, target_utt

    def __len__(self):
        return len(self.keys)
