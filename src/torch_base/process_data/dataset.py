# imports
import os
import numpy as np
from torch.utils.data.dataset import Dataset

"""
SpeechDataset:

    Facilitates the creation of an iterable dataset from speech MFCC's or Filter Banks.

        Args:
            speech_npz (string)   : numpy archive (.npz) containing MFCC or Filter bank data
            transform  (callable) : optional transforms to be applied to feature data

"""


class SpeechDataset(Dataset):

    def __init__(self,
                 speech_npz,
                 transform=None):

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

        # Speakers Dict
        speakers = list(
            # remove duplicates
            set([
                utt_key[:utt_key.index("_")] for utt_key in self.keys
            ])
        )

        self.speakers = {
            speakers[i]: i for i in range(len(speakers))
        }

    def __getitem__(self, index):

        # extract utterance key
        utt_key = self.keys[index]

        # extract speaker i.d.
        speaker_id = utt_key[:utt_key.index("_")]

        # integer value denoting speaker
        speaker_int = self.speakers[speaker_id]

        # utterance features
        utt_feat = self.npz[utt_key]

        if self.transform:
            # apply transforms
            utt_feat = self.transform(utt_feat)

        # assuming inputs are system targets
        speech_dict = {
            "utt_key": utt_key,
            "inpt_feat": utt_feat,
            "speaker_int": speaker_int
        }

        return speech_dict

    def get_num_speakers(self):
        return len(self.speakers)

    def __len__(self):
        return len(self.keys)


"""
TargetSpeechDataset:

    Iterable dataset for input -> target speech features (MFCC's to Filter Banks or vice versa).

        Args:
            inpt_npz         (string)   : numpy archive (.npz) containing MFCC or Filter bank data (input)
            target_npz       (string)   : numpy archive (.npz) containing MFCC or Filter bank data (target)
            inpt_transform   (callable) : optional transforms applied to input feature data
            target_transform (callable) : optional transforms applied to target feature data

"""


class TargetSpeechDataset(Dataset):

    def __init__(self,
                 inpt_npz,
                 target_npz,
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
                inpt_npz
            )

        target_npz = os.path.expanduser(
            target_npz
        )

        if not os.path.isfile(target_npz):
            err_msg = "Specified file : {} does not exist!"
            raise FileNotFoundError(err_msg.format(target_npz))
        else:
            self.target_npz = np.load(
                target_npz
            )

        self.keys = sorted(
            self.inpt_npz.keys()
        )

        # define speech transforms
        self.inpt_transform = inpt_transform
        self.target_transform = target_transform

        self.speakers = set([])

        # Speakers Dict
        speakers = list(
            # remove duplicates
            set([
                utt_key[:utt_key.index("_")] for utt_key in self.keys
            ])
        )

        self.speakers = {
            speakers[i]: i for i in range(len(speakers))
        }

    def __getitem__(self, index):

        # extract utterance key
        utt_key = self.keys[index]

        # extract speaker i.d.
        speaker_id = utt_key[:utt_key.index("_")]

        # get speaker int
        speaker_int = self.speakers[speaker_id]

        # input utterance features
        inpt_feat = self.inpt_npz[
            utt_key
        ]

        # target utterance features
        target_feat = self.target_npz[
            utt_key
        ]

        if self.inpt_transform:
            # apply input transforms
            inpt_feat = self.inpt_transform(
                inpt_feat
            )

        if self.target_transform:
            # apply target transforms
            target_feat = self.target_transform(
                target_feat
            )

        # speech Dict
        speech_dict = {
            "utt_key": utt_key,
            "inpt_feat": inpt_feat,
            "target_feat": target_feat,
            "speaker_int": speaker_int
        }

        return speech_dict

    def get_num_speakers(self):
        return len(self.speakers)

    def __len__(self):
        return len(self.keys)
