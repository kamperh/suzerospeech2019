# imports
import torch
from torch.nn import ConstantPad1d

"""
Function speech_collate

    pads each speech feature in batch to longest seq in that batch and converts sampled features to torch.Tensor

"""


def speech_collate(batch, pad_val=0.0):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    # split features and keys
    utt_keys = [
        b[0] for b in batch
    ]

    utt_feats = [
        b[1] for b in batch
    ]

    # batch utterance feat
    utt_batch, seq_len = speech_collate(utt_feats)

    # max seq length
    seq_len = [b.size(0) for b in batch]
    max_seq = max(seq_len)

    # pad to max length
    batch = [
        ConstantPad1d((0, int(max_seq - b.size(0))), value=pad_val)(b.transpose(0, 1)) for b in batch
    ]

    # sort seq & get sorted indices
    indices = torch.argsort(
        torch.tensor(seq_len),
        descending=True
    )
    seq_len.sort(reverse=True)

    # sort batch (descending order) for torch.rnn compatibility
    batch = [
        batch[i] for i in indices
    ]

    batch = torch.stack(batch, 0)

    # (B, f, T) -> (B, T, f)
    batch = batch.permute(0, 2, 1)

    # ret tensor batch & corresponding seq lengths
    return utt_keys, utt_batch, seq_len

