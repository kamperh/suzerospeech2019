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
    utt_keys = []
    inpt_batch = []
    target_batch = []
    speaker_ints = []

    for b in batch:
        # append values
        utt_keys.append(b["utt_key"])
        inpt_batch.append(b["inpt_feat"])
        if "target_feat" in b:
            target_batch.append(b["target_feat"])
        speaker_ints.append(b["speaker_int"])

    # max seq length
    seq_len = [
        b.size(0) for b in inpt_batch
    ]
    max_seq = max(seq_len)

    # pad to max length
    inpt_batch = [
        ConstantPad1d(
            (0, int(max_seq - b.size(0))),
            value=pad_val
        )(b.transpose(0, 1)) for b in inpt_batch
    ]

    # sort seq & get sorted indices
    indices = torch.argsort(
        torch.tensor(seq_len),
        descending=True
    )
    seq_len.sort(reverse=True)

    # sort batch (descending order) for torch.rnn compatibility
    inpt_batch = [
        inpt_batch[i] for i in indices
    ]

    inpt_batch = torch.stack(
        inpt_batch,
        dim=0
    )

    # (B, f, T) -> (B, T, f)
    inpt_batch = inpt_batch.permute(0, 2, 1)

    # rearrange speaker ints and utt_keys to match batches
    speaker_ints = torch.tensor([
        speaker_ints[i] for i in indices
    ])

    utt_keys = [
        utt_keys[i] for i in indices
    ]

    # Batch Dict
    batch_dict = {
        "utt_keys": utt_keys,
        "input_batch": inpt_batch,
        "speaker_ints": speaker_ints
    }

    if "target_feat" in batch[0]:
        target_batch = [
            ConstantPad1d(
                (0, int(max_seq - b.size(0))),
                value=pad_val
            )(b.transpose(0, 1)) for b in target_batch
        ]

        target_batch = [
            target_batch[i] for i in indices
        ]

        target_batch = torch.stack(
            target_batch,
            dim=0
        )

        # (B, f, T) -> (B, T, f)
        batch_dict["target_batch"] = target_batch.permute(0, 2, 1)

    return batch_dict
