#!/usr/bin/env python

"""
Extract filterbank features for the Buckeye dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from datetime import datetime
from os import path
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys

sys.path.append("..")

from paths import buckeye_datadir
from utils import read_vad
import features


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("subset", type=str, choices=["devpart2", "zs"])    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def read_vad_from_fa(fa_fn, frame_indices=True):
    """
    Read voice activity detected (VAD) regions from a forced alignment file.

    The dictionary has utterance labels as keys and as values the speech
    regions as lists of tuples of (start, end) frame, with the end excluded.
    """
    vad_dict = {}
    prev_utterance = ""
    prev_token_label = ""
    prev_end_time = -1
    start_time = -1
    with open(fa_fn, "r") as f:
        for line in f:
            utterance, start_token, end_token, token_label = line.strip(
                ).split()
            start_token = float(start_token)
            end_token = float(end_token)
            utterance = utterance.replace("_", "-")
            utt_key = utterance[0:3] + "_" + utterance[3:]
            if utt_key not in vad_dict:
                vad_dict[utt_key] = []

            if token_label in ["SIL", "SPN"]:
                continue
            if prev_end_time != start_token or prev_utterance != utterance:
                if prev_end_time != -1:
                    utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
                    if frame_indices:
                        # Convert time to frames
                        start = int(round(start_time * 100))
                        end = int(round(prev_end_time * 100)) + 1
                        vad_dict[utt_key].append((start, end))
                    else:
                        vad_dict[utt_key].append(
                            (start_time, prev_end_time)
                            )
                start_time = start_token

            prev_end_time = end_token
            prev_token_label = token_label
            prev_utterance = utterance

        utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
        if frame_indices:
            # Convert time to frames
            start = int(round(start_time * 100))
            end = int(round(prev_end_time * 100)) + 1  # end index excluded
            vad_dict[utt_key].append((start, end))
        else:
            vad_dict[utt_key].append((start_time, prev_end_time))        
    return vad_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print(datetime.now())

    # Speaker set for the indicated subset
    speaker_fn = path.join(
        "..", "data", "buckeye_" + args.subset + "_speakers.list"
        )
    print("Reading:", speaker_fn)
    speakers = set()
    with open(speaker_fn) as f:
        for line in f:
            speakers.add(line.strip())
    print("Speakers:", sorted(speakers))

    # Raw filterbanks
    feat_dict = {}
    print("Extracting features per speaker:")
    for speaker in sorted(speakers):
        speaker_feat_dict = features.extract_fbank_dir(
            path.join(buckeye_datadir, speaker)
            )
        for wav_key in speaker_feat_dict:
            feat_dict[speaker + "_" + wav_key[3:]] = speaker_feat_dict[wav_key]

    # Read voice activity regions
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    print("Reading:", fa_fn)
    vad_dict = read_vad_from_fa(fa_fn)

    # Only keep voice active regions
    print("Extracting VAD regions:")
    feat_dict = features.extract_vad(feat_dict, vad_dict)

    # Perform per speaker mean and variance normalisation
    print("Per speaker mean and variance normalisation:")
    feat_dict = features.speaker_mvn(feat_dict)

    # Write output
    output_dir = path.join("fbank", "buckeye")
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    output_fn = path.join(output_dir, args.subset + ".npz")
    print("Writing:", output_fn)
    np.savez_compressed(output_fn, **feat_dict)


if __name__ == "__main__":
    main()
