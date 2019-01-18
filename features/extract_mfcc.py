#!/usr/bin/env python

"""
Extract filterbank features for a specified dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from datetime import datetime
from os import path
import argparse
import numpy as np
import os
import sys

sys.path.append("..")

from paths import zerospeech2019_datadir
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
    parser.add_argument("dataset", type=str, choices=["english"])
    parser.add_argument("subset", type=str, choices=["train", "test"])    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print(datetime.now())

    # Raw filterbanks
    data_dir = path.join(zerospeech2019_datadir, args.dataset, args.subset)
    if args.subset == "train":
        print("Extracting unit discovery features:")
        feat_dict = features.extract_mfcc_dir(path.join(data_dir, "unit"))
        print("Extracting target voice features:")
        feat_dict.update(
            features.extract_mfcc_dir(path.join(data_dir, "voice"))
            )
    else:
        print("Extracting test features:")
        feat_dict = features.extract_mfcc_dir(data_dir)

    # Read voice activity regions
    vad_fn = path.join(zerospeech2019_datadir, args.dataset, "vads.txt")
    vad_dict = read_vad(vad_fn)

    # Only keep voice active regions
    print("Extracting VAD regions:")
    feat_dict = features.extract_vad(feat_dict, vad_dict)

    # Perform per speaker mean and variance normalisation
    print("Per speaker mean and variance normalisation:")
    feat_dict = features.speaker_mvn(feat_dict)

    # Write output
    output_dir = path.join("mfcc", args.dataset)
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    output_fn = path.join(output_dir, args.subset + ".dd.npz")
    print("Writing:", output_fn)
    np.savez_compressed(output_fn, **feat_dict)

    print(datetime.now())


if __name__ == "__main__":
    main()
