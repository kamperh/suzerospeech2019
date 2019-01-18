#!/usr/bin/env python

"""
Split audio into separate files according to voice activity detection.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from tqdm import tqdm
import argparse
import glob
import os
import subprocess
import sys

sys.path.append("..")

from paths import zerospeech2019_datadir
from utils import read_vad


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


def split_wav_dir(wav_dir, output_dir, vad_dict):
    """Split the wav files in the given source directory."""
    for wav_fn in tqdm(sorted(glob.glob(path.join(wav_dir, "*.wav")))):
        utt_key = path.splitext(path.split(wav_fn)[-1])[0]
        for (start, end) in vad_dict[utt_key]:
            output_wav_fn = path.join(
                output_dir, utt_key + "_{:06d}-{:06d}".format(int(round(start *
                100)), int(round(end * 100)) + 1) + ".wav"
                )
            cmd = "sox {} {} trim {} {:.5f}".format(
                wav_fn, output_wav_fn, start, end - start
                )
            subprocess.run(cmd.split())


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Output directory
    output_dir = path.join("wav", args.dataset, args.subset)
    if not path.isdir(output_dir):
        os.makedirs(output_dir)

    # Read voice activity regions
    vad_fn = path.join(zerospeech2019_datadir, args.dataset, "vads.txt")
    vad_dict = read_vad(vad_fn, frame_indices=False)

    # Split audio
    data_dir = path.join(zerospeech2019_datadir, args.dataset, args.subset)
    print("Writing to:", output_dir)
    if args.subset == "train":
        print("Splitting unit discovery audio:")
        split_wav_dir(path.join(data_dir, "unit"), output_dir, vad_dict)
        print("Splitting target voice audio:")
        split_wav_dir(path.join(data_dir, "voice"), output_dir, vad_dict)
    else:
        print("Splitting test audio:")
        split_wav_dir(data_dir, output_dir, vad_dict)


if __name__ == "__main__":
    main()

