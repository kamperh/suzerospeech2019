#!/usr/bin/env python

"""
Extract ground truth types of at least 50 frames and 5 characters.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2018, 2019
"""

from os import path
import argparse
import os
import sys

forced_alignment_dir = path.join("..", "data")
output_dir = "lists"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("dataset", type=str, choices=["buckeye"])
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    if args.dataset == "buckeye":
        fa_fn = path.join(forced_alignment_dir, "buckeye_english.wrd")
    # elif args.dataset == "xitsonga":
    #     fa_fn = path.join(forced_alignment_dir, "xitsonga.wrd")

    # Extract all word tokens from the corpus
    print("Reading:", fa_fn)
    words = []
    with open(fa_fn, "r") as f:
        for line in f:
            utterance, start, end, label = line.strip().split()
            utterance = utterance.replace("_", "-")
            start = float(start)
            end = float(end)
            if label in ["SIL", "SPN"]:
                continue
            words.append((utterance, label, (start, end)))

    print("Finding tokens")
    words_50fr5ch = []
    for utterance, label, (start, end) in words:
        start_frame = int(round(float(start) * 100))
        end_frame = int(round(float(end) * 100))
        if end_frame - start_frame >= 50 and len(label) >= 5:
            words_50fr5ch.append((utterance, label, (start_frame, end_frame)))
    print("No. tokens:", len(words_50fr5ch), "out of", len(words))

    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    output_fn = path.join(output_dir, args.dataset + ".samediff.list")
    print("Writing:", output_fn)
    with open(output_fn, "w") as f:
        for utterance, label, (start, end) in words_50fr5ch:
            utterance = utterance[:3] + "_" + utterance[3:]
            f.write(
                label + "_" + utterance + "_%06d-%06d\n" % (int(start),
                int(end) + 1)
                )


if __name__ == "__main__":
    main()
