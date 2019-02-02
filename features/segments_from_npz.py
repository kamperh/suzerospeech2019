#!/usr/bin/env python

"""
Cut segments from a .npz file using a "label_utterance_start-end" convention.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2018, 2019
"""

from tqdm import tqdm
import argparse
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("input_npz_fn", type=str, help="")
    parser.add_argument("segments_fn", type=str, help="")
    parser.add_argument("output_npz_fn", type=str, help="")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Read the .npz file
    print("Reading npz:", args.input_npz_fn)
    input_npz = np.load(args.input_npz_fn)

    # Create input npz segments dict
    utterance_segs = {}  # utterance_segs["s08_02b_029657-029952"]
                         # is (29657, 29952)
    for key in input_npz.keys():
        utterance_segs[key] = tuple(
            [int(i) for i in key.split("_")[-1].split("-")]
            )

    # Create target segments dict
    print("Reading segments:", args.segments_fn)
    target_segs = {}  # target_segs["years_s01_01a_004951-005017"]
                      # is ("s01_01a", 4951, 5017)
    for line in open(args.segments_fn):
        line_split = line.split("_")
        utterance = line_split[-3] + "_" + line_split[-2]
        start, end = line_split[-1].split("-")
        start = int(start)
        end = int(end)
        target_segs[line.strip()] = (utterance, start, end)

    print("Extracting segments:")
    output_npz = {}
    n_target_segs = 0
    for target_seg_key in tqdm(target_segs):
        utterance, target_start, target_end = target_segs[target_seg_key]
        for utterance_key in [
                i for i in utterance_segs.keys() if i.startswith(utterance)]:
            utterannce_start, utterance_end = utterance_segs[utterance_key]
            if (target_start >= utterannce_start and target_start <=
                    utterance_end):
                start = target_start - utterannce_start
                end = target_end - utterannce_start
                output_npz[target_seg_key]  = input_npz[utterance_key][start:end]
                n_target_segs += 1
                break
        # if not target_seg_key in output_npz:
        #     print("Missed:", target_seg_key)

    print(
        "Extracted " + str(n_target_segs) + " out of " + str(len(target_segs))
        + " segments"
        )
    print("Writing:", args.output_npz_fn)
    np.savez(args.output_npz_fn, **output_npz)


if __name__ == "__main__":
    main()

