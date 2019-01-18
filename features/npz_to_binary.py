#!/usr/bin/env python

"""
Convert a NumPy archive file to separate binary files.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("npz_fn", type=str, help="NumPy archive")
    parser.add_argument(
        "binary_dir", type=str,
        help="directory where binary files are written (is created if "
        "it doesn't exist"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print("Reading:", args.npz_fn)
    feat_dict = np.load(args.npz_fn)

    if not path.isdir(args.binary_dir):
        os.makedirs(args.binary_dir)

    for utt_key in tqdm(sorted(feat_dict)):
        fn = path.join(args.binary_dir, utt_key + ".cmp")
        features = feat_dict[utt_key].astype("float32")
        with open(fn, "wb") as f:
            features.tofile(f)


if __name__ == "__main__":
    main()
