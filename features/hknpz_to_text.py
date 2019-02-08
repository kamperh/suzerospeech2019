#!/usr/bin/env python

"""
Convert a NumPy archive file to separate text files.

Author: Herman Kamper, Ewald van der Westhuizen
Contact: kamperh@gmail.com, ewaldvdw@gmail.com
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
    parser.add_argument("--strip_vads", action='store_true', default=False, help="A switch to enable the stripping of VAD indices from the utterance IDs. Default is False since we decided that we are ignoring the VAD indices anyway.")
    parser.add_argument("npz_fn", type=str, help="NumPy archive")
    parser.add_argument(
        "output_dir", type=str,
        help="directory where text files are written (is created if "
        "it doesn't exist)"
        )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def print_report(feat_dict):
    print "Number of unique symbols (feature vector types):", fd_number_of_unique_symbols(feat_dict)
    print "Is continuous:", fd_iscontinuous(feat_dict)
    print "Feature dimensionality:", fd_dimensionality(feat_dict)
    print "Feature data type:", fd_dtype(feat_dict)


#-----------------------------------------------------------------------#
#                   FEATURE DICTIONARY FUNCTIONS                        #
#-----------------------------------------------------------------------#

def fd_iscontinuous(feat_dict):
    """
    Estimate if the features in the dictionary are continuous or discreet.
    """
    elementvalueset = set([])
    for akey, autt in feat_dict.iteritems():
        elementvalueset.update(autt.flatten().tolist())
        if len(elementvalueset) > 2:
            return True
    return False


def fd_number_of_unique_symbols(feat_dict):
    """
    Returns the number of unique feature types in the dictionary.
    """
    symbolset = set([])
    for akey, autt in feat_dict.iteritems():
        for avec in autt:
            symbolset.add(tuple(avec.tolist()))
    return len(symbolset)

def fd_dimensionality(feat_dict):
    """
    Returns a list of vector dimensionalities found in the dictionary.
    We expect only one value to be returned in the list, or else there
    may be an error in the set of vectors.
    """
    dimset = set([])
    for akey, autt in feat_dict.iteritems():
        for avec in autt:
            dimset.add(avec.shape)
    return list(dimset)


def fd_dtype(feat_dict):
    """
    Returns a list of data types found in the dictionary.
    We expect only one value to be returned in the list, or else there
    may be an error in the set of vectors.
    """
    dtypeset = set([])
    for akey, autt in feat_dict.iteritems():
        for avec in autt:
            dtypeset.add(str(avec.dtype))
    return list(dtypeset)


#-----------------------------------------------------------------------#
#                         FEATURE FUNCTIONS                             #
#-----------------------------------------------------------------------#

def fs_iscontinuous(autt):
    """
    Estimate if the features of an utterances are continuous or discreet.
    """
    if len(set(autt.flatten().tolist())) > 2:
        return True
    else:
        return False

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print("Reading:", args.npz_fn)
    feat_dict = np.load(args.npz_fn)

    print_report(feat_dict)

    if not path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print("Writing to:", args.output_dir)
    for utt_key in tqdm(sorted(feat_dict)):

        if args.strip_vads:
            outfn = ''.join(utt_key.rpartition('_')[0:-2])
        else:
            outfn = utt_key

        fn = path.join(args.output_dir, outfn + ".txt")
        np.savetxt(fn, feat_dict[utt_key])

if __name__ == "__main__":
    main()
