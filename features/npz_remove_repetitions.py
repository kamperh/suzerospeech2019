
"""
Convert a numpy archive with one hot encodings to text files in the zs2019 one hot format.

Usage: python npz_remove_repetitions.py <src_npz> <dest_npz>
Where <src_npz> is the numpy archive file and <dest_npz> is the directory where the
zs2019 text files should be saved. Optionally, you can pass in "with_repetition" if you
don't want the unit repetitions to be removed.

Author: Lisa van Staden, Ewald van der Westhuizen
"""

import numpy as np
import sys

import pprint


def main():

    if len(sys.argv) < 3:
        print("usage: " + sys.argv[0] + " <src_npz> <dest_npz> [with_repetition]")
        sys.exit(1)

    # path to npz file
    npz_fn = sys.argv[1]
    feats = np.load(npz_fn)
    # output features dictionary
    outfeats = {}

    dest_npz = sys.argv[2]

    for feat_key in feats:
        outfeats[feat_key] = np.array([feats[feat_key][0]])
        filename = feat_key + ".txt"
        prev_one_index = -1
        for unit in feats[feat_key]:
            one_hot_vec = np.zeros((unit.shape[0],), dtype=int)
            one_index, = np.where(unit == 1)
            one_index = one_index[0]
            if prev_one_index == one_index:
                continue
            if prev_one_index != -1:
                outfeats[feat_key] = np.append(outfeats[feat_key], [unit], axis=0)
            one_hot_vec[one_index] = 1
            prev_one_index = one_index

    np.savez_compressed(dest_npz, **outfeats)


if __name__ == '__main__':
    main()

