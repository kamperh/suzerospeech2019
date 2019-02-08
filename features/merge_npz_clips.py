"""
Glues all files in a numpy archive with the same speaker and utterance id 
together. (Utterances that were split by silences are merged together.)  

Usage: python merge_npz_clips.py <src_npz> <dest_npz>
Where <src_npz> is the numpy archive with the split utterances and <dest_npz>
is the name of the new numpy archive for the merged utterances.

Author: Lisa van Staden
"""

import sys
import numpy as np


def main():

    if len(sys.argv) != 3:
        print("usage: " + sys.argv[0] + " <src_npz> <dest_npz>")

    # path to numpy archive with split utterances
    npz_fn = sys.argv[1]
    feats = np.load(npz_fn)

    output_npz = npz_fn
    if len(sys.argv) > 2:
        # get file name for output npz from arguments
        output_npz = sys.argv[2]

    output_feats = {}  # merged features

    old_key = ""
    for feat_key in feats:
        [speaker, utt_id, frames] = feat_key.split("_")

        new_key = speaker + "_" + utt_id

        if new_key != old_key:
            output_feats[new_key] = feats[feat_key]
            old_key = new_key
        else:
            temp_feats = output_feats[new_key]
            output_feats[new_key] = np.concatenate((temp_feats, feats[feat_key]), axis=0)

    np.savez(output_npz, **output_feats)


if __name__ == '__main__':
    main()





