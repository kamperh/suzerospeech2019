import numpy as np
import sys


def main():

    if len(sys.argv) < 3:
        print("usage: " + sys.argv[0] + " <src_npz> <dest_dir> [with_repetition]")
        sys.exit(1)

    # path to npz file
    npz_fn = sys.argv[1]
    feats = np.load(npz_fn)

    dest_dir = sys.argv[2]

    if len(sys.argv) > 3 and sys.argv[3] == "with_repetition":
        no_repetition = False
    else:
        no_repetition = True

    for feat_key in feats:
        filename = feat_key + ".txt"
        prev_one_index = -1
        sentence = ""
        for unit in feats[feat_key]:
            one_hot_vec = np.zeros((unit.shape[0],), dtype=int)
            one_index, = np.where(unit == 1)
            one_index = one_index[0]
            if prev_one_index == one_index and no_repetition:
                continue
            one_hot_vec[one_index] = 1
            sentence += ' '.join([str(x) for x in one_hot_vec]) + '\n'
            prev_one_index = one_index
        with open(dest_dir + "/" + filename, "w") as my_file:
            my_file.write(sentence)


if __name__ == '__main__':
    main()