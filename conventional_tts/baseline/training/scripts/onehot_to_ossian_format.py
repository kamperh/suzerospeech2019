import sys
from os import path
import glob
import numpy as np
import read_zrsc2019

# takes output from BEER with and split it into separate onehot format for submission : 
# in:
# 0107_400123_0000.txt
#     0 0 0 0 0 0 1 0 0
#     0 0 0 0 0 1 0 0 0
#     0 0 0 0 0 1 0 0 0
#     0 1 0 0 0 0 0 0 0
#     0 0 0 1 0 0 0 0 0
# out: ossian format, phonemes of two characters only 
# 0107_400123_0000.txt
#     3312125645
# if phoneme of lenght one, append a letter after

def main(onehot_dir, dest_dir, text_corpora_path=None):
    text_corpora = []

    for filepath in glob.iglob(path.join(onehot_dir, '*.txt')):

        print '.',
        sentence = []
        text_corpora_line = []
        previous_unit = -1

        for vector in read_zrsc2019.read(filepath):
            # The variable 'vector' is a tuple containing floats, however since
            # these are onehot, the floats are only a one or a zero.
            #unit = str(np.nonzero(vector)[0].item())
            unit = map(int, vector).index(1)

#            assert int(unit) < 100, \
#              "This tool works only for list of phonemes of size <=100"
#            if len(unit) == 1:
#                unit += 'x'  

            if unit != previous_unit:
                sentence.append(unit)
                previous_unit = unit
                text_corpora_line.append(list(sentence))

        text_corpora.append(text_corpora_line)

        with open(path.join(dest_dir, path.basename(filepath)), 'w') as myfile:
            myfile.write(' '.join(map(str, sentence)) + '\n')

    if text_corpora_path != None :
        with open(text_corpora_path, 'w') as myfile:
            for aline in text_corpora:
                for aitem in aline:
                    myfile.write(' '.join(map(str, aitem)) + '  ')
                myfile.write('\n')

if __name__ == "__main__":

   onehot_dir = sys.argv[1]
   dest_dir = sys.argv[2]
   text_corpora_path = None

   if len(sys.argv) == 4:
       text_corpora_path = sys.argv[3]

   main(onehot_dir, dest_dir, text_corpora_path=text_corpora_path)

