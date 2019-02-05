Same-Different Evaluation
=========================

Overview
--------
Same-different is an evaluation task which is slightly different from the
official baseline, but could be a useful way as an intermediate evaluation of
frame-level representations. It is described in:

- M. A. Carlin, S. Thomas, A. Jansen, and H. Hermansky, "Rapid evaluation of
  speech representations for spoken term discovery," in Proc. Interspeech,
  2011.

Here we use it only as a sanity check for the extracted features. At the
moment, this sanity check can only be performed on the ZeroSpeech 2015 data
(because alignments are required to construct the test set).


Preliminaries
-------------
The [speech_dtw](https://github.com/kamperh/speech_dtw/) package is required to
run the code here. To clone it (assuming you are in the directory with this
readme), run:

    mkdir ../../src/  # not necessary using docker
    git clone https://github.com/kamperh/speech_dtw.git ../../src/speech_dtw/

Build the `speech_dtw` tools by running:

    cd ../../src/speech_dtw
    make
    make test
    cd -

Cython is required as well as `python-dev` or `python3-dev`.


Evaluation
----------
This needs to be run on a multi-core machine. Change the `n_cpus` variable in
`run_calcdists.sh` and `run_samediff.sh` to the number of CPUs on the machine.

Evaluate MFCCs:

    # Devpart2
    ./run_calcdists.sh \
        ../features/mfcc/buckeye/devpart2.samediff.dd.npz 
    ./run_samediff.sh  \
        ../features/mfcc/buckeye/devpart2.samediff.dd.npz 

Evaluate filterbanks:

    # Devpart2
    ./run_calcdists.sh \
        ../features/fbank/buckeye/devpart2.samediff.npz 
    ./run_samediff.sh  \
        ../features/fbank/buckeye/devpart2.samediff.npz 

