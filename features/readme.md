Input Feature Extraction
========================

Overview
--------
These steps extract filterbank and MFCC features. Features are saved in NumPy
archives where the keys are `<speaker>_<utterance id>_<start>-<end>` and the
values are NumPy arrays of shape `[n_frames, dim]`.


Filterbanks
-----------
Extract filterbanks for the English sets:

    ./extract_zs2019_fbank.py english train
    ./extract_zs2019_fbank.py english test

The default parameters for the filterbanks are set in `features.py`. To extract
features for Buckeye data, the corresponding `buckeye` scripts can be called
(these do not require the `dataset` argument).

The filterbanks can be converted to binary files by running:

    ./npz_to_binary.py fbank/english/train.npz fbank/english/train/
    ./npz_to_binary.py fbank/english/test.npz fbank/english/test/


MFCCs
-----
Extract MFCCs for the English sets:

    ./extract_zs2019_mfcc.py english train
    ./extract_zs2019_mfcc.py english test

The default parameters for the MFCCs are set in `features.py`.


Splitting audio according to VAD
--------------------------------
For the synthesis component, it is useful to have separate wav files. These can
be obtained by running:

    ./split_wav_vad.py english train
    ./split_wav_vad.py english test


Same-different words for Buckeye
--------------------------------
For the same-different evaluation, isolated words need to be extracted. First
the set of words needs to be determined from the forced alignments:

    ./samediff_words.py buckeye

Then extract the segments from the NumPy archives:

    # Filterbanks
    ./segments_from_npz.py \
        fbank/buckeye/devpart2.dd.npz \
        lists/buckeye.samediff.list \
        fbank/buckeye/devpart2.samediff.dd.npz \

    # MFCC
    ./segments_from_npz.py \
        mfcc/buckeye/devpart2.dd.npz \
        lists/buckeye.samediff.list \
        mfcc/buckeye/devpart2.samediff.dd.npz
