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

    ./extract_fbank.py english train
    ./extract_fbank.py english test

The default parameters for the filterbanks are set in `features.py`.


MFCCs
-----
Extract MFCCs for the English sets:

    ./extract_mfcc.py english train
    ./extract_mfcc.py english test

The default parameters for the MFCCs are set in `features.py`.

