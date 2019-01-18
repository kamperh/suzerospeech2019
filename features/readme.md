Input Feature Extraction
========================

Overview
--------
These steps extract filterbank and MFCC features. Features are saved in NumPy
archives where the keys are `<speaker>_<utterance id>_<start>-<end>` and the
values are NumPy arrays of shape `[n_frames, dim]`.


Filterbanks
-----------


