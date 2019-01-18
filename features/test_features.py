"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from features import *
import numpy as np
import numpy.testing as npt


def test_speaker_mvn():

    np.random.seed(1)

    # Generate a random features dictionary
    feat_dict = {}
    n_speakers = 5
    d_feats = 10
    max_length = 12
    n_samples = 100
    i_speaker = 0
    for n in range(n_samples):
        i_speaker = 0 if i_speaker == n_speakers - 1 else i_speaker + 1
        utt_key = str(i_speaker) + "_" + str(n)
        length = np.random.randint(max_length)
        feat_dict[utt_key] = np.random.randn(length, d_feats)
    
    # Normalise
    feat_dict = speaker_mvn(feat_dict)

    # Test each speaker
    for i_speaker in range(n_speakers):
        features = []
        for utt_key in feat_dict:
            if utt_key.startswith(str(i_speaker) + "_"):
                features.append(feat_dict[utt_key])
        features = np.vstack(features)
        actual_mean = np.mean(features, axis=0)
        actual_std = np.std(features, axis=0)
        npt.assert_almost_equal(actual_mean, np.zeros(d_feats), decimal=5)
        npt.assert_almost_equal(actual_std, np.ones(d_feats), decimal=5)


test_speaker_mvn()
