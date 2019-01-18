"""
Functions for extracting filterbank and MFCC features.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import mfcc
from tqdm import tqdm
import glob
import numpy as np
import scipy.io.wavfile as wav


def get_fbank_for_dir(dir):
    """
    Extract filterbanks for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension.
    """
    feat_dict = {}
    for wav_fn in tqdm(sorted(glob.glob(path.join(dir, "*.wav")))):
        samplerate, signal = wav.read(wav_fn)
        fbanks = logfbank(
            signal, samplerate=samplerate, winlen=0.025, winstep=0.01,
            nfilt=45, nfft=2048, lowfreq=0, highfreq=None, preemph=0
            )
        feat_dict[path.splitext(path.split(wav_fn)[-1])[0]] = fbanks
    return feat_dict


def get_mfcc_for_dir(dir):
    """
    Extract MFCCs for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension. Deltas and double deltas are also extracted.
    """
    pass


def extract_vad(feat_dict, vad_dict):
    """
    Remove silence based on voice activity detection (VAD).

    The `vad_dict` should have the same keys as `feat_dict` with the active
    speech regions given as lists of tuples of (start, end) frame, with the end
    excluded.
    """
    output_dict = {}
    for utt_key in tqdm(sorted(feat_dict)):
        for (start, end) in vad_dict[utt_key]:
            segment_key = utt_key + "_{:06d}-{:06d}".format(start, end)
            output_dict[segment_key] = feat_dict[utt_key][start:end, :]
    return output_dict


def speaker_mvn(feat_dict):
    """
    Perform per-speaker mean and variance normalisation.

    It is assumed that each of the keys in `feat_dict` starts with a speaker
    identifier followed by an underscore.
    """

    speakers = set([key.split("_")[0] for key in feat_dict])

    # Separate features per speaker
    speaker_features = {}
    for utt_key in sorted(feat_dict):
        speaker = utt_key.split("_")[0]
        if speaker not in speaker_features:
            speaker_features[speaker] = []
        speaker_features[speaker].append(feat_dict[utt_key])

    # Determine means and variances per speaker
    speaker_mean = {}
    speaker_std = {}
    for speaker in speakers:
        features = np.vstack(speaker_features[speaker])
        speaker_mean[speaker] = np.mean(features, axis=0)
        speaker_std[speaker] = np.std(features, axis=0)

    # Normalise per speaker
    output_dict = {}
    for utt_key in tqdm(sorted(feat_dict)):
        speaker = utt_key.split("_")[0]
        output_dict[utt_key] = (
            (feat_dict[utt_key] - speaker_mean[speaker]) / 
            speaker_std[speaker]
            )

    return output_dict

