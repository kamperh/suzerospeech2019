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
import scipy.io.wavfile as wav


def fbank_in_dir(dir):
    """
    Extract filterbanks for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension.
    """
    feat_dict = {}
    for wav_fn in tqdm(sorted(glob.glob(path.join(dir, "*.wav")))[:10]):
        samplerate, signal = wav.read(wav_fn)
        fbanks = logfbank(
            signal, samplerate=samplerate, winlen=0.025, winstep=0.01,
            nfilt=45, nfft=2048, lowfreq=0, highfreq=None, preemph=0
            )
        feat_dict[path.splitext(path.split(wav_fn)[-1])[0]] = fbanks
    return feat_dict


def mfcc_in_dir(dir):
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


def speaker_mvn(feat_dict):
    """
    Perform per-speaker mean and variance normalisation.

    It is assumed that each of the keys in `feat_dict` starts with a speaker
    identifier followed by an underscore.
    """
    pass


def main():

    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plt
    fn = "/home/kamperh/endgame/datasets/zerospeech2019/shared/databases/english/train/unit/S015_0361841101.wav"
    rate, sig = wav.read(fn)

    # import soundfile as sf
    # sig, rate = sf.read(fn) 


    mfcc_feat = mfcc(sig,rate)
    print(mfcc_feat.shape)
    # plt.imshow(mfcc_feat[:100,:])
    # plt.axis("off")

    fbank_feat = logfbank(
        sig, samplerate=rate, winlen=0.025, winstep=0.01, nfilt=45, nfft=2048,
        lowfreq=0, highfreq=None, preemph=0
        )

    plt.imshow(fbank_feat[:100,:])
    plt.axis("off")

    import numpy as np
    print(fbank_feat.shape)
    print(np.min(fbank_feat), np.max(fbank_feat))

    print(fbank_feat[:10])

    plt.show()


    # plt.savefig("test.pdf")


if __name__ == "__main__":
    main()

