#!/usr/bin/env python

"""
Extract filterbank features for a specified dataset.
"""

import argparse
import os
from os import path
import numpy as np

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler

sys.path.append("..")

from paths import zerospeech2019_datadir
import audio_fftnet as audio
import glob


# -----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
# -----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
    )
    parser.add_argument("--dataset", type=str, choices=["english"])
    parser.add_argument("--subset", type=str, choices=["train", "test"])
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--num_freq", type=int, default=1025)
    parser.add_argument("--frame_shift_ms", type=int, default=10)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--minf0", type=int, default=40)
    parser.add_argument("--maxf0", type=int, default=500)
    parser.add_argument("--mcep_dim", type=int, default=24)
    parser.add_argument("--mcep_alpha", type=float, default=0.41)
    return parser.parse_args()


def _process_wav(wav_fn, args):
    wav = audio.load_wav(wav_fn, sample_rate=args.sample_rate)
    mfcc = audio.extract_mcc(wav, num_freq=args.num_freq, frame_shift_ms=args.frame_shift_ms,
                             sample_rate=args.sample_rate, minf0=args.minf0, maxf0=args.maxf0,
                             mcep_dim=args.mcep_dim, mcep_alpha=args.mcep_alpha).astype(np.float32)

    key = path.splitext(path.split(wav_fn)[-1])[0]

    # Align audios and mels
    hop_length = int(args.frame_shift_ms / 1000 * args.sample_rate)
    length_diff = len(mfcc) * hop_length - len(wav)
    wav = wav.reshape(-1, 1)
    if length_diff > 0:
        wav = np.pad(wav, [[0, length_diff], [0, 0]], 'constant')
    elif length_diff < 0:
        wav = wav[: hop_length * mfcc.shape[0]]
    return key, wav, mfcc


def preprocess(data_dir, args):
    audio_dict = {}
    feat_dict = {}

    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    futures = []
    wav_list = sorted(glob.glob(path.join(data_dir, "*.wav")))
    for wav_fn in wav_list:
        futures.append(executor.submit(partial(_process_wav, wav_fn, args)))

    for future in tqdm(futures):
        key, wav, mfcc = future.result()
        audio_dict[key] = wav
        feat_dict[key] = mfcc

    return audio_dict, feat_dict


def calc_stats(feat_dict):
    scaler = StandardScaler()
    for feat in feat_dict.values():
        scaler.partial_fit(feat)

    mean = scaler.mean_
    scale = scaler.scale_
    mean[0] = 0.0
    scale[0] = 1.0
    return mean, scale


def main():
    args = check_argv()

    print(datetime.now())

    # Raw filterbanks
    data_dir = path.join(zerospeech2019_datadir, args.dataset, args.subset)
    if args.subset == "train":
        print("Extracting unit discovery features:")
        audio_dict, feat_dict = preprocess(path.join(data_dir, "unit"), args)
        print("Extracting target voice features:")
        targed_audio_dict, target_feat_dict = preprocess(path.join(data_dir, "voice"), args)
        audio_dict.update(targed_audio_dict)
        feat_dict.update(target_feat_dict)
    else:
        print("Extracting test features:")
        audio_dict, feat_dict = preprocess(data_dir, args)

    mean, scale = calc_stats(feat_dict)
    print(mean)
    print(scale)

    # Write output
    output_dir = path.join("mfcc", args.dataset)
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    audio_output_fn = path.join(output_dir, "unsegmented_" + args.subset + "_audio_fftnet.npz")
    mfcc_output_fn = path.join(output_dir, "unsegmented_" + args.subset + "_mfcc_fftnet.npz")
    mean_output_fn = path.join(output_dir, "mean.npy")
    scale_output_fn = path.join(output_dir, "scale.npy")
    print("Writing:", audio_output_fn, mfcc_output_fn)
    np.save(mean_output_fn, mean)
    np.save(scale_output_fn, scale)
    np.savez_compressed(audio_output_fn, **audio_dict)
    np.savez_compressed(mfcc_output_fn, **feat_dict)

    print(datetime.now())


if __name__ == "__main__":
    main()
