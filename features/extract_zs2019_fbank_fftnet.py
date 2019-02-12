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
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--ref_level_db", type=int, default=20)
    parser.add_argument("--min_level_db", type=int, default=-100)
    parser.add_argument("--num_freq", type=int, default=1025)
    parser.add_argument("--frame_shift_ms", type=int, default=10)
    parser.add_argument("--frame_length_ms", type=int, default=25)
    parser.add_argument("--num_mels", type=int, default=45)
    parser.add_argument("--sample_rate", type=int, default=16000)
    return parser.parse_args()


def _process_wav(wav_fn, args):
    wav = audio.load_wav(wav_fn, sample_rate=args.sample_rate)
    fbank = audio.melspectrogram(wav, preemph=args.preemph, ref_level_db=args.ref_level_db,
                                 min_level_db=args.min_level_db, num_freq=args.num_freq,
                                 frame_shift_ms=args.frame_shift_ms, frame_length_ms=args.frame_length_ms,
                                 sample_rate=args.sample_rate, num_mels=args.num_mels).astype(np.float32)

    key = path.splitext(path.split(wav_fn)[-1])[0]

    # Align audios and mels
    hop_length = int(args.frame_shift_ms / 1000 * args.sample_rate)
    length_diff = len(fbank) * hop_length - len(wav)
    wav = wav.reshape(-1, 1)
    if length_diff > 0:
        wav = np.pad(wav, [[0, length_diff], [0, 0]], 'constant')
    elif length_diff < 0:
        wav = wav[: hop_length * fbank.shape[0]]
    return key, wav, fbank


def preprocess(data_dir, args):
    audio_dict = {}
    feat_dict = {}

    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    futures = []
    wav_list = sorted(glob.glob(path.join(data_dir, "*.wav")))
    for wav_fn in wav_list:
        futures.append(executor.submit(partial(_process_wav, wav_fn, args)))

    for future in tqdm(futures):
        key, wav, fbank = future.result()
        audio_dict[key] = wav
        feat_dict[key] = fbank

    return audio_dict, feat_dict


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

    # Write output
    output_dir = path.join("fbank", args.dataset)
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    audio_output_fn = path.join(output_dir, "unsegmented_" + args.subset + "_audio_fftnet.npz")
    fbank_output_fn = path.join(output_dir, "unsegmented_" + args.subset + "_fbank_fftnet.npz")
    print("Writing:", audio_output_fn, fbank_output_fn)
    np.savez_compressed(audio_output_fn, **audio_dict)
    np.savez_compressed(fbank_output_fn, **feat_dict)

    print(datetime.now())


if __name__ == "__main__":
    main()
