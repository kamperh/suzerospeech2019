"""Function for reading and processing ZeroSpeech data.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: March 2019
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
import logging


import numpy as np
import matplotlib.pyplot as plt


from flags import FLAGS


print_log = logging.info


# ------------------------------------------------------------------------------ # -----80~100---- #
# ...
# ------------------------------------------------------------------------------ # -----80~100---- #


def load_speech_features(
        speech_dir,
        input_feats="fbank",  # one of "mfcc" or "fbank"
        target_feats="fbank",  # one of "mfcc" or "fbank"
        language="english",  # one of "english" or "surprise"
        source="fftnet_segments",  # dir containing 'train.npz' and 'test.npz'
        load_parallel=False,  # load parallel speech features
        target_speaker="V001"  # one of "V001" or "V002"
    ):
    input_dir = os.path.join(speech_dir, input_feats, language, source)
    target_dir = os.path.join(speech_dir, target_feats, language, source)
    # load speech features
    input_train = read_npz_features(os.path.join(input_dir, "train.npz"))
    input_test = read_npz_features(os.path.join(input_dir, "test.npz"))
    if input_feats == target_feats:  # reuse input features as target
        target_train = input_train
        target_test = input_test
    else:  # using different features as target
        target_train = read_npz_features(os.path.join(target_dir, "train.npz"))
        target_test = read_npz_features(os.path.join(target_dir, "test.npz"))
    # gather tuples of feature data
    input_sets = (input_train, input_test)
    target_sets = (target_train, target_test)
    # optionally load parallel speech data
    if load_parallel:
        input_parallel = read_npz_features(
            os.path.join(input_dir, "parallel.npz"))
        target_parallel = read_npz_features(
            os.path.join(target_dir, "parallel.npz"))
        input_parallel, target_parallel = build_parallel_pair_data(
            input_parallel, target_parallel, target_speaker=target_speaker)
        input_sets += (input_parallel, )
        target_sets += (target_parallel, )
    # get speaker id lookup tables
    speakers_lists = [input_set["speakers"] for input_set in input_sets]
    speakers_lists += [target_set["speakers"] for target_set in target_sets]
    speaker_to_id, id_to_speaker = get_speaker_lookup(*speakers_lists)
    lookup_tables = (speaker_to_id, id_to_speaker)
    # get speaker ids for input and target sets
    get_speaker_ids = np.vectorize(
        lambda speaker: speaker_to_id[speaker], otypes=[FLAGS.np_int_dtype])
    for input_set in input_sets:
        input_set["speaker_ids"] = get_speaker_ids(input_set["speakers"])
    for target_set in target_sets:
        target_set["speaker_ids"] = get_speaker_ids(target_set["speakers"])
    # return feature sets and lookup tables
    return input_sets, target_sets, lookup_tables


def read_npz_features(npz_file):
    keys = []
    x = []
    x_lengths = []
    speakers = []
    num_data = 0
    if not FLAGS.quiet:
        print_log("Reading npz from file: {}".format(npz_file))
    npz = np.load(npz_file)
    for utterance_key in sorted(npz.keys()):    
        # process next utterance
        data = npz[utterance_key]
        length = data.shape[0]
        speaker = utterance_key.split('_')[0]
        # store utterance data
        x.append(data)
        x_lengths.append(length)
        keys.append(utterance_key)
        speakers.append(speaker)
        num_data += 1
    if not FLAGS.quiet:
        print_log("Example key: {}".format(npz.files[0]))
        print_log("No. of utterances in npz: {}".format(num_data))
    return {"keys": np.asarray(keys),
            "x": np.asarray(x),
            "x_lengths": np.asarray(x_lengths),
            "speakers": np.asarray(speakers)}


def get_speaker_lookup(*speakers):
    """Convert speakers to integer id lookup table and vice versa."""
    all_speakers = set()
    for speaker_list in speakers:
        all_speakers = all_speakers | set(list(speaker_list))  # union speaker sets
    speaker_to_id = {}  # speaker->id lookup
    id_to_speaker = {}  # id->speaker lookup
    for index, speaker in enumerate(sorted(list(all_speakers))):
        speaker_to_id[speaker] = index
        id_to_speaker[index] = speaker
    if not FLAGS.quiet:
        print_log(all_speakers)
    return speaker_to_id, id_to_speaker


def build_parallel_pair_data(
        input_parallel, target_parallel, target_speaker="V001"):
    """Param target_speaker one of "V001" or "V002".
    """
    x_input, x_lengths_input, speakers_input, keys_input = [], [], [], []
    x_target, x_lengths_target, speakers_target, keys_target = [], [], [], []  
    for i, speaker in enumerate(target_parallel["speakers"]):
        target_key = target_parallel["keys"][i].split('_')[1]
        if speaker == target_speaker:
            for j, utt_key in enumerate(input_parallel["keys"]):
                input_key = utt_key.split('_')[1]
                if input_key == target_key and i != j:
                    # input speaker example
                    x_input.append(input_parallel["x"][j])
                    x_lengths_input.append(input_parallel["x_lengths"][j])
                    speakers_input.append(input_parallel["speakers"][j])
                    keys_input.append(utt_key)  # := input_parallel["keys"][j]
                    # parallel target speaker example
                    x_target.append(target_parallel["x"][i])
                    x_lengths_target.append(target_parallel["x_lengths"][i])
                    speakers_target.append(speaker)  # := target_parallel["speakers"][i] == target_speaker
                    keys_target.append(target_parallel["keys"][i])
    input_parallel = {
        "x": np.asarray(x_input),
        "x_lengths": np.asarray(x_lengths_input),
        "speakers": np.asarray(speakers_input),
        "keys": np.asarray(keys_input)}
    target_parallel = {
        "x": np.asarray(x_target),
        "x_lengths": np.asarray(x_lengths_target),
        "speakers": np.asarray(speakers_target),
        "keys": np.asarray(keys_target)}
    return input_parallel, target_parallel


def build_train_validation_data(speech_dict, validation_ratio=0.2):
    val_len = int(len(speech_dict["x"]) * validation_ratio)
    val_split = {
        "x": speech_dict["x"][:val_len],
        "x_lengths": speech_dict["x_lengths"][:val_len],
        "speakers": speech_dict["speakers"][:val_len],
        "keys": speech_dict["keys"][:val_len]}
    train_split = {
        "x": speech_dict["x"][val_len:],
        "x_lengths": speech_dict["x_lengths"][val_len:],
        "speakers": speech_dict["speakers"][val_len:],
        "keys": speech_dict["keys"][val_len:]}
    return val_split, train_split


def limit_dimensionality(speech_dict, d_limit=13, modify_inplace=True):
    x_limited = []
    for i, seq in enumerate(speech_dict["x"]):
        x_limited.append(seq[:, :d_limit])  # limit each segment to d_limit features
    if not modify_inplace:
        speech_dict = speech_dict.copy()
    speech_dict["x"] = np.asarray(x_limited)
    return speech_dict


def truncate_segments(speech_dict, max_length, modify_inplace=True):
    x_truncated = []
    x_lengths_truncated = []
    for i, seq in enumerate(speech_dict["x"]):
        x_truncated.append(seq[:max_length, :])
        x_lengths_truncated.append(min(speech_dict["x_lengths"][i], max_length))
    if not modify_inplace:
        speech_dict = speech_dict.copy()
    speech_dict["x"] = np.asarray(x_truncated)
    speech_dict["x_lengths"] = np.asarray(x_lengths_truncated)
    return speech_dict


def sequential_split_segments(speech_dict, split_length, include_smaller=True,
                              modify_inplace=True):
    x_split = []
    x_lengths_split = []
    keys_split = []
    speakers_split = []
    for i, seq in enumerate(speech_dict["x"]):
        for j in range(int(np.ceil(speech_dict["x_lengths"][i]/split_length))):  # include final smaller split
            seg_length = min(speech_dict["x_lengths"][i] - j*split_length, split_length)
            if include_smaller or seg_length >= split_length:  # if full split length segment or using final small split
                x_split.append(seq[j*split_length:j*split_length+seg_length, :])
                x_lengths_split.append(seg_length)
                keys_split.append("{}_split_{}".format(speech_dict["keys"][i], j))
                speakers_split.append(speech_dict["speakers"][i])
    if not modify_inplace:
        speech_dict = speech_dict.copy()
    speech_dict["x"] = np.asarray(x_split)
    speech_dict["x_lengths"] = np.asarray(x_lengths_split)
    speech_dict["keys"] = np.asarray(keys_split)
    speech_dict["speakers"] = np.asarray(speakers_split)
    return speech_dict


def display_stats(speech_dict, hist_bins=100, figsize=(8, 4)):
    x = speech_dict["x"]
    x_lengths = speech_dict["x_lengths"]
    # compute segment feature stats
    x_mean = 0.
    x_min = np.inf
    x_max = - np.inf
    for seg in x:  # process each individually due to variable lengths
        x_mean += np.mean(seg)
        seg_max = np.max(seg)
        seg_min = np.min(seg)
        x_max = seg_max if seg_max > x_max else x_max
        x_min = seg_min if seg_min < x_min else x_min
    x_mean = x_mean/len(x)
    # compute segment lengths stats
    x_lengths_two_std = np.mean(x_lengths) + 2*np.std(x_lengths)
    num_outliers = 0
    for x_length in x_lengths:
        num_outliers += 1 if x_length > x_lengths_two_std else 0
    # display data stats
    print_log(
        "Feature data mean: {:.3f}".format(x_mean))
    print_log(
        "Feature data max value: {:.3f}".format(x_max))
    print_log(
        "Feature data min value: {:.3f}".format(x_min))
    print_log(
        "Longest segment: {:d} frames".format(np.max(x_lengths)))
    print_log(
        "Mean plus 2 std deviations (~95%): {:d} frames"
        "".format(int(round(x_lengths_two_std))))
    print_log(
        "Outliers above mean plus 2 std deviations: {:d} ({:.3f} %)"
        "".format(num_outliers, num_outliers/len(x_lengths)*100))
    # bin plot of segment lengths
    plt.figure(figsize=figsize)
    plt.hist(x_lengths, bins=hist_bins)
