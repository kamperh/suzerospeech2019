"""
Pytorch Encoding Script for STE Binarization Networks

Andre Nortje adnortje@gmail.com

"""
# ----------------------------------------------------------------------------------------------------------------------
# Preliminaries
# ----------------------------------------------------------------------------------------------------------------------

# General imports
import os
import torch
import itertools
import numpy as np
import argparse as arg
from progress.bar import Bar
import torchvision.transforms as tf
from torch.utils.data import DataLoader

# Torch base code imports
from networks import SpeechAuto, ConvSpeechAuto
from process_data import SpeechDataset, Numpy2Tensor, Tensor2Numpy, CropSpeech

# Constant Values
GOF = 10
MFCC = 13
EMBED_DIM = 50
FILTERBANK = 45
NUM_SPEAKERS = 102


# File names (.npz archives)
FEAT_NPZ = "output_feats.npz"
BITS_NPZ_OH = "disc_feats_one_hot.npz"
BITS_NPZ_MH = "disc_feats_multi_hot.npz"


# Helper functions

"""
multi_hot_2_one_hot function

    ref https://www.reddit.com/r/MachineLearning/comments/31fk7i/converting_target_indices_to_onehotvector/
"""


def multi_2_one_hot(multi_hot, lookup_dict, bottle_neck):
    # multi-hot -> integer targets
    int_targets = [
        lookup_dict["".join(map(str, map(int, mh)))] for mh in multi_hot
    ]

    # int targets -> one hot
    one_hot = np.eye(2 ** bottle_neck)[int_targets]

    return one_hot


# ----------------------------------------------------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------------------------------------------------

# create Argument Parser
parser = arg.ArgumentParser(
    prog="Encoding Script:",
    description="Encodes input using specified system and saves two .npz files namely: "
                "1) output_discrete_feats.npz --> one-hot binary symbols "
                "2) output_feats.npz --> feature reconstructions "
)

parser.add_argument(
    '--system_file',
    '-s',
    metavar='SYSTEM_FILE',
    type=str,
    required=True,
    help='SpeechCompression System'
)

parser.add_argument(
    '--input_file',
    '-if',
    metavar='INPUT_FILE',
    type=str,
    help='Training .npz file'
)

parser.add_argument(
    '--speaker_cond',
    '-sc',
    metavar='SPEAKER_CONDITIONAL',
    type=str,
    help='Speaker on which to condition encodings'
)

args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------
# Encoder Script
# ----------------------------------------------------------------------------------------------------------------------

# GPU || CPU
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# System Definition:

# check file location
system_file = os.path.expanduser(args.system_file)

if not os.path.isfile(system_file):
    raise FileNotFoundError("Specified System File D.N.E")

# Define System Variables
system = None
bottleneck_depth = None

# extract system name
system_name = os.path.basename(system_file).split('.')[0]

if system_name == "SpeechAuto":

    # (1 x 128 one-hot -> 7 bits multi-hot)
    bottleneck_depth = 7

    print("System: {}".format(system_name))

    # define network architecture
    system = SpeechAuto(
        name="SpeechAuto",
        bnd=bottleneck_depth,
        input_size=MFCC,
        target_size=FILTERBANK,
        speaker_cond=(
            EMBED_DIM,
            NUM_SPEAKERS
        )
    ).load(save_file=system_file)

elif system_name == "ConvSpeechAuto":

    # bottleneck depth
    bottleneck_depth = 14

    print("System: {}".format(system_name))

    # def network
    sys = ConvSpeechAuto(
        name="ConvSpeechAuto",
        bnd=bottleneck_depth,
        input_size=MFCC,
        target_size=FILTERBANK,
        gof=GOF,
        speaker_cond=(
            EMBED_DIM,
            NUM_SPEAKERS
        )
    ).load(save_file=system_file)

else:
    err_msg = "Oops, we didn't train this : {}"
    raise ValueError(err_msg.format(args.system_file))

# place model on device
system.to(device)

# inference mode
system.train(False)

# Dataset and DataLoader Definition:

input_file = os.path.expanduser(args.input_file)

if not os.path.isfile(input_file):
    raise FileNotFoundError("Specified input file D.N.E")

input_dataset = SpeechDataset(
    speech_npz=input_file,
    transform=tf.Compose([
        Numpy2Tensor(),
        CropSpeech(
            t=-1,
            feat=13
        )
    ])
)

input_dataLoader = DataLoader(
    dataset=input_dataset,
    batch_size=1,
    shuffle=False
)


# Define Speaker to Condition on:

# check speaker embedding exists
if args.speaker_cond not in input_dataset.speakers:
    err_msg = "Speaker {}, D.N.E!"
    raise KeyError(
        err_msg.format(args.speaker_cond)
    )

# get speaker int
cond_speaker_int = torch.tensor(
    [
        input_dataset.speakers[args.speaker_cond]
    ]
).to(device)

# Start Encoding

# display system name and progress bar
print("\nEncoding using : " + system_name)
print("---------------------------------")

# init progress bar
bar = Bar(
    "Encoding Progress :", max=len(input_dataset)
)

# define encoded feature, multi-hot and one-hot dictionaries
feat_dict = {}
bits_dict_mh = {}
bits_dict_oh = {}

# Binary code lookup table
bin_lookup = {}

bin_codes = [
    list(i) for i in itertools.product([0, 1], repeat=bottleneck_depth)
]

for j, bit_code in enumerate(bin_codes, 0):
    # multi-hot bit key
    mh_bit_key = "".join(
        map(str, bit_code)
    )

    # key : symbol
    bin_lookup[mh_bit_key] = j


# Encoding loop

for data in input_dataLoader:

    # Key and input feature
    utt_key = data["utt_key"][0]
    inpt_feat = data["inpt_feat"].to(device)

    # forward & encode (inference mode)
    with torch.no_grad():
        opt_tensor, bits_tensor = system(
            inpt_feat,
            speaker_id=cond_speaker_int
        )

    # to CPU
    bits_tensor = bits_tensor[0].cpu()
    opt_tensor = opt_tensor[0].cpu()

    # Torch.Tensor -> Numpy array
    opt_np = Tensor2Numpy()(
        opt_tensor
    )
    bits_mh = Tensor2Numpy()(
        bits_tensor
    )

    # [-1, 1] -> [0, 1]
    bits_mh[bits_mh == -1] = 0

    # multi-hot to one-hot
    bits_oh = multi_2_one_hot(
        bits_mh,
        bin_lookup,
        bottle_neck=bottleneck_depth
    )

    # Update Feature Dictionaries
    feat_dict[utt_key] = opt_np
    bits_dict_mh[utt_key] = bits_mh
    bits_dict_oh[utt_key] = bits_oh

    # increment progress bar
    bar.next()

# save Dictionaries
np.savez_compressed(FEAT_NPZ, **feat_dict)
np.savez_compressed(BITS_NPZ_OH, **bits_dict_oh)
np.savez_compressed(BITS_NPZ_MH, **bits_dict_mh)

bar.finish()

# END SCRIPT
