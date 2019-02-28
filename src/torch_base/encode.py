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
import numpy as np
import argparse as arg
from progress.bar import Bar
from bitstring import BitArray
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# Torch base code imports
from process_data import SpeechDataset, Numpy2Tensor, Tensor2Numpy, CropSpeech
from networks import LinearRnnSpeechAuto, ConvSpeechAuto, ConvRnnSpeechAuto, ConvRnnSpeechAutoBND

# Constant Values
GOF = 8
MFCC = 39
V001 = 100
V002 = 101
EMBED_DIM = 100
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


def multi_2_one_hot(multi_hot, bottle_neck):
    # multi-hot -> integer targets
    int_targets = [
        BitArray(mh).uint for mh in multi_hot
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
    "--system_file",
    "-s",
    metavar="SYSTEM_FILE",
    type=str,
    required=True,
    help="Speech compression system"
)

parser.add_argument(
    "--input_file",
    "-if",
    metavar="INPUT_FILE",
    type=str,
    help="File to be encoded"
)


parser.add_argument(
    "--bottleneck_depth",
    "-bnd",
    metavar="BOTTLENECK_DEPTH",
    type=int,
    help="System bottleneck depth"
)

parser.add_argument(
    "--speaker_cond",
    "-sc",
    metavar="SPEAKER_CONDITIONAL",
    choices=["V001", "V002"],
    type=str,
    help="Speaker on which to condition encodings"
)

parser.add_argument(
    "--one_hot",
    "-oh",
    action="store_true"
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

if system_name == "LinearRnnSpeechAuto":

    print("System: {}".format(system_name))

    # define network architecture
    system = LinearRnnSpeechAuto(
        name="SpeechAuto",
        bnd=args.bottleneck_depth,
        input_size=MFCC,
        target_size=FILTERBANK,
        speaker_cond=(
            EMBED_DIM,
            NUM_SPEAKERS
        )
    ).load(save_file=system_file)

elif system_name == "ConvSpeechAuto":

    print("System: {}".format(system_name))

    # def network
    system = ConvSpeechAuto(
        name="ConvSpeechAuto",
        bnd=args.bottleneck_depth,
        input_size=MFCC,
        target_size=FILTERBANK,
        speaker_cond=(
            EMBED_DIM,
            NUM_SPEAKERS
        )
    ).load(save_file=system_file)

elif system_name == "ConvRnnSpeechAuto":

    print("System: {}".format(system_name))

    # def network
    system = ConvRnnSpeechAuto(
        name="ConvSpeechAuto",
        bnd=args.bottleneck_depth,
        input_size=MFCC,
        target_size=FILTERBANK,
        gof=GOF,
        speaker_cond=(
            EMBED_DIM,
            NUM_SPEAKERS
        )
    ).load(save_file=system_file)

elif system_name == "ConvRnnSpeechAutoBND":

    print("System: {}".format(system_name))

    # def network
    system = ConvRnnSpeechAutoBND(
        name="ConvSpeechAutoBND",
        bnd=args.bottleneck_depth,
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
    transform=Compose([
        Numpy2Tensor(),
        CropSpeech(
            t=None,
            feat=MFCC
        )
    ])
)

input_dataLoader = DataLoader(
    dataset=input_dataset,
    batch_size=1,
    shuffle=False
)


# Define Speaker to Condition on:

# get speaker int
cond_speaker_int = torch.tensor([V001]).to(device)

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

    # Update Feature Dictionaries
    feat_dict[utt_key] = opt_np
    bits_dict_mh[utt_key] = bits_mh

    if args.one_hot:
        # multi-hot to one-hot
        bits_oh = multi_2_one_hot(
            bits_mh,
            bottle_neck=args.bottleneck_depth
        )

        bits_dict_oh[utt_key] = bits_oh

    # increment progress bar
    bar.next()

# save Dictionaries
np.savez_compressed(FEAT_NPZ, **feat_dict)
np.savez_compressed(BITS_NPZ_MH, **bits_dict_mh)

if args.one_hot:
    np.savez_compressed(BITS_NPZ_OH, **bits_dict_oh)

bar.finish()

# END SCRIPT
