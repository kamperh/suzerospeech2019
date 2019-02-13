"""
Pytorch Encoding Script

Andre Nortje adnortje@gmail.com

"""

# general imports
import os
import torch
import itertools
import numpy as np
import argparse as arg
from progress.bar import Bar
import torchvision.transforms as tf
from torch.utils.data import DataLoader

# imports from torch base code
from networks import MfccAuto
from process_data import SpeechDataset, Numpy2Tensor, Tensor2Numpy, CropSpeech

# Constants
# save to .npz archives
BITS_NPZ_MH = "output_discrete_feats_multi_hot.npz"
BITS_NPZ_OH = "output_discrete_feats_one_hot.npz"
FEAT_NPZ = "output_feats.npz"

# ----------------------------------------------------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------------------------------------------------

# create Argument Parser
parser = arg.ArgumentParser(
    prog='Encoding Script:',
    description='Encodes input using specified system and saves two .npz files namely: '
                '1) output_discrete_feats.npz --> one-hot binary symbols '
                '2) output_feats.npz --> feature reconstructions '
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

args = parser.parse_args()

# GPU || CPU
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# system file location
sys_file_loc = os.path.expanduser(args.system_file)

# Define System
sys = None
# def bottle-neck (1 x 128 one-hot -> 7 bits multi-hot)
bottle_neck = 7

if not os.path.isfile(sys_file_loc):
    raise FileNotFoundError("Specified System File D.N.E")

sys_name = os.path.basename(sys_file_loc).split('.')[0]

if sys_name == "MfccAuto":

    print(
        "System: {}".format(sys_name)
    )

    # def network
    sys = MfccAuto(
        bnd=bottle_neck,
        input_size=13
    ).load(
        save_file=sys_file_loc
    )

elif sys_name == "FbankAuto":

    print(
        "System: {}".format(sys_name)
    )


# TODO: Add additional systems here once trained !!

else:
    err_msg = "Oops, we didn't train this : {}"
    raise ValueError(err_msg.format(args.system_file))

# model -> device
sys.to(device)

# model -> inference mode
sys.train(False)

# Dataset and DataLoader Definition

inpt_file_loc = os.path.expanduser(args.input_file)

if not os.path.isfile(inpt_file_loc):
    raise FileNotFoundError("Specified Input File D.N.E")

input_dataset = SpeechDataset(
    speech_npz=inpt_file_loc,
    transform=tf.Compose([
        Numpy2Tensor(),
        CropSpeech(
            t=2000,
            feat=13
        )
    ]),
    return_keys=True
)

input_dataLoader = DataLoader(
    dataset=input_dataset,
    batch_size=1,
    shuffle=False
)

"""
multi_hot_2_one_hot function
    
    ref https://www.reddit.com/r/MachineLearning/comments/31fk7i/converting_target_indices_to_onehotvector/
"""


def multi_2_one_hot(multi_hot, lookup_dict):

    # multi-hot -> integer targets
    int_targets = [
        lookup_dict["".join(map(str, map(int, mh)))] for mh in multi_hot
    ]

    # int targets -> one hot
    one_hot = np.eye(2**bottle_neck)[int_targets]

    return one_hot


# display system
print("\nEncoding using : " + sys_name)
print("------------------------------")

# encoded feat dict
feat_dict = {}
bits_dict_mh = {}
bits_dict_oh = {}

# binary code lookup table
lookup_table = {}

# init progress bar
bar = Bar("Encoding Progress :", max=len(input_dataset))


# all possible binary codes
bin_codes = [
    list(i) for i in itertools.product([0, 1], repeat=bottle_neck)
]

# create lookup-table
for j, bit_code in enumerate(bin_codes, 0):
    # multi-hot bit key
    mh_bit_key = "".join(
        map(str, bit_code)
    )
    # key -> symbol
    lookup_table[mh_bit_key] = j


for data in input_dataLoader:

    # Key and input feature
    utt_key, inpt_tensor = data
    utt_key = utt_key[0]

    # forward & encode
    with torch.no_grad():
        # run in inference mode
        opt_tensor, bits_tensor = sys(
            inpt_tensor.to(device)
        )

    # Torch.Tensor -> Np arrays
    opt_np = Tensor2Numpy()(
        opt_tensor.squeeze(0).cpu()
    )

    bits_mh = Tensor2Numpy()(
        bits_tensor.cpu()
    )

    # [-1, 1] -> [0, 1]
    bits_mh[bits_mh == -1] = 0

    # group bits per feat together
    bits_mh = bits_mh.reshape(-1, bottle_neck)

    # update feature dictionary
    feat_dict[utt_key] = opt_np

    # convert bits
    bits_oh = multi_2_one_hot(
        bits_mh, lookup_table
    )

    bits_dict_oh[utt_key] = bits_oh
    bits_dict_mh[utt_key] = bits_mh

    bar.next()

# save Dictionaries
np.savez_compressed(FEAT_NPZ, **feat_dict)
np.savez_compressed(BITS_NPZ_MH, **bits_dict_mh)
np.savez_compressed(BITS_NPZ_OH, **bits_dict_oh)

bar.finish()

# END SCRIPT
