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
from process_data import MfccDataset, Numpy2Tensor, Tensor2Numpy

# Constants
# save to .npz archives
BITS_NPZ = "output_discrete_feats.npz"
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
bottle_neck = 0

if not os.path.isfile(sys_file_loc):
    raise FileNotFoundError("Specified System File D.N.E")

sys_name = os.path.basename(sys_file_loc).split('.')[0]

if sys_name == 'MfccAuto20':

    # def sys bottle_neck
    bottle_neck = 20

    # def network
    sys = MfccAuto(
        bnd=bottle_neck
    ).load(
        save_file=sys_file_loc
    )

# TODO: Add additional systems here once trained :)

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

input_dataset = MfccDataset(
    mfcc_npz=inpt_file_loc,
    transform=tf.Compose([
        Numpy2Tensor()
    ]),
    ret_keys=True
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


def multi_2_one_hot(multi_hot, lookup_table):

    # multi-hot -> integer targets
    int_targets = [
        lookup_table.index(list(mh)) for mh in multi_hot
    ]

    # int targets -> one hot
    one_hot = np.eye(2**bottle_neck)[int_targets]

    return one_hot


# display system
print("\nEncoding using : " + sys_name)
print("------------------------------")

feat_dict = {}
bits_dict = {}

# init progress bar
bar = Bar("Encoding Progress :", max=len(input_dataset))

# create binary lookup table
lookup_table = [
    list(i) for i in itertools.product([0, 1], repeat=bottle_neck)
]

print(len(lookup_table))

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

    bits_mhot = Tensor2Numpy()(
        bits_tensor.cpu()
    )

    # [-1, 1] -> [0, 1]
    bits_mhot[bits_mhot == -1] = 0

    # group bits per feat together
    bits_mhot = bits_mhot.reshape(-1, bottle_neck)

    # update feature dictionary
    feat_dict[utt_key] = opt_np

    # convert bits
    bits_oh = multi_2_one_hot(
        bits_mhot, lookup_table
    )

    bits_dict[utt_key] = bits_oh

    bar.next()

# save Dictionaries
np.savez(FEAT_NPZ, **feat_dict)
np.savez(BITS_NPZ, **bits_dict)

bar.finish()

# END SCRIPT
