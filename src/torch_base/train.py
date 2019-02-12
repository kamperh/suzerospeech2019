"""
Pytorch Training Script

"""

# general imports
import os
import time
import torch.nn as nn
import argparse as arg
import torchvision.transforms as tf
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim import lr_scheduler


# imports from torch base code
from process_data import *
from networks import MfccAuto
from functions import MaskedLoss


# ----------------------------------------------------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------------------------------------------------

# create Argument Parser
parser = arg.ArgumentParser(
    prog='Train: Speech Compression System:',
    description='training script to train a binary speech (mfcc or filter-bank) compression system'
)

parser.add_argument(
    '--system',
    '-s',
    metavar='SYSTEM',
    type=str,
    required=True,
    choices=['MfccAuto'],
    help='SpeechCompression System'
)

parser.add_argument(
    '--epochs',
    '-e',
    metavar='EPOCHS',
    type=int,
    default=100,
    help='Training epochs'
)

parser.add_argument(
    '--learn_rate',
    '-lr',
    metavar='LEARN_RATE',
    type=float,
    default=0.0001,
    help='Learning rate'
)

parser.add_argument(
    '--gamma',
    '-g',
    metavar='GAMMA',
    type=float,
    default=0.1,
    help='Learning rate decay rate'
)

parser.add_argument(
    '--log',
    '-l',
    metavar='LOG_DIR',
    type=str,
    default='./',
    help='Log directory'
)

parser.add_argument(
    '--train',
    '-tf',
    metavar='TRAIN_FILE',
    type=str,
    help='Training .npz file'
)

parser.add_argument(
    '--valid',
    '-vf',
    metavar='VALID_FILE',
    type=str,
    default='./',
    help='Validation .npz file'
)

parser.add_argument(
    '--save',
    '-sv',
    metavar='SAVE_LOC',
    type=str,
    default='./',
    help='Model save location'
)

parser.add_argument(
    '--batch_size',
    '-bs',
    metavar='BATCH_SIZE',
    type=int,
    default=3,
    help='Batch size'
)

parser.add_argument(
    '--bottleneck_depth',
    '-bnd',
    metavar='BND',
    type=int,
    default=3,
    help='Bottleneck depth'
)

parser.add_argument(
    '--crop_width',
    '-c_w',
    metavar='CROP_WIDTH',
    type=int,
    default=500,
    help='Width to crop input'
)

parser.add_argument(
    '--crop_height',
    '-c_h',
    metavar='CROP_HEIGHT',
    type=int,
    default=13,
    help='Height to crop input'
)

parser.add_argument(
    '--verbose',
    '-v',
    action='store_true'
)

parser.add_argument(
    '--checkpoint',
    '-chkp',
    action='store_true'
)

args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------------------------------------------------------

# GPU || CPU
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# Define System
sys = None

if args.system == 'MfccAuto':
    # def network
    sys = MfccAuto(
        bnd=args.bottleneck_depth,
        input_size=args.crop_height
    )


# model -> device
sys.to(device)

# Loss function
criterion = MaskedLoss(
    criterion=nn.MSELoss()
)

# Adam Optimizer
opt = Adam(
    sys.parameters(), args.learn_rate
)

# MultiStep scheduler
scheduler = lr_scheduler.MultiStepLR(
    optimizer=opt,
    milestones=[30, 80, 140],
    gamma=args.gamma
)

# check train, log and save locations
log_loc = os.path.expanduser(args.log)

if not os.path.isdir(log_loc):
    raise NotADirectoryError('Log directory d.n.e')

save_loc = os.path.expanduser(args.save)

if not os.path.isdir(save_loc):
    raise NotADirectoryError('Save directory d.n.e')


# Mfcc Dataset

train_dataset = MfccDataset(
    mfcc_npz=args.train,
    transform=tf.Compose([
        Numpy2Tensor(),
        CropMfcc(
            t=args.crop_width,
            freq=args.crop_height
        )
    ])
)

valid_dataset = MfccDataset(
    mfcc_npz=args.valid,
    transform=tf.Compose([
        Numpy2Tensor(),
        CropMfcc(
            t=args.crop_width,
            freq=args.crop_height
        )
    ])
)


# Mfcc DataLoader

train_dataLoader = BatchBucketSampler(
    data_source=train_dataset,
    batch_size=args.batch_size,
    num_buckets=3,
    shuffle_every_epoch=True
)

valid_dataLoader = BatchBucketSampler(
    data_source=valid_dataset,
    batch_size=args.batch_size,
    num_buckets=3,
    shuffle_every_epoch=True
)

# dataLoader Dict
dataLoaders = {
    'train': train_dataLoader,
    'valid': valid_dataLoader
}


# start epoch, time previously elapsed, best Loss
t_prev = 0.0
best_loss = 10e8
current_epoch = 1

# def state file
state_file = ''.join([save_loc, '/', sys.name, '_chkp.pt'])

if os.path.isfile(state_file):

    print("Continue Training from m.r.c : ")

    # load checkpoint
    chkp = torch.load(state_file)

    # load previous train time
    t_prev = chkp['time']

    # load previous epoch
    current_epoch = chkp['epoch']

    # load previous loss
    best_loss = chkp['best_loss']

    # load model weights
    sys.load_state_dict(chkp['sys'])

    # load optimizer states
    opt.load_state_dict(chkp['optimizer'])

    # load scheduler states
    scheduler.load_state_dict(chkp['scheduler'])

else:
    print("Training New System : ")

# writer for loss logging
writer = None

if args.verbose:
    writer = SummaryWriter(log_loc)

# start timing
train_start = time.time()

for epoch in range(current_epoch, args.epochs + 1, 1):

    # start epoch
    if args.verbose:
        print("Epoch {}/{}".format(epoch, args.epochs))
        print("--------------------------------------")

    epoch_start = time.time()

    for phase in ['train', 'valid']:

        # running MSE loss
        run_loss = 0.0

        if phase is 'train':
            sys.train(True)
            # step scheduler
            scheduler.step()

        elif phase is 'valid':
            sys.train(False)

        i = 0
        for i, data in enumerate(dataLoaders[phase], 0):

            # data -> device
            inpt, seq_len = data
            inpt = inpt.to(device)

            # zero model gradients
            opt.zero_grad()

            # forward
            out, _ = sys(inpt, seq_len)

            # loss
            loss = criterion(out, target=inpt)

            # running loss
            run_loss += loss.item()

            if phase == 'train':
                # backward & optimise
                loss.backward()
                opt.step()

            del loss

        # epoch loss averaged over batch number
        epoch_loss = run_loss / (i + 1)

        if args.verbose:
            print("Phase: {} Loss : {}".format(phase, epoch_loss))
            writer.add_scalar('{}/loss'.format(phase), epoch_loss, epoch)

        # save best system
        if phase is 'valid' and epoch_loss < best_loss:
            best_loss = epoch_loss
            fn = ''.join([save_loc, '/', sys.name, '.pt'])
            torch.save(sys.state_dict(), fn)

    # end of epoch
    epoch_time = (time.time() - epoch_start) / 60

    if args.verbose:
        print("Epoch time: {} min".format(epoch_time))
        print("-------------------------------------")

    if args.checkpoint:
        # save checkpoint
        chkp = {
            'epoch': epoch + 1,
            'sys': sys.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'time': time.time() - train_start + t_prev
        }
        torch.save(chkp, state_file)

# end of Training
train_time = (time.time() - train_start + t_prev) / 60

if args.verbose:
    print('Total Training Time {} min'.format(train_time))
    print('Best Loss : {}'.format(best_loss))
    print('FIN TRAINING')

# close writer
writer.close()
