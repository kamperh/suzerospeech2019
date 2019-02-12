from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FilterbankDataset
from fftnet import FFTNet
from utils import apply_moving_average, ExponentialMovingAverage, mu_law_decode, write_wav


def save_checkpoint(device, args, model, optimizer, step, checkpoint_dir, ema=None):
    model = model.module if isinstance(model, nn.DataParallel) else model

    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": step}
    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}.pt".format(step))
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))

    if ema is not None:
        averaged_model = clone_as_averaged_model(device, args, model, ema)
        averaged_checkpoint_state = {
            "model": averaged_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": step}
        checkpoint_path = os.path.join(
            checkpoint_dir, "model.ckpt-{}.ema.pt".format(step))
        torch.save(averaged_checkpoint_state, checkpoint_path)
        print("Saved averaged checkpoint: {}".format(checkpoint_path))


def clone_as_averaged_model(device, args, model, ema):
    assert ema is not None
    averaged_model = create_model(args).to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


def create_model(hparams):
    if hparams.feature_type == 'mcc':
        lc_channel = args.mcep_dim + 3
    else:
        lc_channel = args.num_mels

    return FFTNet(n_stacks=args.n_stacks,
                  fft_channels=args.fft_channels,
                  quantization_channels=args.quantization_channels,
                  local_condition_channels=lc_channel)


def train_fn(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    upsample_factor = int(args.frame_shift_ms / 1000 * args.sample_rate)

    model = create_model(args)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    if args.resume is not None:
        log("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint['steps']
    else:
        global_step = 0

    print("receptive field: {0} ({1:.2f}ms)".format(
        model.receptive_field, model.receptive_field / args.sample_rate * 1000))

    if args.feature_type == "mcc":
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(args.data_dir, 'mean.npy'))
        scaler.scale_ = np.load(os.path.join(args.data_dir, 'scale.npy'))
        feat_transform = transforms.Compose([lambda x: scaler.transform(x)])
    else:
        feat_transform = None

    dataset = FilterbankDataset(data_dir=args.data_dir,
                                receptive_field=model.receptive_field,
                                sample_size=args.sample_size,
                                upsample_factor=upsample_factor,
                                quantization_channels=args.quantization_channels,
                                use_local_condition=args.use_local_condition,
                                noise_injecting=args.noise_injecting,
                                feat_transform=feat_transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    ema = ExponentialMovingAverage(args.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    while global_step < args.training_steps:
        for i, data in enumerate(dataloader, 0):
            audio, target, local_condition = data
            target = target.squeeze(-1)
            local_condition = local_condition.transpose(1, 2)
            audio, target, h = audio.to(device), target.to(device), local_condition.to(device)

            optimizer.zero_grad()
            output = model(audio[:, :-1, :], h[:, :, 1:])
            loss = criterion(output, target)
            print('step [%3d]: loss: %.3f' % (global_step, loss.item()))

            loss.backward()
            optimizer.step()

            # update moving average
            if ema is not None:
                apply_moving_average(model, ema)

            global_step += 1

            if global_step % args.checkpoint_interval == 0:
                save_checkpoint(device, args, model, optimizer, global_step, args.checkpoint_dir, ema)
                out = output[1, :, :]
                samples = out.argmax(0)
                waveform = mu_law_decode(np.asarray(samples[model.receptive_field:]), args.quantization_channels)
                write_wav(waveform, args.sample_rate,
                          os.path.join(args.checkpoint_dir, "train_eval_{}.wav".format(global_step)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mels", type=int, default=45)
    parser.add_argument("--num_freq", type=int, default=1025)
    parser.add_argument("--mcep_dim", type=int, default=24)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--feature_type", default="melspc")  # mcc or melspc
    parser.add_argument("--frame_shift_ms", type=int, default=10)
    parser.add_argument("--noise_injecting", type=bool, default=True)
    parser.add_argument("--data_dir", default="data", help="Metadata file which contains the keys of audio and melspec")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help='Moving average decay rate.')
    parser.add_argument("--num_workers", type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument("--resume", type=str, default=None, help='Checkpoint path to resume')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints/', help='Directory to save checkpoints.')
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--use_local_condition", action="store_true")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--sample_size", type=int, default=16000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--training_steps", type=int, default=200000)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--n_stacks", type=int, default=11)
    parser.add_argument("--fft_channels", type=int, default=256)
    parser.add_argument("--quantization_channels", type=int, default=256)
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_fn(args)
