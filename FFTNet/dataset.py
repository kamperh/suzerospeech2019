from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from utils import mu_law_encode
from os import path


class FilterbankDataset(Dataset):
    def __init__(self,
                 data_dir,
                 receptive_field,
                 sample_size=5000,
                 upsample_factor=200,
                 quantization_channels=256,
                 use_local_condition=True,
                 noise_injecting=True,
                 feat_transform=None):

        self.receptive_field = receptive_field
        self.sample_size = sample_size
        self.upsample_factor = upsample_factor
        self.quantization_channels = quantization_channels
        self.use_local_condition = use_local_condition
        self.feat_transform = feat_transform
        self.noise_injecting = noise_injecting

        audio = np.load(path.join(data_dir, "unsegmented_train_audio_fftnet.npz"))
        fbank = np.load(path.join(data_dir, "unsegmented_train_mfcc_fftnet.npz"))
        self.audio_buffer = list()
        self.fbank_buffer = list()
        for key in sorted(audio.keys()):
            if len(audio[key]) - self.sample_size - self.receptive_field > 0:
                self.audio_buffer.append(audio[key])
                if self.use_local_condition:
                    feat = fbank[key]
                    if self.feat_transform is not None:
                        feat = self.feat_transform(feat)
                    self.fbank_buffer.append(feat)

    def __len__(self):
        return len(self.audio_buffer)

    def __getitem__(self, index):
        audios = self.audio_buffer[index]
        rand_pos = np.random.randint(0, len(audios) - self.sample_size)

        if self.use_local_condition:
            local_condition = self.fbank_buffer[index]
            local_condition = np.repeat(local_condition, self.upsample_factor, axis=0)
            local_condition = local_condition[rand_pos:rand_pos + self.sample_size]
        else:
            audios = np.pad(audios, [[self.receptive_field, 0], [0, 0]], 'constant')
            local_condition = None

        audios = audios[rand_pos:rand_pos + self.sample_size]
        target = mu_law_encode(audios, self.quantization_channels)
        if self.noise_injecting:
            noise = np.random.normal(0.0, 1.0/self.quantization_channels, audios.shape)
            audios = audios + noise

        audios = np.pad(audios, [[self.receptive_field, 0], [0, 0]], 'constant')
        local_condition = np.pad(local_condition, [[self.receptive_field, 0], [0, 0]], 'constant')
        return torch.FloatTensor(audios), torch.LongTensor(target), torch.FloatTensor(local_condition)
