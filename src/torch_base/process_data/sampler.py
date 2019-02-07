# imports
import numpy as np
from torch.utils.data.sampler import Sampler

"""
Batch Bucket Sampling Class

    buckets mfcc's according to length and returns batch (list)
    
        Args:
            data_source         (Dataset) : torch dataset
            batch_size          (int)     : size of mini-batch
            num_buckets         (int)     : number of buckets to split data into
            shuffle_every_epoch (bool)    : default False, set to True to shuffle buckets every epoch
    
"""


class BatchBucketSampler(Sampler):

    def __init__(self, data_source,
                 batch_size, num_buckets,
                 shuffle_every_epoch=False):

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle_every_epoch = shuffle_every_epoch

        # get data lengths
        self.x_lengths = np.array([
            x.size(0) for x in self.data_source
        ])

        self.num_batches = len(self.x_lengths) // self.batch_size

        sorted_indices = np.argsort(
            self.x_lengths
        )

        bucket_size = int(
            len(self.x_lengths)/self.num_buckets
        )

        # indices -> buckets
        self.buckets = []

        for i_b in range(self.num_buckets):
            bucket = sorted_indices[
                     i_b*bucket_size:(i_b+1)*bucket_size
                     ]
            self.buckets.append(bucket)

        # on 1st epoch
        self._shuffle_buckets()

        # concat buckets into list indices
        self.indices = np.concatenate(
            self.buckets
        )

    def __iter__(self):

        if self.shuffle_every_epoch:
            self._shuffle_buckets()
            self.indices = np.concatenate(
                self.buckets
            )

        for i_b in range(self.num_batches):
            # batch indices
            batch_indices = self.indices[
                            i_b*self.batch_size:(i_b+1)*self.batch_size
                            ]
            batch = [self.data_source[i] for i in batch_indices]
            yield(batch)

    def _shuffle_buckets(self):

        for i_b in range(len(self.buckets)):
            np.random.shuffle(
                self.buckets[i_b]
            )

        return

    def __len__(self):
        return len(self.data_source)
