"""TODO(rpeloff): module doc

Batch iterators usage with TensorFlow or PyTorch data pipelines:
- Iterate over batch elements:
    >>> batcher = SimpleBatcher(...)
    >>> for x_batch, x_lengths in batcher:
    >>>     ...
- Or use generator directly in pipeline:
    >>> batcher = iter(SimpleBatcher(...))
    >>> data_pipeline = tf.data.Dataset.from_generator(batcher, ...)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: February 2019
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np


from constants import NP_FLOAT_DTYPE


# ------------------------------------------------------------------------------ # -----80~100---- #
# Simple batch iterator:
# ------------------------------------------------------------------------------ # -----80~100---- #

class RNNBatcher:


    def __init__(self, x_data, batch_size, shuffle_first_epoch=True, shuffle_every_epoch=False):
        self.x_data = x_data
        self.batch_size = batch_size
        self.shuffle_first_epoch = shuffle_first_epoch
        self.shuffle_every_epoch = shuffle_every_epoch
        # Process RNN data
        self.num_feats = np.shape(self.x_data[0])[-1]
        self.x_lengths = np.array([np.shape(x)[0] for x in self.x_data])
        self.num_batches = int(len(self.x_lengths)/self.batch_size)
        self.indices = np.arange(len(self.x_lengths))
        # Shuffle on first epoch
        if self.shuffle_first_epoch:
            np.random.shuffle(self.indices)


    def __iter__(self):
        if self.shuffle_every_epoch:
            np.random.shuffle(self.indices)
        for i_batch in range(self.num_batches):
            batch_indices = self.indices[i_batch*self.batch_size:
                                         (i_batch+1)*self.batch_size]
            batch_x_lengths = self.x_lengths[batch_indices]
            # Pad to maximum length in batch
            batch_x_padded = np.zeros(
                (len(batch_indices), np.max(batch_x_lengths), self.num_feats),
                dtype=NP_FLOAT_DTYPE)
            for index, length in enumerate(batch_x_lengths):
                seq = self.x_data[batch_indices[index]]
                batch_x_padded[index, :length, :] = seq
            # Yield padded batch
            yield (batch_x_padded, batch_x_lengths)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Simple batch iterator with bucketing:
# ------------------------------------------------------------------------------ # -----80~100---- #

class BucketRNNBatcher:


    def __init__(self, x_data, batch_size, num_buckets, shuffle_every_epoch=False):
        self.x_data = x_data
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle_every_epoch = shuffle_every_epoch
        # Process RNN data
        self.num_feats = np.shape(self.x_data[0])[-1]
        self.x_lengths = np.array([np.shape(x)[0] for x in self.x_data])
        self.num_batches = int(len(self.x_lengths)/self.batch_size)
        # Compute efficient buckets from indices sorted on x_lengths
        indices_sorted = np.argsort(self.x_lengths)
        bucket_size = int(len(self.x_lengths)/self.num_buckets)
        self.buckets = []
        for i_bucket in range(self.num_buckets):
            bucket = indices_sorted[i_bucket*bucket_size:
                                    (i_bucket+1)*bucket_size]
            self.buckets.append(bucket)
        # Shuffle on first epoch
        self._shuffle_buckets()
        # Concatenate buckets into complete list of bucket sorted indices
        self.indices = np.concatenate(self.buckets)


    def _shuffle_buckets(self):
        for i_bucket in range(self.num_buckets):
            np.random.shuffle(self.buckets[i_bucket])


    def __iter__(self):
        if self.shuffle_every_epoch:
            self._shuffle_buckets()
            self.indices = np.concatenate(self.buckets)
        for i_batch in range(self.num_batches):
            batch_indices = self.indices[i_batch*self.batch_size:
                                         (i_batch+1)*self.batch_size]
            batch_x_lengths = self.x_lengths[batch_indices]
            # Pad to maximum length in batch
            batch_x_padded = np.zeros(
                (len(batch_indices), np.max(batch_x_lengths), self.num_feats),
                dtype=NP_FLOAT_DTYPE)
            for index, length in enumerate(batch_x_lengths):
                seq = self.x_data[batch_indices[index]]
                batch_x_padded[index, :length, :] = seq
            # Yield padded batch
            yield (batch_x_padded, batch_x_lengths)
