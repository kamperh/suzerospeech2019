"""TODO(rpeloff): module doc

Batch iterators usage with TensorFlow or PyTorch data pipelines:
- Iterate over batch elements:
    >>> batcher = RNNBatcher(...)
    >>> for x_batch, x_lengths in batcher:
    >>>     ...
- Or use generator directly in pipeline:
    >>> batcher = iter(RNNBatcher(...))
    >>> data_pipeline = tf.data.Dataset.from_generator(batcher, ...)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: February 2019
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
from typing import Tuple


from constants import NP_FLOAT_DTYPE


# ------------------------------------------------------------------------------ # -----80~100---- #
# Simple batch iterator base class:
# ------------------------------------------------------------------------------ # -----80~100---- #


# def flatten_tuple(*tuples):
#     return tuple([element for tup in tuples for element in tup])


class RNNBatcher:


    def __init__(self, *tensors, batch_size=1, shuffle_first_epoch=True,
                 shuffle_every_epoch=False, small_final_batch=False):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle_first_epoch = shuffle_first_epoch
        self.shuffle_every_epoch = shuffle_every_epoch
        self.small_final_batch = small_final_batch
        # Check tensors have same first dimension
        self.n_tensors = np.shape(tensors)[0]
        self.n_data = np.shape(tensors[0])[0]
        for tensor in tensors:
            if np.shape(tensor)[0] != self.n_data:
                raise ValueError("All elements of tensors must have same first dimension. "
                                 "Got dimensions {} and {}.".format(self.n_data, np.shape(tensor)[0]))
        # Process RNN data (does not assume lengths of corresponding tensor elements are the same)
        self.dim_feats = []
        self.seq_lengths = []
        for tensor in self.tensors:
            tensor_shape = np.shape(tensor[0])
            self.dim_feats.append(tensor_shape[-1] if len(tensor_shape) > 0 else 1)
            self.seq_lengths.append(np.array([np.shape(x)[0] if len(np.shape(x)) > 0 else 0 for x in tensor]))
        if self.small_final_batch:  # additonal small final batch
            self.n_batches = int(np.ceil(self.n_data/self.batch_size))
        else:
            self.n_batches = int(np.floor(self.n_data/self.batch_size))
        self.indices = np.arange(self.n_data)
        # Track for shuffling on first epoch
        self.first_epoch = True


    def __iter__(self):
        if self.first_epoch:  # shuffle on first epoch
            self.first_epoch = False
            if self.shuffle_first_epoch and not self.shuffle_every_epoch:
                self._shuffle()
        if self.shuffle_every_epoch:
            self._shuffle()
        return self._generate_padded_batch()


    def __len__(self):
        return self.n_batches


    def _shuffle(self):
        np.random.shuffle(self.indices)


    def _generate_padded_batch(self):
        for i_batch in range(self.n_batches):
            remain_batch_size = min(self.batch_size,
                                    self.n_data - i_batch*self.batch_size)
            batch_indices = self.indices[i_batch*self.batch_size:
                                         (i_batch*self.batch_size + remain_batch_size)]
            batch_lengths = [lengths[batch_indices] for lengths in self.seq_lengths]
            # Pad to maximum length in batch
            batch_padded_tensors = []
            for i_tensor in range(self.n_tensors):
                if np.max(batch_lengths[i_tensor]) == 0:  # non-sequence tensor
                    tensor_padded = self.tensors[i_tensor][batch_indices]
                else:
                    tensor_padded = np.zeros((len(batch_indices),
                                              np.max(batch_lengths[i_tensor]),
                                              self.dim_feats[i_tensor]),
                                             dtype=NP_FLOAT_DTYPE)
                    for index, length in enumerate(batch_lengths[i_tensor]):
                        seq = self.tensors[i_tensor][batch_indices[index]]
                        tensor_padded[index, :length, :] = seq
                batch_padded_tensors.append(tensor_padded)
            # Yield padded tensor batch
            yield (batch_padded_tensors, batch_lengths)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Batch iterator with bucketing for encoder-decoder models:
# ------------------------------------------------------------------------------ # -----80~100---- #


class BucketRNNBatcher(RNNBatcher):


    def __init__(self, *tensors, batch_size=1, n_buckets=3,
                 shuffle_first_epoch=True, shuffle_every_epoch=False,
                 small_final_batch=False, sort_tensor_index=0):
        # Initialise base iterator class
        super(BucketRNNBatcher, self).__init__(
            *tensors,
            batch_size=batch_size,
            shuffle_first_epoch=shuffle_first_epoch,
            shuffle_every_epoch=shuffle_every_epoch,
            small_final_batch=small_final_batch)
        self.n_buckets = n_buckets
        if self.n_buckets is None:
            self.n_buckets = 1
        # Compute efficient buckets from indices sorted on x_lengths
        bucket_size = int(len(self.seq_lengths[sort_tensor_index])/self.n_buckets)
        indices_sorted = np.argsort(self.seq_lengths[sort_tensor_index])
        self.buckets = []
        for i_bucket in range(self.n_buckets):
            bucket = indices_sorted[i_bucket*bucket_size:
                                    (i_bucket+1)*bucket_size]
            self.buckets.append(bucket)

        if (small_final_batch and
            len(self.seq_lengths[sort_tensor_index]) % self.n_buckets != 0):
            final_bucket_size = self.n_data - self.n_buckets*bucket_size
            final_bucket = indices_sorted[self.n_buckets*bucket_size:
                                          self.n_buckets*bucket_size + final_bucket_size]
            self.buckets.append(final_bucket)
        # Concatenate buckets into complete list of bucket sorted indices
        self.indices = np.concatenate(self.buckets)


    def _shuffle(self):
        for i_bucket in range(self.n_buckets):
            np.random.shuffle(self.buckets[i_bucket])
        # Concatenate buckets into complete list of bucket sorted indices
        self.indices = np.concatenate(self.buckets)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Batch iterator with temperature scheduling for encoder-decoder models:
# ------------------------------------------------------------------------------ # -----80~100---- #


class TemperatureRNNBatcher(BucketRNNBatcher):


    def __init__(self, *tensors, batch_size=1, n_buckets=3, temperatures=None,
                 shuffle_first_epoch=True, shuffle_every_epoch=False,
                 small_final_batch=False, sort_tensor_index=0):
        # Initialise base iterator class
        super(TemperatureRNNBatcher, self).__init__(
            *tensors,
            batch_size=batch_size,
            n_buckets=n_buckets,
            shuffle_first_epoch=shuffle_first_epoch,
            shuffle_every_epoch=shuffle_every_epoch,
            small_final_batch=small_final_batch,
            sort_tensor_index=sort_tensor_index)
        self.temperatures = temperatures
        self.epoch = 0  # track current epoch for temperature schedule


    def __iter__(self):
        current_temperature = (self.temperatures[self.epoch]
                               if self.epoch < len(self.temperatures)
                               else self.temperatures[-1])
        self.epoch += 1  # update epoch for next iter loop
        # Yield padded tensor batch (from base iterator) along with temperature
        for batch_tensors, batch_lengths in super(TemperatureRNNBatcher, self).__iter__():
            yield (batch_tensors, batch_lengths, current_temperature)
