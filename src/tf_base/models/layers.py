"""Utility layers for building TensorFlow models.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: February 2019
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf


from flags import FLAGS


# ------------------------------------------------------------------------------ # -----80~100---- #
# Zero-masking for variable length sequences:
# ------------------------------------------------------------------------------ # -----80~100---- #


class ZeroMask(tf.keras.layers.Layer):

    def __init__(self, *, x_lengths=None, padded_input=None, max_length=None):
        super(ZeroMask, self).__init__()
        if x_lengths is None and padded_input is None:
            raise ValueError(
                "One of x_lengths or padded_input must be specified.")
        self.x_lengths = x_lengths
        self.padded_input = padded_input
        self.max_length = max_length

    def build(self, input_shape):
        if self.padded_input is not None:  # get x_lengths from padded input
            self._in_mask = tf.sign(tf.reduce_max(tf.abs(self.padded_input), 2))
            self.x_lengths = tf.math.count_nonzero(self._in_mask, axis=-1)
        self.mask = tf.sequence_mask(
            self.x_lengths, self.max_length, dtype=FLAGS.tf_float_dtype)
        self.mask = tf.expand_dims(self.mask, axis=-1)
        super(ZeroMask, self).build(input_shape)  # call base `build` logic

    def call(self, inputs):
        return tf.math.multiply(inputs, self.mask)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Wrapped convolution block:
# ------------------------------------------------------------------------------ # -----80~100---- #


class WrapConv2D(tf.keras.layers.Layer):

    def __init__(
            self, conv2D=tf.keras.layers.Conv2D, batch_norm=False,
            activation="relu", downsample=None, dropout=None, **conv2D_kwargs):
        """Convolution wrapped with regularizer, activation, and down-sampling.

        Parameters
        ----------
        conv2D : Layer class, optional
            Convolution layer applied to inputs (e.g. tf.keras.layers.Conv2D).
        batch_norm : boolean, optional
            Apply batch norm after convolution, but before activation.
        activation : string, optional
            Activation layer applied after regularized output (e.g. ReLU, PReLU, etc.).
        dropout : Layer or callable, optional
            Dropout layer applied to the activation output (e.g. spatial dropout).
        downsample : Layer or callable, optional
            Down-sampling layer applied to the final output (e.g. max pooling).
        """
        super(WrapConv2D, self).__init__()
        self.conv2D = conv2D(**conv2D_kwargs)
        self.batch_norm = batch_norm
        self.bn2D = tf.keras.layers.BatchNormalization() if self.batch_norm else None
        self.activation = tf.keras.layers.Activation(activation)
        self.dropout = dropout() if dropout is not None else None
        self.downsample = downsample() if downsample is not None else None

    def call(self, inputs):
        outputs = self.conv2D(inputs)
        if self.bn2D is not None:
            outputs = self.bn2D(outputs)
        outputs = self.activation(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        if self.downsample is not None:
            outputs = self.downsample(outputs)
        return outputs

    @property
    def updates(self):
        """Based on Keras TF guide:
        https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
        """
        updates = super(WrapConv2D, self).updates  # get base layer update ops

        if self.bn2D is not None:
            # for old_value, new_value in self.bn2D.updates:
                # updates.append(tf.assign(old_value, new_value))
            updates += self.bn2D.updates

        return updates

    def compute_output_shape(self, input_shape):
        shape = self.conv2D.compute_output_shape(input_shape)
        if self.downsample is not None:
            shape = self.downsample.compute_output_shape(shape)
        return tf.TensorShape(shape)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Residual 2D convolution block:
# ------------------------------------------------------------------------------ # -----80~100---- #


class ResidualConv2D(tf.keras.layers.Layer):
    
    def __init__(
            self, conv2a_depth, conv2b_depth=None, project_shortcut=False,
            conv2a_kernel=(3, 3), conv2b_kernel=(1, 1), deconv=False,
            strides=(1, 1), activation="relu", batch_norm=False, **conv2D_kwargs):
        """Adapted from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64.
        """
        super(ResidualConv2D, self).__init__()
        self.conv2a_depth = conv2a_depth
        self.conv2b_depth = conv2b_depth
        self.output_channels = conv2a_depth if self.conv2b_depth is None else self.conv2b_depth
        self.project_shortcut = project_shortcut
        self.strides = strides
        self.batch_norm = batch_norm
        self.deconv = deconv
        if self.deconv:
            conv2D = tf.keras.layers.Conv2DTranspose
        else:
            conv2D = tf.keras.layers.Conv2D
        # residual conv2D layers
        self.conv2a = conv2D(
            conv2a_depth,
            conv2a_kernel,
            strides=strides,
            activation=None,
            **conv2D_kwargs)
        if self.conv2b_depth is not None:
            self.conv2b = conv2D(
                conv2b_depth,
                conv2b_kernel,
                strides=(1, 1),
                activation=None,
                **conv2D_kwargs)
        self.bn2a = tf.keras.layers.BatchNormalization() if self.batch_norm else None
        self.bn2b = tf.keras.layers.BatchNormalization() if self.batch_norm else None
        self.activation = tf.keras.layers.Activation(activation)

        if project_shortcut or strides != (1, 1):
            # projection layer to match input residual to output dimensions
            # strides downsamples input height and width
            # output_channels increases input channels
            self.conv2project = conv2D(
                self.output_channels,
                (1, 1),  # 1×1 convolutions
                strides=strides,
                activation=None,
                **conv2D_kwargs)
            self.bn2project = tf.keras.layers.BatchNormalization() if self.batch_norm else None

    def call(self, inputs, training=False):
        residual = inputs

        outputs = self.conv2a(inputs)
        if self.bn2a is not None:
            outputs = self.bn2a(outputs)
        outputs = self.activation(outputs)

        if self.conv2b_depth is  not None:
            outputs = self.conv2b(outputs)
            if self.bn2b is not None:
                outputs = self.bn2b(outputs)

        if self.project_shortcut or self.strides != (1, 1):
            residual = self.conv2project(residual)
            if self.bn2project is not None:
                residual = self.bn2project(residual)

        outputs += residual
        outputs = self.activation(outputs)
        return outputs

    @property
    def updates(self):
        """Based on Keras TF guide:
        https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
        """
        updates = super(ResidualConv2D, self).updates  # get base layer update ops

        if self.bn2a is not None:
            # for old_value, new_value in self.bn2a.updates:
            #     updates.append(tf.assign(old_value, new_value))
            updates += self.bn2a.updates

        if self.bn2b is not None:
            # for old_value, new_value in self.bn2b.updates:
            #     updates.append(tf.assign(old_value, new_value))
            updates += self.bn2b.updates

        if self.project_shortcut or self.strides != (1, 1):
            if self.bn2project is not None:
                # for old_value, new_value in self.bn2project.updates:
                #     updates.append(tf.assign(old_value, new_value))
                updates += self.bn2project.updates

        return updates

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_channels
        return tf.TensorShape(shape)

    @property
    def weights(self):
        trainable_variables = []
        trainable_variables += self.conv2a.trainable_variables
        if self.conv2b_depth is not None:
            trainable_variables += self.conv2a.trainable_variables
        if self.project_shortcut or self.strides != (1, 1):
            trainable_variables += self.conv2project.trainable_variables
        return trainable_variables


# ------------------------------------------------------------------------------ # -----80~100---- #
# Residual dense block:
# ------------------------------------------------------------------------------ # -----80~100---- #


class ResidualDense(tf.keras.layers.Layer):
    
    def __init__(
            self, a_units, b_units=None, project_shortcut=False,
            activation="relu", batch_norm=False, **dense_kwargs):
        """Adapted from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64.
        """

        super(ResidualDense, self).__init__()
        self.a_units = a_units
        self.b_units = b_units
        self.output_units = a_units if b_units is None else b_units
        self.project_shortcut = project_shortcut
        self.batch_norm = batch_norm

        # residual dense layers
        self.dense_a = tf.keras.layers.Dense(
            self.a_units,
            activation=None,
            **dense_kwargs)
        if self.b_units is not None:
            self.dense_b = tf.keras.layers.Dense(
                self.b_units,
                activation=None,
                **dense_kwargs)
        self.bn_a = tf.keras.layers.BatchNormalization() if self.batch_norm else None
        self.bn_b = tf.keras.layers.BatchNormalization() if self.batch_norm else None
        self.activation = tf.keras.layers.Activation(activation)

        if project_shortcut:
            # projection layer to match input residual to output dimensions
            # output_units increases input units
            self.dense_project = tf.keras.layers.Dense(
                self.output_units,
                activation=None,
                **dense_kwargs)
            self.bn_project = tf.keras.layers.BatchNormalization() if self.batch_norm else None

    def call(self, inputs, training=False):
        residual = inputs

        outputs = self.dense_a(inputs)
        if self.bn_a is not None:
            outputs = self.bn_a(outputs)
        outputs = self.activation(outputs)
        
        if self.b_units is not None:
            outputs = self.dense_b(outputs)
            if self.bn_b is not None:
                outputs = self.bn_b(outputs)

        if self.project_shortcut:
            residual = self.dense_project(residual)
            if self.bn_project is not None:
                residual = self.bn_project(residual)

        outputs += residual
        outputs = self.activation(outputs)
        return outputs

    @property
    def updates(self):
        """Based on Keras TF guide:
        https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
        """
        updates = super(ResidualDense, self).updates  # get base layer update ops

        if self.bn_a is not None:
            # for old_value, new_value in self.bn2a.updates:
            #     updates.append(tf.assign(old_value, new_value))
            updates += self.bn_a.updates

        if self.bn_b is not None:
            # for old_value, new_value in self.bn2b.updates:
            #     updates.append(tf.assign(old_value, new_value))
            updates += self.bn_b.updates

        if self.project_shortcut:
            if self.bn_project is not None:
                # for old_value, new_value in self.bn2project.updates:
                #     updates.append(tf.assign(old_value, new_value))
                updates += self.bn_project.updates

        return updates

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_units
        return tf.TensorShape(shape)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Temporal attention
# ------------------------------------------------------------------------------ # -----80~100---- #


# class Attention1D(tf.keras.layers.Layer):

#     def __init__(self, *, x_lengths=None, padded_input=None, max_length=None):
#     """Implement attention mechanism on a sequence input.

#     input shape := (batch_size, time_steps, input_dim)

#     Code adapted from: https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
#     """
#     self.input_permute = tf.keras.layers.Permute((2, 1))  # -> (batch_size, input_dim, time_steps)
#     self.attention = tf.keras.layers.Dense(time_steps...??? , activation='softmax')
#     # TODO(rpeloff)
