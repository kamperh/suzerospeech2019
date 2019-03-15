"""Layers for building convolutional neural networks.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: February 2019
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf



# ------------------------------------------------------------------------------ # -----80~100---- #
# General convolution layer block:
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
        shape = tf.TensorShape(input_shape).as_list()
        # ...
        return tf.TensorShape(shape)





# ------------------------------------------------------------------------------ # -----80~100---- #
# TODO(reploff) OLD---TO BE REMOVED
# ------------------------------------------------------------------------------ # -----80~100---- #


# ------------------------------------------------------------------------------ # -----80~100---- #
# Convolutional neural network (CNN):
# ------------------------------------------------------------------------------ # -----80~100---- #


def build_cnn(
        x_input, filters, kernel_size, strides, padding="valid",
        data_format="channels_last", dilation_rate=None, activation=None,
        pool_size=None, use_bias=True, keep_prob=1., spatial_dropout=False,
        apply_batch_norm=False, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", conv2d_kwargs=None, pool2d_kwargs=None,
        name="cnn"):
    """Build a simple convolutional neural network (CNN).

    Based on tf.keras.layers.Conv2D and tf.keras.layers.MaxPool2D. Optional
    additional parameters are passed to these classes using the `conv2d_kwargs`
    and `pool2d_kwargs` dicts (see docs for more information on their
    parameters).

    Parameters
    ----------
    x_input : tensor
        Input tensor to the network (see tf.keras.layers.Input).
    filters : list of integers
        Sequence of output dimensions for each 2D convolutional layer (i.e. number of output filters after each layer).
    kernel_size : list of list of 2 integers
        Sequence of lists, each specifying the height and width of the 2D convolution window in the corresponding layer.
    strides : list of list of 2 integers
        Sequence of lists, each specifying the strides of the 2D convolution along the height and width in the corresponding layer.
    padding : str, optional
        One of "valid" or "same" (the default is "valid", which does not perform padding).
    data_format : str, optional
        One of "channels_last" or "channels_first" (the default is "channels_last", which correspond to inputs with shape [batch, height, width, channels]).
    dilation_rate : list of list of 2 integers, optional
        Sequence of lists, each specifying the dilation rate to use for dilated convolution in the corresponding layer (the default is None, which does not use dilated convolutions). Note, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
    activation : str or function, optional
        Activation function to use after each convolution layer (the default is None, which applies linear activation a(x) = x).
    pool_size : list of list of 2 integers, optional
        Sequence of lists, each specifying the height and width of the 2D max pooling window in the corresponding layer (the default is None, which does not apply the max pooling operation after each convolutional layer). Note, specifying None for a particular layer will not apply max pooling in that layer.
    use_bias : bool, optional
        Specify whether layers use a bias vector (the default is True, which adds bias variable to output filters)
    keep_prob : float or tensor, optional
        Fraction of units to keep in the dropout layer (the default is 1.0, which keeps all units).
    spatial_dropout : bool, optional
        Specify whether to drop entire 2D feature maps (the default is False, which drops individual elements).
    apply_batch_norm : bool, optional
        Specify whether to apply batch norm to the output of each convolutional layer (the default is False, which does not apply batch norm).
    kernel_initializer : str, optional
        Initializer for the kernel weights matrix (the default is "glorot_uniform", which applies Glorot/Xavier uniform initialization).
    bias_initializer : str, optional
        Initializer for the bias vector (the default is "zeros", which initializes with zeros everywhere).
    conv2d_kwargs : dict, optional
        Additional keyword arguments to be passed to tf.keras.layers.Conv2D (the default is None).
    pool2d_kwargs : dict, optional
        Additional keyword arguments to be passed to tf.keras.layers.MaxPool2D (the default is None).
    name : str, optional
        Name scope for the convolutional neural network (the default is "cnn").

    Notes
    -----
    Output size after covolution for "same" padding:
    - output_height = ceil(float(in_height) / float(strides[1]))
    - output_width  = ceil(float(in_width) / float(strides[2]))

    Output size after covolution for "valid" padding:
    - output_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    - output_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

    See https://www.tensorflow.org/api_guides/python/nn#Convolution for more
    information on how output size and padding is calculated.
    """
    if dilation_rate is None:
        dilation_rate = [(1, 1)] * len(filters)
    if pool_size is None:
        pool_size = [None] * len(filters)
    if conv2d_kwargs is None:
        conv2d_kwargs = {}
    if pool2d_kwargs is None:
        pool2d_kwargs = {}
    assert (len(filters) == len(kernel_size) and
            len(filters) == len(strides) and
            len(filters) == len(dilation_rate) and
            len(filters) == len(pool_size)), ("The first dimension of filters, "
                                              "kernel_size, dilation_rate and "
                                              "pool_size must be the same.")
    x_output = x_input
    for index, conv_params in enumerate(zip(filters, kernel_size,
                                            strides, dilation_rate,
                                            pool_size)):
        # with tf.variable_scope("cnn_layer_{}".format(index)):
        # CNN layer params
        (conv_filters, conv_kernel_size,
         conv_strides, conv_dilation_rate,
         conv_pool_size) = conv_params  # unpack the conv layer params
        # Convolution
        conv_layer = tf.keras.layers.Conv2D(
            conv_filters, conv_kernel_size, strides=conv_strides,
            padding=padding, data_format=data_format,
            dilation_rate=conv_dilation_rate, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **conv2d_kwargs)
        x_output = conv_layer(x_output)
        # Dropout
        if spatial_dropout:
            dropout = tf.keras.layers.SpatialDropout2D(
                rate=1-keep_prob, data_format=data_format)
        else:
            dropout = tf.keras.layers.Dropout(rate=1-keep_prob)
        x_output = dropout(x_output)
        # Max pooling
        if conv_pool_size is not None:
            max_pool = tf.keras.layers.MaxPool2D(
                conv_pool_size, data_format=data_format, **pool2d_kwargs)
            x_output = max_pool(x_output)
        # Batch norm (last to normalize input to next layer)
        if apply_batch_norm:
            batch_norm = tf.keras.layers.BatchNormalization()
            x_output = batch_norm(x_output)
    return tf.keras.models.Model(x_input, x_output, name=name)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Transpose CNN (or "Deconvolution"):
# ------------------------------------------------------------------------------ # -----80~100---- #


def build_transpose_cnn(
        x_input, filters, kernel_size, strides, output_shape=None,
        padding="valid", data_format="channels_last", dilation_rate=None,
        activation=None, use_bias=True, keep_prob=1., spatial_dropout=False,
        apply_batch_norm=False, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", conv2d_kwargs=None, name="transpose_cnn"):
    """Build a transposed convolutional neural network (CNN) using unpooling.

    The transposed convolution is an approximation of the gradient of
    convolution. In particular, this approach approximates the max pooling
    gradient and applies max unpooling to feature maps before applying a
    classical convolution based on specified filters, kernel size, etc.

    See `build_upsampling_cnn` for an upsampling approach to transposed
    convolution which approximates sum pooling gradient. See this answer for
    comparison between the two approaches: https://stackoverflow.com/a/48228379.

    Parameters
    ----------
    See doc for `build_cnn`, which has mostly the same parameters.

    Based on tf.keras.layers.Conv2DTranspose and tf.keras.layers.MaxPool2D.
    Optional additional parameters are passed to these classes using the
    `conv2d_kwargs` and `pool2d_kwargs` dicts (see docs for more information on
    their parameters).

    The parameter `output_shape` should be an integer tensor with shape [2]
    (for example `tf.placeholder_with_default([0, 0], shape=[2])`) which sets
    the desired output height and width after the final tranpose convolutional
    layer (by cropping or zero padding the output).

    Notes
    -----
    Output size after transpose covolution for "same" padding:
    - output_height = in_height * strides[1]
    - output_width  = in_width * strides[2]

    Output size after transpose covolution for "valid" padding:
    - output_height = in_height * strides[1] + max(filter_height - strides[1], 0)
    - output_width  = in_width * strides[2] + max(filter_width - strides[2], 0)

    See https://www.tensorflow.org/api_guides/python/nn#Convolution for more
    information on how output size and padding is calculated.
    """
    if dilation_rate is None:
        dilation_rate = [(1, 1)] * len(filters)
    if conv2d_kwargs is None:
        conv2d_kwargs = {}
    assert (len(filters) == len(kernel_size) and
            len(filters) == len(strides) and
            len(filters) == len(dilation_rate)), (
                "The first dimension of filters, "
                "kernel_size, dilation_rate and "
                "pool_size must be the same.")
    x_output = x_input
    for index, conv_params in enumerate(zip(filters, kernel_size,
                                            strides, dilation_rate)):
        # CNN layer params
        (conv_filters, conv_kernel_size,
         conv_strides, conv_dilation_rate) = conv_params  # unpack the layer params
        # Transpose convolution (unpooling + convolution)
        deconv_layer = tf.keras.layers.Conv2DTranspose(
            conv_filters, conv_kernel_size, strides=conv_strides,
            padding=padding, data_format=data_format,
            dilation_rate=conv_dilation_rate, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **conv2d_kwargs)
        x_output = deconv_layer(x_output)
        # Dropout
        if spatial_dropout:
            dropout = tf.keras.layers.SpatialDropout2D(
                rate=1-keep_prob, data_format=data_format)
        else:
            dropout = tf.keras.layers.Dropout(rate=1-keep_prob)
        x_output = dropout(x_output)
        # Batch norm (last to normalize input to next layer)
        if apply_batch_norm:
            batch_norm = tf.keras.layers.BatchNormalization()
            x_output = batch_norm(x_output)
    # Crop/zero pad output to resize height and width to `output_shape`
    if output_shape is not None:
        shape_contains_zero = tf.greater(
            tf.reduce_sum(tf.cast(tf.equal(output_shape, [0, 0]), dtype=tf.int32)),
            tf.constant(0))
        resize_func = lambda x: tf.cond(
            shape_contains_zero,
            true_fn=lambda: x,
            false_fn=lambda: tf.image.resize_image_with_crop_or_pad(
                x, target_height=output_shape[0], target_width=output_shape[1]))
        resize = tf.keras.layers.Lambda(resize_func)  # wrap as a Layer object
        x_output = resize(x_output)
    return tf.keras.models.Model(x_input, x_output, name=name)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Upsampling CNN (classical upsampling followed by regular convolution):
# ------------------------------------------------------------------------------ # -----80~100---- #


def build_upsample_cnn(
        x_input, factors, filters, kernel_size, strides, output_shape=None,
        padding="valid", data_format="channels_last", dilation_rate=None,
        activation=None, use_bias=True, keep_prob=1., spatial_dropout=False,
        apply_batch_norm=False, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", conv2d_kwargs=None, name="upsample_cnn"):
    """Build a transposed convolutional neural network (CNN) using upsampling.

    See `build_transpose_cnn` for more information on transposed convolutions.

    Parameters
    ----------
    See doc for `build_transpose_cnn`, which has mostly the same parameters.

    The parameter `factors` is a sequence of lists, each specifying the
    upsampling factors for height and width in the corresponding layer.

    Notes
    -----
    Output size after upsampling:
    - upsample_height = in_height * factors[1]
    - upsample_width = in_width * factors[2]

    Output size after covolution for "same" padding:
    - output_height = ceil(float(upsample_height) / float(strides[1]))
    - output_width  = ceil(float(upsample_height) / float(strides[2]))

    Output size after covolution for "valid" padding:
    - output_height = ceil(float(upsample_height - filter_height + 1) / float(strides[1]))
    - output_width  = ceil(float(upsample_height - filter_width + 1) / float(strides[2]))
    """
    if dilation_rate is None:
        dilation_rate = [(1, 1)] * len(filters)
    if conv2d_kwargs is None:
        conv2d_kwargs = {}
    assert (len(filters) == len(kernel_size) and
            len(filters) == len(strides) and
            len(filters) == len(dilation_rate) and 
            len(filters) == len(factors)), ("The first dimension of filters, "
                                            "kernel_size, dilation_rate and "
                                            "pool_size must be the same.")
    x_output = x_input
    for index, conv_params in enumerate(zip(filters, kernel_size, strides,
                                            dilation_rate, factors)):
        # CNN layer params
        (conv_filters, conv_kernel_size,
         conv_strides, conv_dilation_rate,
         upsample_factors) = conv_params  # unpack the layer params
        # Upsample
        upsample = tf.keras.layers.UpSampling2D(
            upsample_factors, data_format=data_format)
        x_output = upsample(x_output)
        # Convolution
        conv_layer = tf.keras.layers.Conv2D(
            conv_filters, conv_kernel_size, strides=conv_strides,
            padding=padding, data_format=data_format,
            dilation_rate=conv_dilation_rate, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **conv2d_kwargs)
        x_output = conv_layer(x_output)
        # Dropout
        if spatial_dropout:
            dropout = tf.keras.layers.SpatialDropout2D(
                rate=1-keep_prob, data_format=data_format)
        else:
            dropout = tf.keras.layers.Dropout(rate=1-keep_prob)
        x_output = dropout(x_output)
        # Batch norm (last to normalize input to next layer)
        if apply_batch_norm:
            batch_norm = tf.keras.layers.BatchNormalization()
            x_output = batch_norm(x_output)
    # Crop/zero pad output to resize height and width to `output_shape`
    if output_shape is not None:
        shape_contains_zero = tf.greater(
            tf.reduce_sum(tf.cast(tf.equal(output_shape, [0, 0]), dtype=tf.int32)),
            tf.constant(0))
        resize_func = lambda x: tf.cond(
            shape_contains_zero,
            true_fn=lambda: x,
            false_fn=lambda: tf.image.resize_image_with_crop_or_pad(
                x, target_height=output_shape[0], target_width=output_shape[1]))
        resize = tf.keras.layers.Lambda(resize_func)  # wrap as a Layer object
        x_output = resize(x_output)
    return tf.keras.models.Model(x_input, x_output, name=name)


# TODO(rpeloff): see https://pytorch.org/docs/stable/_modules/torch/nn/modules/pixelshuffle.html
# def build_pixelshuffle_cnn(...) ?
