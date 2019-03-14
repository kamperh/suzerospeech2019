"""Various loss functions.

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
# Reconstruction loss functions:
# ------------------------------------------------------------------------------ # -----80~100---- #

def mse_reconstruction_loss(x_target, x_decoded, data_variance=None,
                            sequence_lengths=None):
    """
    Specify data_variance for normalised mean squared error (NMSE)
    reconstruction loss, where `data_variance = np.var(train_data)`.
    """
    if sequence_lengths is None:
        reconstruction_loss = tf.losses.mean_squared_error(x_target, x_decoded)
    else:
        reconstruction_loss = tf.reduce_mean(
            tf.square(x_target - x_decoded), axis=-1)
        # compute zero-padding mask
        mask = tf.sign(tf.reduce_max(tf.abs(x_target), axis=2))
        reconstruction_loss *= mask
        # mask zero-padding and average over actual sequence lengths
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=-1)
        reconstruction_loss /= tf.cast(sequence_lengths, FLAGS.tf_float_dtype)
        # finally average loss over batch
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    if data_variance is not None:
        reconstruction_loss /= data_variance  # normalised MSE
    return reconstruction_loss


def binary_cross_entropy_reconstruction_loss(x_target, x_decoded, sequence_lengths=None, eps=1e-10):
    """
    """
    if sequence_lengths is None:
#         x_sigmoid = tf.sigmoid(x_decoded)  # apply sigmoid to output logits
#         reconstruction_loss = tf.reduce_mean(
#             -1. * tf.reduce_sum(
#                 x_target * tf.log(x_sigmoid + eps) + (1 - x_target) * tf.log(1 - x_sigmoid + eps),
#                 axis=1))
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=x_target, logits=x_decoded), axis=-1))
    else:
        x_sigmoid = tf.sigmoid(x_decoded)  # apply sigmoid to output logits
        reconstruction_loss = -1. * tf.reduce_sum(
            x_target * tf.log(x_sigmoid + eps) + (1 - x_target) * tf.log(1 - x_sigmoid + eps),
            axis=-1)
        # compute zero-padding mask
        mask = tf.sign(tf.reduce_max(tf.abs(x_target), axis=2))
        reconstruction_loss *= mask
        # mask zero-padding and average over actual sequence lengths
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=-1)
        reconstruction_loss /= tf.cast(sequence_lengths, FLAGS.tf_float_dtype)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    return reconstruction_loss


def cross_entropy_reconstruction_loss(x_target, x_decoded, sequence_lengths=None, eps=1e-10):
    """
    """
    if sequence_lengths is None:
        x_sigmoid = tf.sigmoid(x_decoded)  # apply sigmoid to output logits
        reconstruction_loss = tf.reduce_mean(
            -1. * tf.reduce_sum(x_target * tf.log(x_sigmoid + eps), axis=-1))
    else:
        x_sigmoid = tf.sigmoid(x_decoded)  # apply sigmoid to output logits
        reconstruction_loss = -1. * tf.reduce_sum(
            x_target * tf.log(x_sigmoid + eps), axis=-1)
        # compute zero-padding mask
        mask = tf.sign(tf.reduce_max(tf.abs(x_target), axis=2))
        reconstruction_loss *= mask
        # mask zero-padding and average over actual sequence lengths
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=-1)
        reconstruction_loss /= tf.cast(sequence_lengths, FLAGS.tf_float_dtype)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    return reconstruction_loss


# ------------------------------------------------------------------------------ # -----80~100---- #
# Variational Auto-Encoder (VQ-VAE) loss functions:
# ------------------------------------------------------------------------------ # -----80~100---- #


def vae_loss_gaussian(reconstruction_loss, z_mean=None, z_log_sigma_sq=None,
                      sigma_sq=1.):
    """
    Use p(x|z) = Normal(x; f(z), sigma^2 I), with y = f(z) the decoder output,
    and mean squared error reconstruction loss.
    """
    reconstruction_loss = 1./(2*sigma_sq) * reconstruction_loss  # scaled MSE
    regularisation_loss = -0.5*tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    vae_loss = reconstruction_loss + tf.reduce_mean(regularisation_loss)
    return vae_loss, regularisation_loss


def vae_loss_bernoulli(reconstruction_loss, z_mean=None, z_log_sigma_sq=None):
    """
    Use a Bernoulli distribution for p(x|z), with the y = f(z) the mean,
    and cross-entropy reconstruction loss.
    """
    regularisation_loss = -0.5*tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    vae_loss = tf.reduce_mean(reconstruction_loss + regularisation_loss)
    return vae_loss, regularisation_loss


# ------------------------------------------------------------------------------ # -----80~100---- #
# Vector Quantised Variational Auto-Encoder (VQ-VAE) loss function:
# ------------------------------------------------------------------------------ # -----80~100---- #


def vq_vae_loss(z_embed, z_quantised, beta=0.25):
    """

    Parameters
    ----------
    z_embed : Tensor
        Output of the enoder z_e prior to quantisation.
    z_quantised : Tensor 
        Nearest prototype vector e_q to z_e.

    Notes
    -----
    The parameter `z_quantised` should be the quantised vector prior to
    straight-through estimation; using the straight-through output z_q does not
    allow gradients to propagate to the protope vectors and the codebook (i.e.
    discrete symbols) will not be updated during training.

    The commitment cost (beta) should be set appropriately. It's useful to try a
    couple of values. It mostly depends on the scale of the reconstruction cost
    log p(x|z). If the reconstruction cost is 100x higher, the commitment_cost
    should also be multiplied with the same amount. (Source:
    https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb)
    """
    # Vector Quantisation dictionary/codebook learning loss
    vq_loss = tf.reduce_mean(  # mean of L2 norm squared
        tf.norm(tf.stop_gradient(z_embed) - z_quantised, axis=-1) ** 2)
    commitment_loss = tf.reduce_mean(  # mean of L2 norm squared
        tf.norm(z_embed - tf.stop_gradient(z_quantised), axis=-1) ** 2)
    vq_vae_loss = vq_loss + beta * commitment_loss
    return vq_vae_loss, commitment_loss
