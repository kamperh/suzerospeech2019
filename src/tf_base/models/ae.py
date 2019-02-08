"""TODO(rpeloff): module doc

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: February 2019
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf


from constants import TF_FLOAT_DTYPE


# ------------------------------------------------------------------------------ # -----80~100---- #
# Basic auto-encoder
# ------------------------------------------------------------------------------ # -----80~100---- #


def build_autoencoder(
        x_input, encoder_hidden_units, z_units, decoder_hidden_units,
        activation="relu"):
    """
    Build an autoencoder with the number of encoder and decoder units.

    The number of encoder and decoder units are given as lists. The middle
    (encoding/latent) layer has dimensionality `z_units`. This layer and the final
    layer are linear.

    TODO(rpeloff): function doc
    """

    # Encoder
    x_encoded = x_input
    for layer_index, units in enumerate(encoder_hidden_units):
        with tf.variable_scope("ae_encoder_{}".format(layer_index)):
            x_encoded = tf.keras.layers.Dense(
                units=units, activation=activation)(x_encoded)

    # Latent variable
    with tf.variable_scope("ae_latent"):
        x_encoded = tf.keras.layers.Dense(units=z_units)(x_encoded)
        z_latent = x_encoded

    # Decoder
    x_decoded = z_latent
    for layer_index, units in enumerate(decoder_hidden_units):
        with tf.variable_scope("ae_decoder_{}".format(layer_index)):
            if layer_index != len(decoder_hidden_units) - 1:
                x_decoded = tf.keras.layers.Dense(
                    units=units, activation=activation)(x_decoded)
            else:
                x_decoded = tf.keras.layers.Dense(units=units)(x_decoded)
    y_reconstruction = x_decoded

    return {"z": z_latent, "y": y_reconstruction}


# ------------------------------------------------------------------------------ # -----80~100---- #
# Variational auto-encoder
# ------------------------------------------------------------------------------ # -----80~100---- #


def build_vae(
        x_input, encoder_hidden_units, z_units, decoder_hidden_units,
        activation="relu"):
    """
    Build a VAE with the number of encoder and decoder units.

    The number of encoder and decoder units are given as lists. The middle
    (encoding/latent) layer has dimensionality `z_units`. The final
    layer is linear.

    TODO(rpeloff): function doc

    Return
    ------
    A dictionary with the mean `z_mean`, and log variance squared
    `z_log_sigma_sq` of the latent variable; the latent variable `z` itself
    (the output of the encoder); and the final output `y` of the network (the
    output of the decoder).
    """

    # Encoder
    x_encoded = x_input
    for layer_index, units in enumerate(encoder_hidden_units):
        with tf.variable_scope("vae_encoder_{}".format(layer_index)):
            x_encoded = tf.keras.layers.Dense(
                units=units, activation=activation)(x_encoded)

    # Latent variable
    with tf.variable_scope("vae_latent"):
        with tf.variable_scope("mean"):
            z_mean = tf.keras.layers.Dense(units=z_units)(x_encoded)
        with tf.variable_scope("log_sigma_sq"):
            z_log_sigma_sq = tf.keras.layers.Dense(units=z_units)(x_encoded)
        with tf.variable_scope("epsilon_noise"):
            eps = tf.random.normal(shape=(tf.shape(x_encoded)[0], z_units),
                                   mean=0, stddev=1, dtype=TF_FLOAT_DTYPE)
        # Reparametrisation trick
        z_latent = z_mean + tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)

    # Decoder
    x_decoded = z_latent
    for layer_index, units in enumerate(decoder_hidden_units):
        with tf.variable_scope("vae_decoder_{}".format(layer_index)):
            if layer_index != len(decoder_hidden_units) - 1:
                x_decoded = tf.keras.layers.Dense(
                    units=units, activation=activation)(x_decoded)
            else:
                x_decoded = tf.keras.layers.Dense(units=units)(x_decoded)
    y_reconstruction = x_decoded

    return {"z_mean": z_mean, "z_log_sigma_sq": z_log_sigma_sq,
            "z": z_latent, "y": y_reconstruction}


def vae_loss_gaussian(x, y, sigma_sq, z_mean, z_log_sigma_sq):
    """
    Use p(x|z) = Normal(x; f(z), sigma^2 I), with y = f(z) the decoder output.
    """
    
    # Gaussian reconstruction loss
    reconstruction_loss = 1./(2*sigma_sq) * tf.losses.mean_squared_error(x, y)
    
    # Regularisation loss
    regularisation_loss = -0.5*tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1
        )
    
    return reconstruction_loss + tf.reduce_mean(regularisation_loss)


def vae_loss_bernoulli(x, y, z_mean, z_log_sigma_sq):
    """
    Use a Bernoulli distribution for p(x|z), with the y = f(z) the mean.
    """
    
    # Bernoulli reconstruction loss
    reconstruction_loss = -tf.reduce_sum(
        x*tf.log(1e-10 + y) + (1 - x)*tf.log(1e-10 + 1 - y), 1
        )
    
    # Regularisation loss
    regularisation_loss = -0.5*tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1
        )
    
    return tf.reduce_mean(reconstruction_loss + regularisation_loss)
