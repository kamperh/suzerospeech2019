"""TODO(rpeloff): module doc

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: February 2019
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf


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
