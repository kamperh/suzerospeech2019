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
# Basic Auto-Encoder
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

    # Latent layer
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
# Variational Auto-Encoder (VAE)
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

    # Latent layer
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


# ------------------------------------------------------------------------------ # -----80~100---- #
# Vector Quantized Variational Auto-Encoder (VQ-VAE)
# ------------------------------------------------------------------------------ # -----80~100---- #


def build_vq_vae(
        x_input, encoder_hidden_units, z_units, decoder_hidden_units, K,
        activation="relu"):
    """
    Build a VQ-VAE with the number of encoder/decoder units and K categories.

    The number of encoder and decoder units are given as lists. The middle
    (encoding/latent) layer has dimensionality `z_units`. The final
    layer is linear.

    TODO(rpeloff): function doc

    Return
    ------
    A dictionary with the embeddings, the embedded output `z_e` from the
    encoder, the quantised output `z_q` from the encoder, and the final output
    `y` from the decoder.
    """

    # Encoder
    x_encoded = x_input
    for layer_index, units in enumerate(encoder_hidden_units):
        with tf.variable_scope("vq-vae_encoder_{}".format(layer_index)):
            x_encoded = tf.keras.layers.Dense(
                units=units, activation=activation)(x_encoded)

    # Latent layer
    with tf.variable_scope("vq-vae_latent"):
        with tf.variable_scope("embedding"):
            z_embed = tf.keras.layers.Dense(units=z_units)(x_encoded)
        with tf.variable_scope("vq-vae_quantise"):  # quantisation
            vq = build_vector_quantisation(z_embed, K, z_units)
            q_embeds = vq["q_embeds"]
            k_embed = vq["k_embed"]
            z_quantised = vq["z_quantised"]

    # Decoder
    x_decoded = z_quantised
    for layer_index, units in enumerate(decoder_hidden_units):
        with tf.variable_scope("vq-vae_decoder_{}".format(layer_index)):
            if layer_index != len(decoder_hidden_units) - 1:
                x_decoded = tf.keras.layers.Dense(
                    units=units, activation=activation)(x_decoded)
            else:
                x_decoded = tf.keras.layers.Dense(units=units)(x_decoded)
    y_reconstruction = x_decoded

    return {"q_embeds": q_embeds, "k_q_indices": k_embed, "z_embed": z_embed,
            "z_quantised": z_quantised, "y": y_reconstruction}


def build_vector_quantisation(x, K, D):
    """
    A vector quantisation layer with `K` components of dimensionality `D`.

    See https://github.com/hiwonjoon/tf-vqvae/blob/master/model.py.
    """

    # Embeddings
    q_embeds = tf.get_variable(
        "quantisation_embeds", [K, D], dtype=TF_FLOAT_DTYPE,
        initializer=tf.contrib.layers.xavier_initializer())

    # Quantise inputs
    embeds_tiled = tf.reshape(q_embeds, [1, K, D])  # [batch_size, K, D]
    x_tiled = tf.tile(tf.expand_dims(x, -2), [1, K, 1])
    dist = tf.norm(x_tiled - embeds_tiled, axis=-1)
    k = tf.argmin(dist, axis=-1)
    z_q = tf.gather(q_embeds, k)

    return {"q_embeds": q_embeds, "k_embed": k, "z_quantised": z_q}


def vq_vae_loss(
        x_input, z_embed, z_quantised, q_embeds, y_output,
        learning_rate=0.001, beta=0.25, sigma_sq=0.5):
    """
    Return the different loss components and the training operation.
    
    If `sigma_sq` is "bernoulli", then p(x|z) is assumed to be a Bernoulli
    distribution.
    """
    # Losses
    if sigma_sq == "bernoulli":
        recon_loss = tf.reduce_mean(
            -1*tf.reduce_sum(
                x_input*tf.log(1e-10 + y_output) + (1 - x_input)*tf.log(1e-10 + 1 - y_output),
                axis=1))
    else:
        recon_loss = 1./(2*sigma_sq)*tf.losses.mean_squared_error(x_input,
                                                                  y_output)
    vq_loss = tf.reduce_mean(
        tf.norm(tf.stop_gradient(z_embed) - z_quantised, axis=-1)**2
        )
    commit_loss = tf.reduce_mean(
        tf.norm(z_embed - tf.stop_gradient(z_quantised), axis=-1)**2
        )
    loss = recon_loss + vq_loss + beta*commit_loss

    # Backpropagation: Copy gradients for quantisation
    with tf.variable_scope("backward"):

        # Decoder gradients
        decoder_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "vqvae_dec"
            )
        decoder_vars.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn_decoder")
            )
        # print(decoder_vars)
        decoder_grads = list(
            zip(tf.gradients(loss, decoder_vars), decoder_vars)
            )

        # Encoder gradients
        encoder_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "vqvae_enc"
            )
        encoder_vars.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn_encoder")
            )
        # print(encoder_vars)
        z_q_grad = tf.gradients(recon_loss, z_quantised)
        encoder_grads = [
            (tf.gradients(z_embed, var, z_q_grad)[0] +
            beta*tf.gradients(commit_loss, var)[0], var) for var in
            encoder_vars
            ]

        # Quantisation gradients
        embeds_grads = list(zip(tf.gradients(vq_loss, q_embeds), [q_embeds]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(
            decoder_grads + encoder_grads + embeds_grads
            )

    return loss, recon_loss, vq_loss, commit_loss, train_op


# ------------------------------------------------------------------------------ # -----80~100---- #
# Categorical Variational Auto-Encoder (CatVAE)
# ------------------------------------------------------------------------------ # -----80~100---- #
# Code adapted from https://github.com/ericjang/gumbel-softmax/


def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1) distribution."""
    uniform_sample = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(uniform_sample + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 
    """Draw a sample from the Gumbel-Softmax distribution."""
    y_gumbel_logits = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y_gumbel_logits/temperature)


def gumbel_softmax(logits, temperature, hard_discrete=None):
    """Sample from the Gumbel-Softmax distribution and optionally hard discretise.
    """
    y = gumbel_softmax_sample(logits, temperature)  # approximation of categorical variable z
    if hard_discrete is not None:  # quantise approximate categorical distribution to hard discrete one-hot (only during testing/inference)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.cond(hard_discrete,
                    true_fn=lambda: tf.stop_gradient(y_hard - y) + y,
                    false_fn=lambda: y)
    return y


def build_cat_vae(
        x_input, encoder_hidden_units, decoder_hidden_units, K, N,
        activation="relu"):
    """
    Build a categorical VAE with `N` distributions each with `K` components.
    Parameters
    ----------
    The number of encoder and decoder units are given as lists.
    Return
    ------
    A dictionary with the log of the categorical distribution based directly on
    the logits `log_logits_categorical`, the one-hot latent variable output `z`
    from the encoder, the final output `y` from the decoder, and the temperate
    variable `tau`.
    """

    tau = tf.placeholder(TF_FLOAT_DTYPE, [])
    hard_discrete_handle = tf.placeholder_with_default(False, [], "hard_discrete_handle")

    # Encoder
    x_encoded = x_input
    for layer_index, units in enumerate(encoder_hidden_units):
        with tf.variable_scope("cat_vae_encoder_{}".format(layer_index)):
            x_encoded = tf.keras.layers.Dense(
                units=units, activation=activation)(x_encoded)

    # Latent variable
    with tf.variable_scope("cat_vae_latent"):
        logits = tf.keras.layers.Dense(units=K*N)(x_encoded)  # the log[pi_i]'s
        softmax_logits = tf.nn.softmax(logits)  # class probabilities before categorical approximation
        log_logits_categorical = tf.log(softmax_logits + 1e-20)
        z_cat = tf.reshape(
            gumbel_softmax(logits, tau, hard_discrete=hard_discrete_handle),
            [-1, N, K])

    # Decoder
    x_decoded = tf.reshape(z_cat, [-1, N*K])
    for layer_index, units in enumerate(decoder_hidden_units):
        with tf.variable_scope("cat_vae_decoder_{}".format(layer_index)):
            if layer_index != len(decoder_hidden_units) - 1:
                x_decoded = tf.keras.layers.Dense(
                    units=units, activation=activation)(x_decoded)
            else:
                x_decoded = tf.keras.layers.Dense(units=units)(x_decoded)
    y_reconstruction = x_decoded

    return {
        "softmax_logits": softmax_logits,
        "log_logits_categorical": log_logits_categorical,
        "z_categorical": z_cat,
        "y": y_reconstruction,
        "tau": tau,
        "hard_discrete_handle": hard_discrete_handle}
