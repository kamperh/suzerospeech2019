"""Layers for building latent variables.

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
# Variational Auto-Encoder (VAE) latent variable:
# ------------------------------------------------------------------------------ # -----80~100---- #


class VAELatent(tf.keras.layers.Layer):

    def __init__(self, z_units):
        """Variational Auto-Encoder latent variable layer with z hidden units.

        Code adapted from https://blog.keras.io/building-autoencoders-in-keras.html.
        """
        super(VAELatent, self).__init__()
        self.z_units = z_units
        # Latent distribution parameter layers
        self.z_mean_dense = tf.keras.layers.Dense(units=z_units)
        self.z_log_sigma_sq_dense = tf.keras.layers.Dense(units=z_units)
        self.z_latent = tf.keras.layers.Lambda(self._sample)  # wrap sample func in Keras layer

    def build(self, input_shape):
        batch_size = input_shape[0]
        self.epsilon = tf.random.normal(
            shape=(batch_size, self.z_units), mean=0, stddev=1,
            dtype=TF_FLOAT_DTYPE)
        super(VAELatent, self).build(input_shape)  # call base `build` logic

    def _sample(self, inputs):
        """Sample with reparametrisation trick."""
        z_mean, z_log_sigma_sq = inputs
        return z_mean + tf.sqrt(tf.exp(z_log_sigma_sq)) * self.epsilon

    def call(self, inputs):
        # Sample latent distribution
        z_mean = self.z_mean_dense(inputs)
        z_log_sigma_sq = self.z_log_sigma_sq_dense(inputs)
        z_latent = self.z_latent([z_mean, z_log_sigma_sq])
        return z_latent, z_mean, z_log_sigma_sq


# ------------------------------------------------------------------------------ # -----80~100---- #
# Vector Quantized Variational Auto-Encoder (VQ-VAE) discrete latent variable:
# ------------------------------------------------------------------------------ # -----80~100---- #


class VQVAELatent(tf.keras.layers.Layer):

    def __init__(self, z_units, k_discrete, initializer=None, regularizer=None):
        """Vector Quantized Variational Auto-Encoder latent variable layer.

        Code adapted from https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
        and https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vq_vae.py
        and https://github.com/hiwonjoon/tf-vqvae/blob/master/model.py.
        """
        super(VQVAELatent, self).__init__()
        self.z_units = z_units
        self.k_discrete = k_discrete
        self.initializer = initializer
        self.regularizer = regularizer
        # Vector quantisation layers
        self.z_embed = tf.keras.layers.Dense(units=z_units)  # continuous embedding layer
        self.vector_quantise = tf.keras.layers.Lambda(  # wrap vector quantisation func in Keras layer
            lambda inputs: vector_quantisation(
                inputs, self.k_discrete, self.z_units,
                self.initializer, self.regularizer))
        
#         # Quantisation codebook containing discrete symbols (or embeddings)
#         self.embed_codebook = tf.keras.layers.Embedding(
#             self.k_discrete, self.z_unit)
#         self.embed_codebook = tf.get_variable(  # TODO: switch to tf.Variable for reuse; currently cannot create two VQ layers
#             "embedding_codebook", [self.k_discrete, self.z_units],
#             dtype=FLAGS.tf_float_dtype, initializer=initializer,
#             regularizer=regularizer)

    def call(self, inputs):
        # Compute continuous embeddings and quantise to discrete symbols
        z_continuous = self.z_embed(inputs)
        k_nearest, z_quantised, z_one_hot, embed_codebook = self.vector_quantise(z_continuous)
        # There is no real gradient for the vector quantise operation, so we
        # approximate the gradient similar to a straight-through estimator
        # and copy gradients from the decoder input to the encoder output:
        # re-compute the quantised vectors as sum of the the continuous vectors
        # and their difference with the quantised vectors, stopping the gradient
        # of the quantisation operation such that the partial derivative w.r.t.
        # z_embed is 1.
        z_straight_through = z_continuous + tf.stop_gradient(z_quantised - z_continuous)
#         z_quantised = z_straight_through
        # Compute perplexity which is useful to track during training as it
        # indicates how many codes (i.e. symbols) are "active" on average
        avg_probs = tf.reduce_mean(z_one_hot, 0)
        perplexity = tf.exp(
            -1. * tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))
        
        self.perplexity = perplexity
        self.embed_codebook = embed_codebook
        self.k_nearest = k_nearest
        return z_straight_through, z_quantised, z_continuous, z_one_hot
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.z_units
        return tf.TensorShape(shape)


def vector_quantisation(inputs, k_discrete, z_units, initializer, regularizer):
        """VQ layer with k discrete symbols, each with dimensionality z.

        Quantise each continuous input vector to a discrete symbol.
        """
        embed_codebook = tf.get_variable(  # TODO: switch to tf.Variable for reuse; currently cannot create two VQ layers
            "embedding_codebook", [k_discrete, z_units],
            dtype=FLAGS.tf_float_dtype, initializer=initializer,
            regularizer=regularizer)
        inputs_broadcast = tf.expand_dims(inputs, axis=-2)
        distances = tf.norm(inputs_broadcast - embed_codebook, axis=-1)  # for each input vector, compute L2 distances to discrete symbols
        k_nearest = tf.argmin(distances, axis=-1)  # use 1-nearest neighbour to get index of discrete symbol for each continuous vector
        inputs_quantised = tf.gather(embed_codebook, k_nearest)
        one_hot_encodings = tf.one_hot(k_nearest, k_discrete)
        return k_nearest, inputs_quantised, one_hot_encodings, embed_codebook


# ------------------------------------------------------------------------------ # -----80~100---- #
# Categorical Variational Auto-Encoder (CatVAE) categorical latent variable:
# ------------------------------------------------------------------------------ # -----80~100---- #


class CatVAELatent(tf.keras.layers.Layer):

    def __init__(self, n_distributions, k_categories, tau,
                 straight_through=True):
        """Categorical Variational Auto-Encoder latent variable layer.
        
        Code adapted from https://github.com/ericjang/gumbel-softmax/.

        ... n distributions each with k components.

        Return
        ------
        The log of the categorical distribution based directly on the logits
        log_logits_categorical, the one-hot latent variable z_latent, and the
        temperate variable tau.
        """
        super(CatVAELatent, self).__init__()
        self.n_distributions = n_distributions
        self.k_categories = k_categories
        self.tau = tau
        self.straight_through = straight_through
        # Categorical VAE layers
        self.logits = tf.keras.layers.Dense(
            units=self.n_distributions*self.k_categories)
        self.gumbel_softmax = tf.keras.layers.Lambda(
            lambda inputs: gumbel_softmax(
                inputs, self.k_categories, self.tau, self.straight_through))  # wrap gumbel softmax func in Keras layer

    def call(self, inputs):
        logits = self.logits(inputs)
        logits = tf.reshape(logits, [-1, self.n_distributions, self.k_categories])
        softmax_logits = tf.nn.softmax(logits)  # class probabilities before categorical approximation
        log_logits_categorical = tf.log(softmax_logits + 1e-20)
        z_cat = self.gumbel_softmax(logits)
        # z_cat = tf.reshape(z_cat, [-1, self.n_distributions, self.k_categories])
        z_cat = tf.reshape(z_cat, [-1, self.n_distributions*self.k_categories])

        return z_cat, softmax_logits, log_logits_categorical


def gumbel_softmax(inputs, k_categories, temperature, straight_through):
    """Gumbel-Softmax layer to approximate Categorical one-hot distribution.

    Used as a reparameterized continuous approximation to the Categorical
    one-hot distribution. This was concurrently introduced as the Gumbel-Softmax
    [Jang et al., 2016] and Concrete [Maddison et al., 2016] distributions.

    Parameter straight_through specifies whether to use Straight-Through Gumbel
    Estimator to discretise z_categorical in the forward pass using argmax,
    while using the continuous approximation in the backward pass.
    """
    y_cat = gumbel_softmax_sample(inputs, temperature)  # approximation of categorical variable z
    if straight_through:  # discretise continuous categorical approximation to hard discrete one-hot
        y_hard = tf.cast(
            tf.one_hot(tf.argmax(y_cat, axis=-1), k_categories),
            y_cat.dtype)
        y_cat = y_cat + tf.stop_gradient(y_hard - y_cat)  # straight-through estimator
    return y_cat


def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1) distribution."""
    uniform_sample = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(uniform_sample + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 
    """Draw a sample from the Gumbel-Softmax distribution.

    See also:
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedOneHotCategorical.
    """
    y_gumbel_logits = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y_gumbel_logits/temperature)
