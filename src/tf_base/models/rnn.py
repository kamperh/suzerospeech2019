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
# Utillity functions:
# ------------------------------------------------------------------------------ # -----80~100---- #


def _get_rnn_cell(units, rnn_cell="lstm", rnn_cell_kwargs=None):
    """
    The `kwargs` parameters are passed directly to the constructor of the cell
    class, e.g. peephole connections can be used by adding `use_peepholes=True`
    when `rnn_type` is "lstm".

    TODO(rpeloff): function doc
    """
    cell_args = {}
    if rnn_cell_kwargs is not None:
        cell_args.update(rnn_cell_kwargs)
    if rnn_cell == "lstm":
        cell_args["state_is_tuple"] = True  # default LSTM parameters
        cell = tf.nn.rnn_cell.LSTMCell(units, **cell_args)
    elif rnn_cell == "gru":
        cell = tf.nn.rnn_cell.GRUCell(units, **cell_args)
    elif rnn_cell == "rnn":
        cell = tf.nn.rnn_cell.BasicRNNCell(units, **cell_args)
    else:
        raise ValueError("Got invalid RNN cell specifier: {}".format(rnn_cell))
    return cell


# ------------------------------------------------------------------------------ # -----80~100---- #
# Recurrent neural network (RNN):
# ------------------------------------------------------------------------------ # -----80~100---- #


def build_rnn(
        x_input, x_lengths, units, rnn_cell="lstm", rnn_cell_kwargs=None,
        keep_prob=1., scope=None):
    """
    Build a recurrent neural network (RNN) with architecture `rnn_cell`.

    The RNN is dynamic, with `x_lengths` giving the lengths as a Tensor with
    shape [n_data]. The input `x` should be padded to have shape [n_data,
    n_padded, d_in].

    TODO(rpeloff): function doc

    Parameters
    ----------
    rnn_type : str
        Can be "lstm", "gru" or "rnn".
    rnn_cell_kwargs : dict
        These are passed directly to the constructor of the cell class, e.g.
        peephole connections can be used by adding `use_peepholes=True` when
        `rnn_type` is "lstm".
    """
    # RNN cell
    cell = _get_rnn_cell(units, rnn_cell, rnn_cell_kwargs)

    # Dropout
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=1., output_keep_prob=keep_prob)

    # Dynamic RNN
    return tf.nn.dynamic_rnn(
        cell, x_input, sequence_length=x_lengths, dtype=tf.float32, scope=scope)


# ------------------------------------------------------------------------------ # -----80~100---- #
# Multi-layer RNN:
# ------------------------------------------------------------------------------ # -----80~100---- #

def build_multi_layer_rnn(
        x_input, x_lengths, layer_units, rnn_cell="lstm", rnn_cell_kwargs=None,
        keep_prob=1., scope=None):
    """
    Build a multi-layer recurrent neural network (RNN) with architecture `rnn_cell`.

    TODO(rpeloff): function doc
    """
    x_output = x_input
    with tf.variable_scope(scope):
        for layer_index, units in enumerate(layer_units):
            rnn_layer_scope = "rnn_layer_{}".format(layer_index)
            x_output, states = build_rnn(
                x_output, x_lengths, units, rnn_cell, rnn_cell_kwargs,
                keep_prob, rnn_layer_scope)
    return x_output, states
