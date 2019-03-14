"""Define flags used throughout the `src` packages.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: February 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import namedtuple


import numpy as np
import tensorflow as tf


_flags = namedtuple(
    "FLAGS",
    [
        "tf_float_dtype", "tf_int_dtype", "np_float_dtype", "np_int_dtype",
        "quiet"])


FLAGS = _flags(  # default flags, update with FLAGS._replace(**kwargs)
    tf_float_dtype=tf.float32, tf_int_dtype=tf.int32,  # define data type used for TensorFlow float/integer tensors
    np_float_dtype=np.float32, np_int_dtype=np.int32,  # define data type used for NumPy float/integer ndarrays
    quiet=False
)
