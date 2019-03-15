"""Functions for logging.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: March 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging


from flags import FLAGS


def set_logger(log_path, log_fn="train.log"):
    """Set logging to output logs in console and file stored at `log_path`.

    Useful to replace print("...") statements with logging.info("...") in order
    to store log information for later viewing, as well as display it
    in the console.

    Parameters
    ----------
    log_path : str
        Path to store log file.
    log_fn : str, optional
        Filename of log file (the default is "train.log").

    Notes
    -----
    Based on logging function from:
    - https://cs230-stanford.github.io/logging-hyperparams.html.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to file "log_path/log_fn"
        log_file = os.path.join(log_path, log_fn)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console (allows us to replace `print` with `logging.info`)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

        if not FLAGS.quiet:
            logging.info("Logging to file {}".format(log_file))
