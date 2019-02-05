ABX evaluation of features
==========================

The idea is to have a set of feature in a .npz file, say `features_for_eval.npz`, which is in Herman's feature dictionary format.
This `features_for_eval.npz` can then be input to a pipeline (a simple to call script) which will perform the ABX evaluation and bitrate calculation and output the results.
The ABX evaluation and bitrate calculation is already available in the baseline docker devkit.

* Either use the existing baseline docker process which will be streamlined to take in our `features_for_eval.npz` format, or
* Create a clone of how the evaluation process is done in the baseline docker so that each person can have their own instance the evaluation process.

