ABX evaluation of features
==========================

The idea is to have a set of feature in a .npz file, say `features_for_eval.npz`, which is in Herman's feature dictionary format (hknzp-format).
This `features_for_eval.npz` is the argument to the `run_abx_eval_remotely.sh` script which will execute the evaluation pipeline (i.e. perform the ABX evaluation and bitrate calculation) and output the results.
The ABX evaluation and bitrate calculation is already available in the baseline docker devkit.

Usage example
-------------

Run the following command in a terminal:

    bash run_abx_eval_remotely.sh features_for_eval.npz

Output should look something like:

    Copying features_for_eval.npz...
    features_for_eval.npz
        133,815,216 100%  326.30MB/s    0:00:00 (xfr#1, to-chk=0/1)
    Running /home/suzero/run_abx_eval.sh on feature file /home/suzero/tmp.XZB7EL9WlT/features_for_eval.npz
    100%|##########| 13994/13994 [00:14<00:00, 985.18it/s]
    ('Reading:', '/home/suzero/tmp.XZB7EL9WlT/features_for_eval.npz')
    ('Writing to:', '/home/suzero/tmp.XZB7EL9WlT/textfeatures')
    Copying baseline_submission.zip ...
    Unzipping baseline_submission.zip ...
    Copying text format feature files ...
    Cleaning temporary files...
    Copying submission zip to docker home dir ...
    Submission zip-file name: /home/zs2019/evaluations/tmp.XZB7EL9WlT/tmp.XZB7EL9WlT.zip
    Executing docker ...
    Evaluating ABX discriminability
    Job 1: computing distances for block 0 on 466
    Job 1: computing distances for block 1 on 466
    ...
    Job 1: computing distances for block 464 on 466
    Job 1: computing distances for block 465 on 466
    Evaluating bitrate

    ('ABX average score =', 19.823893905320777)
    ABX calculated using dtw_cosine
    Read 48829 distinct symbols
    Total number of lines: 48829
    Total duration: 1708.692065
    Estimated bitrate (bits/s): 445.096980029

    ABX score is stored at /home/zs2019/evaluations/tmp.XZB7EL9WlT/tmp.XZB7EL9WlT.abx.txt
    Bitrate score is stored at /home/zs2019/evaluations/tmp.XZB7EL9WlT/tmp.XZB7EL9WlT.bitrate.txt


The `run_abx_eval.sh` script sits locally on the evaluation host machine and is called in the `run_abx_eval_remotely.sh` script through ssh.
The `suzero_evaluate.sh` script is a copy of the baseline docker `evaluate.sh` script and slightly modified for our purposes. The only changes are to the directories where ABX work is being done and results are output.

