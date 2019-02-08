ABX evaluation of features
==========================

The idea is to have a set of features in an .npz file, say `features_for_eval.npz`, which is in Herman's feature dictionary format (hknzp-format).
This `features_for_eval.npz` is the argument to the `run_abx_eval_remotely.sh` script which will execute the evaluation pipeline (i.e. perform the ABX evaluation and bitrate calculation) and output the results.
The ABX evaluation and bitrate calculation is already available in the baseline docker devkit.

Usage example
-------------

Run the following command in a terminal:

    bash run_abx_eval_remotely.sh features_for_eval.npz

It is not required that you be in any particular directory to run the script.
Just give the `path/to/the/features_for_eval.npz` file (absolute or relative).
The script executes a remote copy and an ssh command to the remote machine running the evaluation process.
This requires authentication for the remote logins.
You can either type in the password for the `suzero` user on the remote machine each time, or (*preferably*) you can add your ssh public key to the `~/.ssh/authorized_keys` file on the remote machine to avoid having to type the password.
If you are not familiar with key-based authentication, ask me (Ewald) and we will set it up.
Once the script has kicked off, output should look something like this:

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

    ABX average score: 19.824
    ABX calculated using dtw_cosine
    Read 48829 distinct symbols
    Total number of lines: 48829
    Total duration: 1708.692065
    Estimated bitrate (bits/s): 445.096980029

    ABX score is stored at /home/zs2019/evaluations/tmp.XZB7EL9WlT/tmp.XZB7EL9WlT.abx.txt
    Bitrate score is stored at /home/zs2019/evaluations/tmp.XZB7EL9WlT/tmp.XZB7EL9WlT.bitrate.txt


Other files in this evaluation directory
----------------------------------------
 
The `run_abx_eval.sh` script sits locally on the evaluation host machine and is called in the `run_abx_eval_remotely.sh` script through ssh.
The `suzero_evaluate.sh` script is a copy of the baseline docker `evaluate.sh` script, but slightly modified for our purposes. The only changes are to the directories where ABX processing is being done and results are output.
