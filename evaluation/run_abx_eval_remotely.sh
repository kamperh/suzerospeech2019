#!/bin/bash

# This script will run the ABX evaluation on a remote machine which contains the zs2019 baseline docker image.
# The only argument to this script is the filename of the hknpz (.npz) feature file.
#
# Author: Ewald van der Westhuizen
# Date: Feb 2019
#

innpzfn=$1
innpzfn_bn=$(basename $1)

# Copy the .npz file into a tmp dir on the remote machine.
tmpdir=$(basename $(mktemp -u))
echo "Copying ${innpzfn}..."
ssh suzero@146.232.221.153 mkdir '~/feature_evals/'"${tmpdir}"
rsync -P ${innpzfn} suzero@146.232.221.153:~/feature_evals/${tmpdir}/${innpzfn_bn}

# kick off the evaluation process with an ssh command
ssh suzero@146.232.221.153 bash '~/suzerospeech2019/evaluation/run_abx_eval.sh' '~/feature_evals/'"${tmpdir}/${innpzfn_bn}"
