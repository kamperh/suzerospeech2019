#!bin/bash

function failure { [ ! -z "$1" ] && echo "Error: $1"; exit 1; }

function on_exit() {
    source deactivate
    if [ ! $KEEP_TEMP ]
    then
        rm -rf $TMP_DIR/working;
    fi
}


trap on_exit EXIT

set -e

# This is the script for ZS2019 Challenge's evaluation of submissions


if [ "$#" -ne 4 ]; then 
	echo "usage: bash evaluate_submission.sh <submission_zip> <language> <which_embedding> <dtw_cosine|dtw_kl|levenshtein>"
        echo ""
        echo "submission_zip: zip file containing submission"
        echo "language: language to evaluate ('english' or 'surprise')"
        echo "which_embedding: embedding to evaluate ('test', 'auxiliary_embedding1', or"
        echo "                 'auxiliary_embedding2')"
        echo "distance: distance function to use in ABX ('levenshtein' or 'dtw')"
        echo ""
        echo "Please see http://www.zerospeech.com/2019/ for complete documentation"
	exit
fi

EVAL_DIR="$HOME/system/eval"
SUBMISSION_ZIP=$1
LANG=$2
EMBEDDING=$3
DISTANCE=$4

# Checking parameter consistancy
if [ $LANG = 'surprise' ]; then
    failure "Surprise language is not available yet"
elif [ ! $LANG = "english" ]; then
    failure "$LANG is not a valid language, it should be english or surprise"
fi

if [ ! $EMBEDDING = 'test' ] && [ ! $EMBEDDING = 'auxiliary_embedding1' ] && [ ! $EMBEDDING = 'auxiliary_embedding2' ]; then
   failure "Embedding should correspond to test, auxiliary_embedding1 or auxiliary_embedding2"
fi 

if [ ! $DISTANCE = 'dtw_cosine' ] && [ ! $DISTANCE = 'levenshtein' ] && [ ! $DISTANCE = 'dtw_kl' ]; then
   failure "Distance should be either dtw_cosine, dtw_kl or levenshtein"
fi

TASK_ACROSS="$HOME/system/info_test/$LANG/by-context-across-speakers.abx"
BITRATE_FILELIST="$HOME/system/info_test/$LANG/bitrate_filelist.txt"

SUBMISSION_NAME=$(basename $SUBMISSION_ZIP | rev | cut -d "." -f 2- | rev)
ABX_SCORE_FILE=$SUBMISSION_NAME.abx.txt
BITRATE_SCORE_FILE=$SUBMISSION_NAME.bitrate.txt

#TMP_DIR=$(mktemp -d)
TMP_DIR="$(dirname ${SUBMISSION_ZIP})/working"
#echo "${TMP_DIR}"

# checking the zip file integrity
unzip -t "$SUBMISSION_ZIP" > /dev/null || failure "corrupted $SUBMISSION_ZIP"

# check file list
unzip "$SUBMISSION_ZIP" -d "$TMP_DIR" > /dev/null

if [ ! -d "$TMP_DIR/$LANG/$EMBEDDING" ]
then
	failure "Embedding directory $LANG/$EMBEDDING does not exist"
fi

source activate eval

echo "Evaluating ABX discriminability"
mkdir -p $TMP_DIR/abx_npz_files

# create npz files out of all onehot embeddings for ABX evaluation
python $EVAL_DIR/scripts/make_abx_files.py \
       $TMP_DIR/$LANG/$EMBEDDING $TMP_DIR/abx_npz_files || exit 1

# Create .features file
python $EVAL_DIR/ABXpy/ABXpy/misc/any2h5features.py \
       $TMP_DIR/abx_npz_files $TMP_DIR/features.features
# Computing distances
if [ $DISTANCE = "levenshtein" ]; then
    abx-distance $TMP_DIR/features.features $TASK_ACROSS \
        $TMP_DIR/distance_across -d $DISTANCE
elif [ $DISTANCE = "dtw_kl" ]; then
    abx-distance $TMP_DIR/features.features $TASK_ACROSS \
        $TMP_DIR/distance_across -d $DISTANCE
elif [ $DISTANCE = "dtw_cosine" ]; then
    abx-distance $TMP_DIR/features.features $TASK_ACROSS \
        $TMP_DIR/distance_across -n 1
else
    failure "$DISTANCE not implemented: choose 'dtw' or 'levenshtein'"
fi

# Calculating scores
abx-score $TASK_ACROSS $TMP_DIR/distance_across $TMP_DIR/score_across
# Collapsing results in readable format
abx-analyze $TMP_DIR/score_across $TASK_ACROSS $TMP_DIR/analyze_across
# Print average score
python $EVAL_DIR/scripts/meanABX.py $TMP_DIR/analyze_across across > ${TMP_DIR}/../$ABX_SCORE_FILE
echo ABX calculated using $DISTANCE >> ${TMP_DIR}/../$ABX_SCORE_FILE

echo "Evaluating bitrate"
python $EVAL_DIR/scripts/bitrate.py $TMP_DIR/$LANG/$EMBEDDING/ \
     $BITRATE_FILELIST > ${TMP_DIR}/../$BITRATE_SCORE_FILE || exit 1

echo ""
cat ${TMP_DIR}/../$ABX_SCORE_FILE
cat ${TMP_DIR}/../$BITRATE_SCORE_FILE

echo ""
echo "ABX score is stored at ${TMP_DIR%/working}/$ABX_SCORE_FILE"
echo "Bitrate score is stored at ${TMP_DIR%/working}/$BITRATE_SCORE_FILE"

#echo $TMP_DIR
KEEP_TEMP=true

