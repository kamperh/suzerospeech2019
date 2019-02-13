#!/bin/bash

# Script that trains a voice based on a directory of decoded sentences in onehot vector format

function failure { [ ! -z "$1" ] && echo "Error: $1"; exit 1; }

function on_exit() {
    source deactivate
    if [ ! $KEEP_TEMP ]
    then
        rm -rf $TMP_OSSIAN;
    fi
    cd $previous_dir
}

trap on_exit EXIT

set -e

if [ "$#" -ne 3 ]; then 
        echo "Usage: bash train_synth.sh <decoding_dir> <language> <clean|noclean>"
	echo ""
        echo "Please see http://www.zerospeech.com/2019/ for complete documentation"
	exit
fi

TRAINING_DIR="$HOME/baseline/training"
OSSIAN_DIR="$TRAINING_DIR/ossian"

DECODING_DIR="$1"
LANGUAGE="$2"
MODE="default"
WAV_DIR="/shared/databases/$LANGUAGE/train/voice"

# Checking parameter consistancy
if [ ! $LANGUAGE = "surprise" ] && [ ! $LANGUAGE = "english" ] && [ ! $LANGUAGE = "english_small" ]; then
    failure "$LANGUAGE is not a valid language, it should be surprise, english or english_small"
fi

export TMP=$(mktemp -d)
TMP_CORPUS="$TMP/corpus"
TMP_TRAIN="$TMP/train"

mkdir -p $TMP_CORPUS $TMP_TRAIN

# Conversion of one-hot embeddings to Ossian input format

echo "Converting one-hot embeddings to Ossian input format..."
python $TRAINING_DIR/scripts/onehot_to_ossian_format.py \
        $DECODING_DIR \
        $TMP_TRAIN \
        $TMP/all_files_as_text.txt
echo "Done."

mkdir -p $TMP_CORPUS/$LANGUAGE/speakers/V001/wav
mkdir -p $TMP_CORPUS/$LANGUAGE/speakers/V001/txt
cp $WAV_DIR/V001* $TMP_CORPUS/$LANGUAGE/speakers/V001/wav/
cp $TMP_TRAIN/V001* $TMP_CORPUS/$LANGUAGE/speakers/V001/txt/

if ls $WAV_DIR/V002* 1> /dev/null 2>&1
then
        mkdir -p $TMP_CORPUS/$LANGUAGE/speakers/V002/wav
        mkdir -p $TMP_CORPUS/$LANGUAGE/speakers/V002/txt
        cp $WAV_DIR/V002* $TMP_CORPUS/$LANGUAGE/speakers/V002/wav/
        cp $TMP_TRAIN/V002* $TMP_CORPUS/$LANGUAGE/speakers/V002/txt/
fi

mkdir -p $TMP_CORPUS/$LANGUAGE/text_corpora
cp $TMP/all_files_as_text.txt $TMP_CORPUS/$LANGUAGE/text_corpora/text.txt
python $TRAINING_DIR/scripts/check_corpus.py $TMP_CORPUS

# Train voices

source activate ossian
#MODE="gpu"
cd $OSSIAN_DIR
if [ "$3" = "clean" ]
then
	rm -rf corpus/$LANGUAGE train/$LANGUAGE voices/$LANGUAGE
fi
echo bash train_ossian.sh $TMP_CORPUS default $LANGUAGE V001
#bash train_ossian.sh $TMP_CORPUS $MODE $LANGUAGE V001 || exit 1
echo "Running train_ossian.sh. Follow along in train_ossian_V001.log"
bash train_ossian.sh $TMP_CORPUS $MODE $LANGUAGE V001 &> train_ossian_V001.log &
#if ls $WAV_DIR/V002* 1> /dev/null 2>&1
#then
#	bash train_ossian.sh $TMP_CORPUS $MODE $LANGUAGE V002 || exit 1
#fi
echo "Running train_ossian.sh. Follow along in train_ossian_V002.log"
bash train_ossian.sh $TMP_CORPUS $MODE $LANGUAGE V002 &> train_ossian_V002.log &

wait

