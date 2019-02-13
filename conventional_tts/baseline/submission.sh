#!bin/bash


# Script that makes a submission zip from the trained baseline.
# It decodes all sentences from the test set and synthesize all files
# from the synthesis.txt file list.

function failure { [ ! -z "$1" ] && echo "Error: $1"; exit 1; }

function on_exit() {
    source deactivate
    cd $previous_dir
}

trap on_exit EXIT

set -e

if [ "$#" -ne 3 ]; then
	echo "usage: bash submission.sh <submission_dir> <zip|nozip>\\"
	echo "            [<english|english_small|surprise>]"
	echo ""
	echo "<submission_dir> is the directory in which the submission files will be saved"
	echo ""
	echo "If 'zip' a zipfile will be created next to the submission directory"
	echo ""
	echo "<english|english_small|surprise> is the name of the corpus on which the"
        echo "model was trained, which will also determine the test corpus."
	echo ""
        echo "Please see http://www.zerospeech.com/2019/ for complete documentation"
	exit
fi

SUBMISSION_DIR=$(realpath $1)
ZIP=$2
LANGUAGE=$3

if [ $LANGUAGE = 'both' ]; then

    bash $0 $1 nozip english
    bash $0 $1 zip surprise
    exit
fi
# Checking parameter consistancy
if  [ ! $LANGUAGE = "surprise" ] && [ ! $LANGUAGE = "english" ] && [ ! $LANGUAGE = "english_small" ]; then
    failure "$LANGUAGE is not a valid language, it should be surprise, english or english_small"
fi


BASELINE_DIR="$HOME/baseline"
TRAINING_DIR="$BASELINE_DIR/training"
BEER_DIR="$TRAINING_DIR/beer/recipes/zrc2019"

CORPUS_DIR="/shared/databases/$LANGUAGE/"
TEST_WAV_DIR="$CORPUS_DIR/test/"
OUTPUT_DIR="$SUBMISSION_DIR/$LANGUAGE/test/"
AUX_OUTPUT_DIR="$SUBMISSION_DIR/$LANGUAGE/auxiliary_embedding1/"

previous_dir=$(pwd)

rm -rf $OUTPUT_DIR $AUX_OUTPUT_DIR
mkdir -p $OUTPUT_DIR $AUX_OUTPUT_DIR

# Generation of the metadata file
<<<<<<< HEAD
cat > "$SUBMISSION_DIR/metadata.yaml" <<EOF
=======
cat > "$SUBMISSION_DIR/metadata" <<EOF
>>>>>>> a1296ea12468d7a6df96898aa45cfb1952ab6976
  author:
    ZeroSpeech organizers, CoML Team
  affiliation:
   PSL University, CNRS, ENS, EHESS, INRIA
  abx distance:
    dtw_cosine
  auxiliary1 description:
    Embeddings discovered by the Ondel algorithm, before conversion to
    one-hot vectors
  auxiliary2 description:
    not used
  open source:
    true
  system description:
    See https://zerospeech.com/2019/getting_started.html#baseline-system
  using parallel train:
    false
  using external data:
    false
EOF

# Decode new sentences

source activate beer
cd $BEER_DIR/
bash train_beer.sh $TEST_WAV_DIR $LANGUAGE novel 1 cleantrans $OUTPUT_DIR $AUX_OUTPUT_DIR || exit 1

# Resynthesize

source activate ossian
bash $BASELINE_DIR/synthesize.sh $OUTPUT_DIR $LANGUAGE

if [ $LANGUAGE = "english_small" ]; then
        cd $SUBMISSION_DIR
        mv english_small english
fi

if [ $ZIP = "zip" ]
then
	cd $SUBMISSION_DIR
	zip -r $SUBMISSION_DIR.zip .
fi

cd $previous_dir

echo "Submission on language $LANGUAGE is ready at $SUBMISSION_DIR "
