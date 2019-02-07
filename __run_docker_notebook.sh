#!/bin/bash

DOCKER_NAME=$1  # 1st arg optional container name
DOCKER_NAME=${DOCKER_NAME:-suzerospeech2019-notebook}

DATA_LOCATION=$2  # 2nd arg optional data mount location

echo ""
echo "Start Docker test container: ${DOCKER_NAME}"
echo "Speech data loaded to '/suzerospeech2019/features' from: ${DATA_LOCATION}"
echo ""

docker run \
    --runtime=nvidia \
    -v $(pwd):/suzerospeech2019 \
    -v ${DATA_LOCATION}:/suzerospeech2019/features \
    -u $(id -u):$(id -g) \
    -w /suzerospeech2019 \
    --rm \
    -it \
    -e JUPYTER_DATA_DIR=/suzerospeech2019/.jupyter \
    -p 8889:8889 \
    --name ${DOCKER_NAME} \
    suzerospeech2019/tf-py36.gpu \
    jupyter notebook --no-browser --ip=0.0.0.0 --port=8889 --NotebookApp.token='' --notebook-dir='/suzerospeech2019'