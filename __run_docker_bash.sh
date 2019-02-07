#!/bin/bash

DOCKER_NAME=$1  # first arg optional container name
DOCKER_NAME=${DOCKER_NAME:-suzerospeech2019-test}

echo "Start Docker test container: ${DOCKER_NAME}"

docker run \
    --runtime=nvidia \
    -v $(pwd):/suzerospeech2019 \
    -u $(id -u):$(id -g) \
    -w /suzerospeech2019 \
    --rm \
    -it \
    --name ${DOCKER_NAME} \
    suzerospeech2019/tf-py36.gpu \
    bash