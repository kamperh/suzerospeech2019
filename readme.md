Stellenbosch University ZeroSpeech 2019 System
==============================================

Overview
--------
The [ZeroSpeech challenges](https://zerospeech.com/) aim to answer the question
of how we can build speech processing systems directly from speech audio
without any labels. It has the dual motivation of understanding language
acquisition in humans and developing technology for extremely low-resource
languages. The task in [ZeroSpeech 2019](https://zerospeech.com/2019/) is "TTS
without T", i.e. text-to-speech without textual input. This is the repository
for `suzerospeech`, the Stellenbosch University ZeroSpeech 2019 system.


Disclaimer
----------
The code provided here is not pretty. But we believe that research should be
reproducible. We provide no guarantees with the code, but please let us know if
you have any problems, find bugs or have general comments.


Docker
------
This recipe comes with Dockerfiles which can be used to build images containing
all of the required dependencies. At the moment, we use a Dockerfile which is
different from the Dockerfile provided as part of the challenge. To use our
docker image you need to first:

- Install [Docker](https://docs.docker.com/install/) and follow the [post
  installation
  steps](https://docs.docker.com/install/linux/linux-postinstall/).
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

To build the docker image, run the following:

    cd docker
    docker build -f Dockerfile.tf-py36.cpu -t tf-py36 .
    cd ..

There is also a GPU version of the image. The rest of the steps in this recipe
can be run in a container in interactive mode. Start the docker image with the
required data directories mounted:

    docker run \
        -v ~/endgame/datasets/zerospeech2019/shared/databases/english/:/data/english \
        -v "$(pwd)":/home -it -p 8887:8887 tf-py36

To run on a GPU, `--runtime=nvidia` is additionally required.


Feature extraction
------------------
Move to `features/` and execute the steps in
[features/readme.md](features/readme.md).


Consistency rules for this repository
-------------------------------------
- British spelling for naming and documentation.


Dependencies
------------
- [Python 3](https://www.python.org/)
- [python_speech_features](https://github.com/jameslyons/python_speech_features)


License
-------
This code is distributed under the Creative Commons Attribution-ShareAlike
license ([CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)).
