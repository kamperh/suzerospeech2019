Input Feature Extraction
========================

Overview
--------
These steps extract filterbank and MFCC features.


Docker
------
Start the docker image with the required directories mounted:

    docker run \
        -v ~/endgame/datasets/zerospeech2019/shared/databases/english/:/data/english \
        -v "$(pwd)":/home -it -p 8887:8887 tf-py36


Filterbanks
-----------


