#!/bin/bash

DATA_DIR_ROOT="./data"
DATA_DIR="${DATA_DIR_ROOT}/wmt15-de-en"

mkdir -p ${DATA_DIR}

pushd . > /dev/null

cd ${DATA_DIR_ROOT}

echo -e "\nDownloading dataset"

wget https://s3.amazonaws.com/opennmt-trainingdata/wmt15-de-en.tgz

echo -e "\nDecompressing dataset\n"

tar xvf wmt15-de-en.tgz

echo -e "\nConcatenating corpora"

cd wmt15-de-en

# concatenate corpora - note concatenation has to be in 
# the same order for both languages

# we will split this into training and validation sets
cat commoncrawl.de-en.de europarl-v7.de-en.de news-commentary-v10.de-en.de > train.de
# the test set already officially exists
mv newstest2013.de valid.de

wget -O test.de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de 

# do the same thing to English corpora

cat commoncrawl.de-en.en europarl-v7.de-en.en news-commentary-v10.de-en.en > train.en
mv newstest2013.en valid.en

wget -O test.en https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en

popd > /dev/null

echo -e "\nData download complete\n"
