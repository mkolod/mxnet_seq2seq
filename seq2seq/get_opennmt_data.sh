#!/bin/bash

DATA_DIR_ROOT="./data"
DATA_DIR="${DATA_DIR_ROOT}/wmt15-de-en"
IN_SRC_DATA_PATH="${DATA_DIR}/all.en"
IN_TARG_DATA_PATH="${DATA_DIR}/all.de"
OUT_SRC_TRAIN_PATH="${DATA_DIR}/train.en"
OUT_TARG_TRAIN_PATH="${DATA_DIR}/train.de"
OUT_SRC_VALID_PATH="${DATA_DIR}/valid.en"
OUT_TARG_VALID_PATH="${DATA_DIR}/valid.de"
VALIDATION_FRACTION=0.2
SHUFFLE_SEED=42

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
cat commoncrawl.de-en.de europarl-v7.de-en.de news-commentary-v10.de-en.de > all.de
# the test set already officially exists
mv newstest2013.de test.de

# do the same thing to English corpora

cat commoncrawl.de-en.en europarl-v7.de-en.en news-commentary-v10.de-en.en > all.en
mv newstest2013.en test.en

popd > /dev/null

echo -e "\nShuffling examples and splitting into training and validation samples"


# shuffle examples so validation data isn't completely from one
# corpus while training is from another

python split_train_valid.py \
  --in-src-data-path ${IN_SRC_DATA_PATH} \
  --in-targ-data-path ${IN_TARG_DATA_PATH} \
  --out-src-train-path ${OUT_SRC_TRAIN_PATH} \
  --out-targ-train-path ${OUT_TARG_TRAIN_PATH} \
  --out-src-valid-path ${OUT_SRC_VALID_PATH} \
  --out-targ-valid-path ${OUT_TARG_VALID_PATH} \
  --validation-fraction 0.2 \
  --shuffle-seed 42


