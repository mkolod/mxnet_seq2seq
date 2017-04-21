#!/bin/bash
mkdir -p data/
cd data/ 

# check if exists and use cached version
#wget http://statmt.org/europarl/v7/es-en.tgz
#tar xvf es-en.tgz && rm es-en.tgz

TRAIN_SENT_CT=10000
VAL_SENT_CT=1000

head -n ${TRAIN_SENT_CT} europarl-v7.es-en.en > europarl-v7.es-en.en_train_small
head -n ${TRAIN_SENT_CT} europarl-v7.es-en.es > europarl-v7.es-en.es_train_small

tail -n ${VAL_SENT_CT} europarl-v7.es-en.en >  europarl-v7.es-en.en_valid_small
tail -n ${VAL_SENT_CT} europarl-v7.es-en.es > europarl-v7.es-en.es_valid_small
