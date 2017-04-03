#!/bin/bash
mkdir data/
cd data/ 
wget http://statmt.org/europarl/v7/es-en.tgz
tar xvf es-en.tgz && rm es-en.tgz

SMALL_SENT_CT=180000

head -n ${SMALL_SENT_CT} europarl-v7.es-en.en > europarl-v7.es-en.en_train_small
head -n ${SMALL_SENT_CT} europarl-v7.es-en.es > europarl-v7.es-en.es_train_small

tail -n +${SMALL_SENT_CT} europarl-v7.es-en.en | head -n ${SMALL_SENT_CT} > europarl-v7.es-en.en_valid_small
tail -n +${SMALL_SENT_CT} europarl-v7.es-en.es | head -n ${SMALL_SENT_CT} > europarl-v7.es-en.es_valid_small
