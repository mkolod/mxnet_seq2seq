import numpy as np
import mxnet as mx
from collections import namedtuple, Counter
from unidecode import unidecode
from itertools import groupby
from mxnet.io import DataBatch, DataIter
from random import shuffle
from mxnet import ndarray
from time import time
import uuid
import os
from tqdm import tqdm

from utils import tokenize_text, invert_dict, get_s2s_data, Dataset

from seq2seq_iterator import Seq2SeqIter

import operator
import dill as pickle
import re
import warnings

if __name__ == '__main__':

    # Get rid of annoying Python deprecation warnings from built-in JSON encoder
    warnings.filterwarnings("ignore", category=DeprecationWarning)   
 
    start = time()

    dataset = get_s2s_data(
        src_train_path= './data/europarl-v7.es-en.en_train_small',
        src_valid_path= './data/europarl-v7.es-en.en_valid_small',
        targ_train_path= './data/europarl-v7.es-en.es_train_small',
        targ_valid_path= './data/europarl-v7.es-en.en_valid_small'
    )
   
    preproc_duration = time() - start
    print("\nPreprocessing data took %.4f seconds\n" % preproc_duration)

    min_len = 5

    max_len = 50
    increment = 5

    all_pairs = [(i, j) for i in xrange(
            min_len,max_len+increment,increment
        ) for j in xrange(
            min_len,max_len+increment,increment
        )]


    print("Constructing train iterator")
    train_iter = Seq2SeqIter(dataset.src_train_sent, dataset.targ_train_sent, dataset.src_vocab, dataset.inv_src_vocab,
                     dataset.targ_vocab, dataset.inv_targ_vocab, layout='TN', batch_size=32, buckets=all_pairs)

    train_iter.bucketize()

    with open('./data/train_iterator.pkl', 'wb') as f:
        pickle.dump(train_iter, f, pickle.HIGHEST_PROTOCOL)


    print("Constructing valid iterator")
    valid_iter = Seq2SeqIter(dataset.src_valid_sent, dataset.targ_valid_sent, dataset.src_vocab, dataset.inv_src_vocab,
                     dataset.targ_vocab, dataset.inv_targ_vocab, layout='TN', batch_size=32, buckets=all_pairs)

    valid_iter.bucketize()

    with open('./data/valid_iterator.pkl', 'wb') as f:
        pickle.dump(train_iter, f, pickle.HIGHEST_PROTOCOL)


