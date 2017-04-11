import mxnet as mx
import re
import operator
import string
from unidecode import unidecode
from collections import defaultdict
from time import time
from common import Dicts, invert_dict

# import hickle

#try:
#   import cPickle as pickle
#except:
#   import pickle

import dill as pickle

def preprocess_lines(fname):
    lines = unidecode(open(fname).read().decode('utf-8')).split('\n')
    lines = map(lambda x: filter(lambda y: y != '', re.sub('\s+', ' ', re.sub('([' + string.punctuation + '])', r' \1 ', x) ).split(' ')), lines)
    lines = filter(lambda x: x != [], lines)
    return lines

def word_count(lines, top_k=50000):
    counts = defaultdict(long)
    for line in lines:
        for word in line:
            counts[word] += 1
    return counts

def merge_counts(dict1, dict2):
    return { k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2) }

def top_words_train_valid(train_file, valid_file, top_k=50000, reserved_tokens=['<UNK>', '<PAD>', '<EOS>', '<GO>']):

    train_counts = word_count(preprocess_lines(train_file))
    valid_counts = word_count(preprocess_lines(valid_file))
    counts   = merge_counts(train_counts, valid_counts)

    del train_counts
    del valid_counts    
    
    sorted_x = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = map(lambda x: x[0], sorted_x[:top_k])
    start_idx = len(reserved_tokens)
    sorted_x = zip(sorted_x, range(start_idx, len(sorted_x) + start_idx))
    # value 0 is reserved for <UNK> or its semantic equivalent
    tops = defaultdict(lambda: 0, sorted_x)
    return tops

if __name__ == '__main__':
    start = time()
    src_train_path = './data/europarl-v7.es-en.en'
    src_valid_path = './data/europarl-v7.es-en.en_valid_small'
    src_vocab = top_words_train_valid(src_train_path, src_valid_path)
    inv_src_vocab = invert_dict(src_vocab)

    targ_train_path = './data/europarl-v7.es-en.es'
    targ_valid_path = './data/europarl-v7.es-en.es_valid_small'
    targ_vocab = top_words_train_valid(targ_train_path, targ_valid_path)
    inv_targ_vocab = invert_dict(targ_vocab)

    end = time()
    duration = end - start
    print("\nTime: %.2f seconds\n" % (duration / 2))

    dicts = Dicts(src_vocab=src_vocab, inv_src_vocab=inv_src_vocab,
                  targ_vocab=targ_vocab, inv_targ_vocab=inv_targ_vocab)

    dict_path = 'dicts.pkl'
    print(dicts)
    with open(dict_path, 'wb') as f:
        pickle.dump(dicts, f)
    del dicts
    del src_vocab
    del inv_src_vocab
    del targ_vocab
    del inv_targ_vocab

    start = time()
    with open(dict_path, 'rb') as f:
        dicts = pickle.load(f)    
    end = time()
    duration = end - start

    print("Dict pickle loading time: %.3f seconds" % duration)

    src_vocab = dicts.src_vocab
    inv_src_vocab = dicts.inv_src_vocab
    targ_vocab = dicts.targ_vocab
    inv_targ_vocab = dicts.inv_targ_vocab

    end = time()
    duration = end - start
    print("\nTime: %.2f seconds\n" % (duration / 2))
    print(src_vocab['the'])
    print(src_vocab['European'])
    print(src_vocab['zyzzyva'])
    print(targ_vocab['nosotros'])
    print(targ_vocab['Alicante'])
    print(targ_vocab['ewjfkfjekrlfe'])
