import mxnet as mx
import re
import operator
import string
import dill as pickle
from unidecode import unidecode
from collections import defaultdict
from time import time
from collections import namedtuple

Dicts = namedtuple(
    'Dicts',
    ['src_vocab', 'inv_src_vocab', 'targ_vocab', 'inv_targ_vocab'])

Dataset = namedtuple(
    'Dataset',
    ['src_train_sent', 'src_valid_sent', 'src_vocab', 'inv_src_vocab', 
     'targ_train_sent', 'targ_valid_sent', 'targ_vocab', 'inv_targ_vocab'])


def invert_dict(d):
    return {v: k for k, v in d.iteritems()}

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

def top_words_train_valid(train_fname, valid_fname, top_k=50000, reserved_tokens=['<UNK>', '<PAD>', '<EOS>', '<GO>']):

    train_counts = word_count(preprocess_lines(train_fname))
    valid_counts = word_count(preprocess_lines(valid_fname))
    counts   = merge_counts(train_counts, valid_counts)

    del train_counts
    del valid_counts

    sorted_x = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = map(lambda x: x[0], sorted_x) #sorted_x[:top_k])
    start_idx = len(reserved_tokens)
    sorted_x = zip(sorted_x, range(start_idx, len(sorted_x) + start_idx))
    # value 0 is reserved for <UNK> or its semantic equivalent
    tops = defaultdict(lambda: 0, sorted_x)
    return tops


# Decode text as UTF-8
# Remove diacritical signs and convert to Latin alphabet
# Separate punctuation as separate "words"
#def tokenize_text(fname, vocab=None, invalid_label=0, start_label=1):
#    lines = unidecode(open(fname).read().decode('utf-8')).split('\n')
#    lines = map(lambda x: re.sub('[^A-Za-z0-9\s]', '', x).split(' '), lines)
#    
#    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)
#    return sentences, vocab

def tokenize_text(path, vocab, invalid_label=0, start_label=4):
    lines = preprocess_lines(path)
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)
    return sentences, vocab

def get_s2s_data(src_train_path, src_valid_path, targ_train_path, targ_valid_path,
         reserved_tokens=['<UNK>', '<PAD>', '<EOS>', '<GO>']):

        print("Creating joint source dictionary")
        src_dict = top_words_train_valid(src_train_path, src_valid_path)
       
        print("Tokenizing src_train_path") 
	src_train_sent, _ = tokenize_text(src_train_path, vocab=src_dict)
        print("Tokenizing targ_train_path")
        src_valid_sent, _ = tokenize_text(src_valid_path, vocab=src_dict)

        for i in range(len(reserved_tokens)):
            src_dict[reserved_tokens[i]] = i
            
	inv_src_dict = invert_dict(src_dict)

        print("Creating joint target dictionary")
        targ_dict = top_words_train_valid(targ_train_path, targ_valid_path)

        print("Tokenizing targ_train_path")
	targ_train_sent, _ = tokenize_text(targ_train_path, vocab=targ_dict)
        print("Tokenizing targ_valid_path")
        targ_valid_sent, _ = tokenize_text(targ_valid_path, vocab=targ_dict)

        for i in range(len(reserved_tokens)):
            targ_dict[reserved_tokens[i]] = i
  
        inv_targ_dict = invert_dict(targ_dict)

	return Dataset(
		src_train_sent=src_train_sent, src_valid_sent=src_valid_sent, src_vocab=src_dict, inv_src_vocab=inv_src_dict,
		targ_train_sent=targ_train_sent, targ_valid_sent=targ_valid_sent, targ_vocab=targ_dict, inv_targ_vocab=inv_targ_dict)

