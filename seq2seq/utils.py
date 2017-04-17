import mxnet as mx
import os
import re
import operator
import string
from unidecode import unidecode
from collections import defaultdict
from time import time
from collections import namedtuple
from tqdm import tqdm

Dicts = namedtuple(
    'Dicts',
    ['src_vocab', 'inv_src_vocab', 'targ_vocab', 'inv_targ_vocab'])

Dataset = namedtuple(
    'Dataset',
    ['src_train_sent', 'src_valid_sent', 'src_vocab', 'inv_src_vocab', 
     'targ_train_sent', 'targ_valid_sent', 'targ_vocab', 'inv_targ_vocab'])

def invert_dict(d):
    return {v: k for k, v in d.iteritems()}

def encode_sentences(sentences, vocab, unk_id=1):
    res = []
    for sent in sentences:
        coded = []
        for word in sent:
            coded.append(vocab[word]) if word in vocab else coded.append(unk_id)
        res.append(coded)
    return res 

def linecount_wc(path):
    return int(os.popen('wc -l %s' % path).read().split()[0])

def preprocess_lines(fname):
    # I could read the file all at once, but then I couldn't
    # report progress for big files via tqdm to give an ETA
    fast_line_count = linecount_wc(fname)
    print("\nReading file: %s" % fname)
    with open(fname, 'r') as f:
        lines = []
        for line in tqdm(f, desc='Reading progress', total=fast_line_count):
            line = unidecode(line.decode('utf-8'))
            line = re.sub('\s+', ' ', re.sub('([' + string.punctuation + '])', r' \1 ', line)).split(' ')
            if line == []:
                continue
            lines.append(line)
    return lines


def word_count(lines, data_name=''):
    counts = defaultdict(long)
    for line in tqdm(lines, desc='word count (%s)' % data_name):
        for word in line:
            counts[word] += 1
    return counts

def merge_counts(dict1, dict2):
    return { k: dict1.get(k, 0) + dict2.get(k, 0) for k in tqdm(set(dict1) | set(dict2), desc='merge word counts') }

def top_words_train_valid(train_fname, valid_fname, top_k=50000, unk_key=1, reserved_tokens=['<PAD>', '<UNK>', '<EOS>', '<GO>']):

    counts = word_count(preprocess_lines(train_fname), data_name='train')

    print("Choosing top n words for the dictionary.")
    sorted_x = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = map(lambda x: x[0], sorted_x[:top_k]) # sorted_x
    start_idx = len(reserved_tokens)
    sorted_x = zip(sorted_x, range(start_idx, len(sorted_x) + start_idx))
    # value 0 is reserved for <UNK> or its semantic equivalent
    # tops = defaultdict(lambda: 0, sorted_x)
    tops = dict(sorted_x)

    for i in range(len(reserved_tokens)):
        tops[reserved_tokens[i]] = i

    inv_tops = invert_dict(tops)
    inv_tops[unk_key] = '<UNK>'
    return tops, inv_tops

def tokenize_text(path, vocab):
    lines = preprocess_lines(path)
    print("Encoding sentences")
    sentences = encode_sentences(lines, vocab)
    return sentences

def array_to_text(array, inv_vocab):
    sent = []
    for token in array:
        sent.append(inv_vocab[token])
    return " ".join(sent)

def get_s2s_data(src_train_path, src_valid_path, targ_train_path, targ_valid_path,
    reserved_tokens=['<UNK>', '<PAD>', '<EOS>', '<GO>']):

    print("Creating joint source dictionary")
    src_dict, inv_src_dict = top_words_train_valid(src_train_path, src_valid_path)
       
    print("Tokenizing src_train_path") 
    src_train_sent = tokenize_text(src_train_path, vocab=src_dict)
    print("Tokenizing targ_train_path")
    src_valid_sent = tokenize_text(src_valid_path, vocab=src_dict)

    print("Creating joint target dictionary")
    targ_dict, inv_targ_dict = top_words_train_valid(targ_train_path, targ_valid_path)

    print("Tokenizing targ_train_path")
    targ_train_sent = tokenize_text(targ_train_path, vocab=targ_dict)
    print("Tokenizing targ_valid_path")
    targ_valid_sent = tokenize_text(targ_valid_path, vocab=targ_dict)

    print("\nEncoded source language sentences:\n")
    for i in range(5):
        print(array_to_text(src_train_sent[i], inv_src_dict))            

    print("\nEncoded target language sentences:\n")
    for i in range(5):
        print(array_to_text(targ_train_sent[i], inv_targ_dict))            


    return Dataset(
        src_train_sent=src_train_sent, src_valid_sent=src_valid_sent, src_vocab=src_dict, inv_src_vocab=inv_src_dict,
        targ_train_sent=targ_train_sent, targ_valid_sent=targ_valid_sent, targ_vocab=targ_dict, inv_targ_vocab=inv_targ_dict)

