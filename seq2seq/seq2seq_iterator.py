import numpy as np
import mxnet as mx
from collections import namedtuple, Counter
from unidecode import unidecode
from itertools import groupby
from mxnet.io import DataBatch, DataIter
from random import shuffle
from mxnet import ndarray

import operator
import pickle
import re
import warnings

# Decode text as UTF-8
# Remove diacritical signs and convert to Latin alphabet
# Separate punctuation as separate "words"
def tokenize_text(fname, vocab=None, invalid_label=0, start_label=1, sep_punctuation=True):
    lines = unidecode(open(fname).read().decode('utf-8')).split('\n')
    lines = map(lambda x: re.findall(r"\w+|[^\w\s]", x, re.UNICODE), lines)    
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)
    return sentences, vocab

Dataset = namedtuple(
    'Dataset', 
    ['src_sent', 'src_vocab', 'inv_src_vocab', 'targ_sent', 'targ_vocab', 'inv_targ_vocab'])

def invert_dict(d):
    return {v: k for k, v in d.iteritems()}

def get_s2s_data(src_path, targ_path, start_label=1, invalid_label=0, pad_symbol='<PAD>'):
    src_sent, src_vocab = tokenize_text(src_path, start_label=start_label,
                                invalid_label=invalid_label)
    
    src_vocab[pad_symbol] = invalid_label
    inv_src_vocab = invert_dict(src_vocab)

    targ_sent, targ_vocab = tokenize_text(targ_path, start_label=start_label, #new_start+1,
                                          invalid_label=invalid_label)
    
    targ_vocab[pad_symbol] = invalid_label
    inv_targ_vocab = invert_dict(targ_vocab)
    
    return Dataset(
        src_sent=src_sent, src_vocab=src_vocab, inv_src_vocab=inv_src_vocab,
        targ_sent=targ_sent, targ_vocab=targ_vocab, inv_targ_vocab=inv_targ_vocab)

class Seq2SeqIter(DataIter):

    class TwoDBisect:
        def __init__(self, buckets):
            self.buckets = sorted(buckets, key=operator.itemgetter(0, 1))
            self.x, self.y = zip(*buckets)
            self.x, self.y = np.array(list(self.x)), np.array(list(self.y))

        def twod_bisect(self, source, target):    
            offset1 = np.searchsorted(self.x, len(source), side='left')
            offset2 = np.where(self.y[offset1:] >= len(target))[0]        
            return self.buckets[offset1 + offset2[0]]     
    
    def __init__(
        self, dataset, buckets=None, batch_size=32, max_sent_len=None,
        data_name='data', label_name='softmax_label', dtype=np.int32, layout='NTC'):
        self.major_axis = layout.find('N')
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.layout = layout
        self.batch_size = batch_size
        self.src_sent = dataset.src_sent
        self.targ_sent = dataset.targ_sent
        if buckets:
            z = zip(*buckets)
            self.max_sent_len = max(max(z[0]), max(z[1]))
        else:
            self.max_sent_len = max_sent_len
        if self.max_sent_len:
            self.src_sent, self.targ_sent = self.filter_long_sent(
                self.src_sent, self.targ_sent, self.max_sent_len) 
        self.src_vocab = dataset.src_vocab
        self.targ_vocab = dataset.targ_vocab
        self.inv_src_vocab = dataset.inv_src_vocab
        self.inv_targ_vocab = dataset.inv_targ_vocab
        # Can't filter smaller counts per bucket if those sentences still exist!
        self.buckets = buckets if buckets else self.gen_buckets(
            self.src_sent, self.targ_sent, filter_smaller_counts_than=1, max_sent_len=max_sent_len)
        self.bisect = Seq2SeqIter.TwoDBisect(self.buckets)
        self.max_sent_len = max_sent_len
        self.pad_id = self.src_vocab['<PAD>']
        # After bucketization, we should probably del self.src_sent and self.targ_sent
        # to free up memory.
        self.sorted_keys = None
        self.bucketed_data, self.bucket_idx_to_key = self.bucketize()
        self.bucket_key_to_idx = invert_dict(dict(enumerate(self.bucket_idx_to_key)))
        self.interbucket_idx = -1
        self.curr_bucket_id = None
        self.curr_chunks = None
        self.curr_buck = None
        self.switch_bucket = True
        self.num_buckets = len(self.bucket_idx_to_key)
        self.bucket_iterator_indices = list(range(self.num_buckets))
        self.default_bucket_key = self.sorted_keys[-1]

        if self.major_axis == 0:
            self.provide_data = [(data_name, (batch_size, self.default_bucket_key))]
            self.provide_label = [(label_name, (batch_size, self.default_bucket_key))]
        elif self.major_axis == 1:
            self.provide_data = [(data_name, (self.default_bucket_key, batch_size))]
            self.provide_label = [(label_name, (self.default_bucket_key, batch_size))]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")
    
    def bucketize(self):
        tuples = []
        ctr = 0
        for src, targ in zip(self.src_sent, self.targ_sent):
            len_tup = self.bisect.twod_bisect(src, targ)
            rev_src = src[::-1] 
            tuples.append((src, targ, len_tup))
            
        sorted_keys = sorted(tuples, key=operator.itemgetter(2))
        grouped = groupby(sorted_keys, lambda x: x[2])
        self.sorted_keys = map(lambda x: x[2], sorted_keys)
        bucketed_data = [] 
        bucket_idx_to_key = []
        
        for group in grouped:
            
            # get src and targ sentences, ignore the last elem of the tuple 
            # (the grouping key of (src_len, targ_len))
            key, value = group[0], map(lambda x: x[:2], group[1])
            if len(value) < self.batch_size:
                continue

            # create padded representation
            new_src = np.full((len(value), key[0]), self.pad_id, dtype=self.dtype)
            new_targ = np.full((len(value), key[1]), self.pad_id, dtype=self.dtype)
            
            for idx, example in enumerate(value):
                curr_src, curr_targ = example
                rev_src = curr_src[::-1]
                new_src[idx, :-(len(rev_src)+1):-1] = curr_src
                new_targ[idx, :len(curr_targ)] = curr_targ
                            
            bucketed_data.append((new_src, new_targ))

            bucket_idx_to_key.append(key)
        return bucketed_data, bucket_idx_to_key
    
    def current_bucket_key(self):
        return self.bucket_idx_to_key[self.interbucket_idx]
    
    def current_bucket_index(self):
        return self.bucket_iterator_indices[self.interbucket_idx]

    # shuffle the data within buckets, and reset iterator
    def reset(self):
        self.interbucket_idx = -1
        for idx in xrange(len(self.bucketed_data)):
            current = self.bucketed_data[idx]
            src, targ = current
            indices = np.array(range(src.shape[0]))
            np.random.shuffle(indices)
            src = src[indices]
            targ = targ[indices]
            self.bucketed_data[idx] = (src, targ)
        shuffle(self.bucket_iterator_indices)

    # iterate over data
    def next(self):
        try:
            if self.switch_bucket:
                self.interbucket_idx += 1
                self.curr_bucket_id = self.bucket_iterator_indices[self.interbucket_idx]
                self.curr_buck = self.bucketed_data[self.curr_bucket_id]
                src_buck_len, src_buck_wid = self.curr_buck[0].shape
                targ_buck_len, targ_buck_wid = self.curr_buck[1].shape                 
                if src_buck_len == 0 or src_buck_wid == 0:
                    raise StopIteration
                if targ_buck_len == 0 or targ_buck_wid == 0:
                    raise StopIteration
                self.curr_chunks = self.chunks(range(src_buck_len), self.batch_size)
                self.switch_bucket = False
            current = self.curr_chunks.next()
            src_ex = ndarray.array(self.curr_buck[0][current])
            targ_ex = ndarray.array(self.curr_buck[1][current])
            if self.major_axis:
                src_ex = src_ex.T
                targ_ex = targ_ex.T
            
            return DataBatch([src_ex], [targ_ex], pad=0,
                             bucket_key=self.bucket_idx_to_key[self.curr_bucket_id][0],
                             provide_data=[(self.data_name, src_ex.shape)],
                             provide_label=[(self.label_name, targ_ex.shape)])
                
        except StopIteration as si:
            if self.interbucket_idx == self.num_buckets - 1:
                self.reset()
                self.switch_bucket = True
                raise si
            else:
                self.switch_bucket = True
                return self.next()

    @staticmethod
    def chunks(iterable, batch_size, trim_incomplete_batches=True):
        n = max(1, batch_size)
        end = len(iterable)/n*n if trim_incomplete_batches else len(iterable)
        return (iterable[i:i+n] for i in xrange(0, end, n))
    
    @staticmethod 
    def filter_long_sent(src_sent, targ_sent, max_len):
        result = filter(lambda x: len(x[0]) <= max_len and len(x[1]) <= max_len, zip(src_sent, targ_sent))
        return zip(*result)

    @staticmethod
    def gen_buckets(src_sent, targ_sent, filter_smaller_counts_than=None, max_sent_len=60, min_sent_len=1):
        length_pairs = map(lambda x: (len(x[0]), len(x[1])), zip(src_sent, targ_sent))
        counts = list(Counter(length_pairs).items())
        c_sorted = sorted(counts, key=operator.itemgetter(0, 1))
        buckets = [i[0] for i in c_sorted if i[1] >= filter_smaller_counts_than and 
                   (max_sent_len is None or i[0][0] <= max_sent_len) and
                   (max_sent_len is None or i[0][1] <= max_sent_len) and
                   (min_sent_len is None or i[0][0] >= min_sent_len) and
                   (min_sent_len is None or i[0][1] >= min_sent_len)]
        return buckets

def print_text(iterator, max_examples_per_bucket=1):

    inv_src_vocab = iterator.inv_src_vocab
    inv_targ_vocab = iterator.inv_targ_vocab

    try:
        while True:
            data = iterator.next()
            src_text = data.data[0].asnumpy()
            targ_text = data.label[0].asnumpy()
            src_vocab = iterator.src_vocab
            targ_vocab = iterator.targ_vocab
            inv_src_vocab = iterator.inv_src_vocab
            inv_targ_vocab = iterator.inv_targ_vocab

            for i in range(min(max_examples_per_bucket, len(src_text))):
                print("\n" + "-" * 40 + "\n")
                source = src_text[i]
                s = []
                target = targ_text[i]
                t = []
                for j in range(len(source)):
                    s.append(inv_src_vocab[int(source[j])])
                print("source: %s" % " ".join(s))
                for j in range(len(target)):
                    t.append(inv_targ_vocab[int(target[j])])
                print("\ntarget: %s" % " ".join(t))
            
    except StopIteration:
        return

if __name__ == '__main__':

    # Get rid of annoying Python deprecation warnings from built-in JSON encoder
    warnings.filterwarnings("ignore", category=DeprecationWarning)   

    dataset = get_s2s_data(
        src_path='./data/europarl-v7.es-en.en_train_small',
        targ_path='./data/europarl-v7.es-en.es_train_small',
        start_label=1,
        invalid_label=0
    )

    src_sent = dataset.src_sent
    targ_sent = dataset.targ_sent

    sent_len = lambda x: map(lambda y: len(y), x)
    max_len = lambda x: max(sent_len(x))
    min_len = lambda x: min(sent_len(x))

    min_len = min(min(sent_len(src_sent)), min(sent_len(targ_sent)))

    max_len = 65
    increment = 5

    all_pairs = [(i, j) for i in xrange(
            min_len,max_len+increment,increment
        ) for j in xrange(
            min_len,max_len+increment,increment
        )]

    i1 = Seq2SeqIter(dataset, buckets=all_pairs)

    print_text(i1) 
