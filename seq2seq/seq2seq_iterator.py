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

import operator
import pickle
#import dill as pickle
import re
import warnings

class TwoDBisect:
    def __init__(self, buckets):
        self.buckets = sorted(buckets, key=operator.itemgetter(0, 1))
        self.x, self.y = zip(*buckets)
        self.x, self.y = np.array(list(self.x)), np.array(list(self.y))

    def twod_bisect(self, source, target):    
        offset1 = np.searchsorted(self.x, len(source), side='left')
        offset2 = np.where(self.y[offset1:] >= len(target))[0]        
        return self.buckets[offset1 + offset2[0]]     

class Seq2SeqIter(DataIter):

    def __init__(
        self, src_sent, targ_sent, src_vocab, inv_src_vocab, targ_vocab, inv_targ_vocab,
        buckets=None, batch_size=32, max_sent_len=None,
        src_data_name='src_data', targ_data_name='targ_data',
        label_name='softmax_label', dtype=np.int32, layout='TN'):
        self.src_data_name = src_data_name
        self.targ_data_name = targ_data_name
        self.label_name = label_name
        self.dtype = dtype
        self.layout = layout
        self.batch_size = batch_size
        self.src_sent = src_sent
        self.targ_sent = targ_sent
        self.src_vocab = src_vocab
        self.inv_src_vocab = inv_src_vocab
        self.targ_vocab = inv_targ_vocab
        if buckets:
            z = zip(*buckets)
            self.max_sent_len = max(max(z[0]), max(z[1]))
        else:
            self.max_sent_len = max_sent_len
        if self.max_sent_len:
            self.train_sent, self.targ_sent = self.filter_long_sent(
                self.src_sent, self.targ_sent, self.max_sent_len) 
        self.src_vocab = src_vocab
        self.targ_vocab = targ_vocab
        self.inv_src_vocab = inv_src_vocab
        self.inv_targ_vocab = inv_targ_vocab
        # Can't filter smaller counts per bucket if those sentences still exist!
        self.buckets = buckets if buckets else self.gen_buckets(
            self.train_sent, self.targ_sent, filter_smaller_counts_than=1, max_sent_len=max_sent_len)
        self.bisect = TwoDBisect(self.buckets)
        self.max_sent_len = max_sent_len
        self.pad_id = self.src_vocab['<PAD>']
        self.eos_id = self.src_vocab['<EOS>']
        self.unk_id = self.src_vocab['<UNK>']
        self.go_id  = self.src_vocab['<GO>']
        # After bucketization, we should probably del self.train_sent and self.targ_sent
        # to free up memory.
        self.sorted_keys = None
        self.bucketed_data = None
        self.bucket_idx_to_key = None
        #  self.bucketize()
        self.bucket_key_to_idx = None
        self.interbucket_idx = -1
        self.curr_bucket_id = None
        self.curr_chunks = None
        self.curr_buck = None
        self.switch_bucket = True
        self.num_buckets = -1
        self.bucket_iterator_indices = []
        self.default_bucket_key = -1 
        self.mappings = None
        self.provide_data = None
        self.provide_label = None

#        if self.layout == 'TN':
#            self.provide_data = [
#                mx.io.DataDesc(self.src_data_name, (self.default_bucket_key[0], self.batch_size), layout='TN'),
#                mx.io.DataDesc(self.targ_data_name, (self.default_bucket_key[0], self.batch_size), layout='TN')
#            ]
#            self.provide_label = [mx.io.DataDesc(self.label_name, (self.default_bucket_key[1], self.batch_size), layout='TN')] 
#        elif self.layout == 'NT':
#            self.provide_data = [
#                (self.src_data_name, (self.batch_size, self.default_bucket_key[0])),
#                (self.targ_data_name, (self.batch_size, self.default_bucket_key[0]))]
#            self.provide_label = [(self.label_name, (self.batch_size, self.default_bucket_key[1]))]
#        else:
#            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

    def initialize(self):
        if self.layout == 'TN':
            self.provide_data = [
                mx.io.DataDesc(self.src_data_name, (self.default_bucket_key[0], self.batch_size), layout='TN'),
                mx.io.DataDesc(self.targ_data_name, (self.default_bucket_key[0], self.batch_size), layout='TN')
            ]
            self.provide_label = [mx.io.DataDesc(self.label_name, (self.default_bucket_key[1], self.batch_size), layout='TN')]
        elif self.layout == 'NT':
            self.provide_data = [
                (self.src_data_name, (self.batch_size, self.default_bucket_key[0])),
                (self.targ_data_name, (self.batch_size, self.default_bucket_key[0]))]
            self.provide_label = [(self.label_name, (self.batch_size, self.default_bucket_key[1]))]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

    
    def bucketize(self):
        tuples = []
        ctr = 0
        for src, targ in zip(self.train_sent, self.targ_sent):
            len_tup = self.bisect.twod_bisect(src, targ)
            tuples.append((src, targ, len_tup))
            
        sorted_keys = sorted(tuples, key=operator.itemgetter(2))
        grouped = groupby(sorted_keys, lambda x: x[2])
        self.sorted_keys = map(lambda x: x[2], sorted_keys)
        self.bucketed_data = [] 
        self.bucket_idx_to_key = []

        global_count = 0L
        error_count  = 0L        

        for group in grouped:
            
            # get src and targ sentences, ignore the last elem of the tuple 
            # (the grouping key of (src_len, targ_len))
            key, value = group[0], map(lambda x: x[:2], group[1])
            if len(value) < self.batch_size:
                continue

            # create padded representation
            new_src = np.full((len(value), key[0]), self.pad_id, dtype=self.dtype)
            new_targ = np.full((len(value), key[1] + 1), self.pad_id, dtype=self.dtype)
            new_label = np.full((len(value), key[1] + 1), self.pad_id, dtype=self.dtype)
            
            for idx, example in enumerate(value):
                try:
                    global_count += 1
                    curr_src, curr_targ = example
                    rev_src = curr_src[::-1]
                    new_src[idx, -len(curr_src):] = rev_src

                    new_targ[idx, 0] = self.go_id
                    new_targ[idx, 1:(len(curr_targ)+1)] = curr_targ

                    new_label[idx, 0:len(curr_targ)] = curr_targ
                    new_label[idx, len(curr_targ)] = self.eos_id
                except ValueError as ve:
                    error_count += 1
                    print(ve.message)
                    print("global count: %d, error count: %d" % (global_count, error_count))
                    continue
                            
            self.bucketed_data.append((new_src, new_targ, new_label))

            self.bucket_idx_to_key.append((key[0], key[1]+1))


        self.bucket_key_to_idx = invert_dict(dict(enumerate(self.bucket_idx_to_key)))
        self.interbucket_idx = -1
        self.curr_bucket_id = None
        self.curr_chunks = None
        self.curr_buck = None
        self.switch_bucket = True
        self.num_buckets = len(self.bucket_idx_to_key)
        self.bucket_iterator_indices = list(range(self.num_buckets))
        self.default_bucket_key = self.sorted_keys[-1]

#        if self.layout == 'TN':
#            self.provide_data = [
#                mx.io.DataDesc(self.src_data_name, (self.default_bucket_key[0], self.batch_size), layout='TN'),
#                mx.io.DataDesc(self.targ_data_name, (self.default_bucket_key[0], self.batch_size), layout='TN')
#            ]
#            self.provide_label = [mx.io.DataDesc(self.label_name, (self.default_bucket_key[1], self.batch_size), layout='TN')]
#        elif self.layout == 'NT':
#            self.provide_data = [
#                (self.src_data_name, (self.batch_size, self.default_bucket_key[0])),
#                (self.targ_data_name, (self.batch_size, self.default_bucket_key[0]))]
#            self.provide_label = [(self.label_name, (self.batch_size, self.default_bucket_key[1]))]
#        else:
#            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

#        return bucketed_data, bucket_idx_to_key
    
    def current_bucket_key(self):
        return self.bucket_idx_to_key[self.interbucket_idx]
    
    def current_bucket_index(self):
        return self.bucket_iterator_indices[self.interbucket_idx]

    # shuffle the data within buckets, and reset iterator
    def reset(self):
        self.interbucket_idx = -1
        for idx in xrange(len(self.bucketed_data)):
            current = self.bucketed_data[idx]
            src, targ, label = current
            indices = np.array(range(src.shape[0]))
            np.random.shuffle(indices)
            src = src[indices]
            targ = targ[indices]
            label = label[indices]
            self.bucketed_data[idx] = (src, targ, label)
        shuffle(self.bucket_iterator_indices)

    @staticmethod
    def _normalize_path(path, name, ext='npz'):
        path = os.path.normpath(path) + os.sep
        return path + name + '.' + ext

    def save(self, file_path):
        if not self.bucketed_data:
            raise Exception("Bucketed data does not exist. First run bucketize() on the iterator to create buckets.")
        bucketed_data = self.bucketed_data
        directory = os.path.dirname(file_path) 
        self.bucketed_data = None
        # Note: This has to come before metadata to store mappings in the Seq2SeqIter instance.
        print("Saving NumPy arrays.")
        self._serialize_list_tup_np_arr(bucketed_data, directory)        
        print("Done saving NumPy arrays.")
        print("Saving metadata.")
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, 2)
        print("Metadata saved.")
        
    @staticmethod
    def load(metadata_path):
        directory = os.path.dirname(metadata_path)
        with open(metadata_path, 'rb') as f:
            iterator = pickle.load(f)
        iterator._deserialize_np_arrays(directory)
        iterator.mappings = None
        # Create inverse lookup of buckets.
        # If TN, ba
        #src_ex.shape[0], src_ex.shape[1])
        iterator.bucket_idx_to_key = []
        for bucket in iterator.bucketed_data:
            src_len = np.shape(bucket[0])[1]
            label_len = np.shape(bucket[2])[1]
            iterator.bucket_idx_to_key.append((src_len, label_len))
        iterator.bucket_key_to_idx = invert_dict(dict(enumerate(iterator.bucket_idx_to_key)))
        return iterator        

    @staticmethod
    def _random_uuid():
        return str(uuid.uuid5(uuid.NAMESPACE_OID, str(time())))

    def _serialize_list_tup_np_arr(self, data, path, extension='npz'):
        self.mappings = []
        for entry in tqdm(data, desc='Saving NumPy arrays'):
            src, targ, label = entry
            uuid = Seq2SeqIter._random_uuid()
            self.mappings.append(uuid)
            with open(Seq2SeqIter._normalize_path(path, uuid), 'wb') as f:
                np.savez_compressed(f, src=src, targ=targ, label=label)

    def _deserialize_np_arrays(self, directory):
        self.bucketed_data = []
        for entry in tqdm(self.mappings, desc='Deserializing NumPy arrays'):
            with open(Seq2SeqIter._normalize_path(directory, entry)) as f:
                npz_file = np.load(f)
                src = npz_file['src']
                targ = npz_file['targ']
                label = npz_file['label']
            self.bucketed_data.append((src, targ, label))

    # iterate over data
    def next(self):
        while True:
            try:
                if self.switch_bucket:
                    self.interbucket_idx += 1
                    if self.interbucket_idx >= self.num_buckets - 1:
                        raise StopIteration
                    self.curr_bucket_id = self.bucket_iterator_indices[self.interbucket_idx]
                    self.curr_buck = self.bucketed_data[self.curr_bucket_id]
                    src_buck_len, src_buck_wid = self.curr_buck[0].shape
                    targ_buck_len, targ_buck_wid = self.curr_buck[1].shape                 
                    if src_buck_len == 0 or src_buck_wid == 0:
                        print("src_buck_len == 0 or src_buck_wid == 0")
                        continue
                    if targ_buck_len == 0 or targ_buck_wid == 0:
                         print("targ_buck_len == 0 or targ_buck_wid == 0")
                         continue
                    self.curr_chunks = self.chunks(range(src_buck_len), self.batch_size)
                    self.switch_bucket = False

                try:
                    current = self.curr_chunks.next()
                except StopIteration as si:
                    print("end of bucket %d of %d" % (self.interbucket_idx, len(self.bucket_iterator_indices)))
                    self.switch_bucket = True
                    continue
                src_ex = ndarray.array(self.curr_buck[0][current])
                targ_ex = ndarray.array(self.curr_buck[1][current])
                label_ex = ndarray.array(self.curr_buck[2][current])

                if self.layout == 'TN':
                    src_ex = src_ex.T
                    targ_ex = targ_ex.T
                    label_ex = label_ex.T

                if self.layout == 'TN':
                    provide_data = [
                        mx.io.DataDesc(self.src_data_name, (src_ex.shape[0], src_ex.shape[1]), layout='TN'),
                        mx.io.DataDesc(self.targ_data_name, (targ_ex.shape[0], targ_ex.shape[1]), layout='TN')] # src_ex.shape[1] # self.batch_size
                    provide_label = [mx.io.DataDesc(self.label_name, (targ_ex.shape[0], self.batch_size), layout='TN')] # targ_ex.shape[1]
 
                elif self.layout == 'NT':
                    provide_data = [
                        (self.src_data_name, (self.batch_size, src_ex.shape[0])),
                        (self.targ_data_name, (self.batch_size, targ_ex.shape[0]))]
                    provide_label = [(self.label_name, (self.batch_size, targ_ex.shape[0]))]
                else:
                    raise Exception("Layout must be 'TN' or 'NT'") 

 
                batch = DataBatch([src_ex, targ_ex], [label_ex], pad=0,
                                 bucket_key=self.bucket_idx_to_key[self.curr_bucket_id],
                                 provide_data=provide_data,
                                 provide_label=provide_label)
                return batch

            # This is probably redundant                
            except StopIteration as si:
                if self.interbucket_idx == self.num_buckets - 1:
                    self.reset()
                    self.switch_bucket = True
                    raise si
#                else:
#                    self.switch_bucket = True
#                    return self.next()


                
#        except StopIteration as si:
#            if self.interbucket_idx == self.num_buckets - 1:
#                self.reset()
#                self.switch_bucket = True
#                raise si
#            else:
#                self.switch_bucket = True
#                return self.next()

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

