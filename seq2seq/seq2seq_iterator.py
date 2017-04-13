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
import dill as pickle
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
#
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
                curr_src, curr_targ = example
                rev_src = curr_src[::-1]
                new_src[idx, -len(curr_src):] = rev_src

                new_targ[idx, 0] = self.go_id
                new_targ[idx, 1:(len(curr_targ)+1)] = curr_targ

                new_label[idx, 0:len(curr_targ)] = curr_targ
                new_label[idx, len(curr_targ)] = self.eos_id
                            
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
        del self.bucketed_data
        print("Saving metadata.")
        directory = os.path.dirname(file_path) 
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, 2)
        print("Metadata saved.")
        print("Saving NumPy arrays.")
        self._serialize_list_tup_np_arr(bucketed_data, directory)        
        print("Done saving NumPy arrays.")
        
    @staticmethod
    def load():
        # Load metadata file.
        # Get mappings from the instantiated object.
        # Load NumPy arrays to bucketed_data.
        # Create inverse lookup of buckets.
        # Remove the mappings fom the Seq2SeqIter instance.
        # Return Seq2SeqIter instance. 
        pass

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

    def _deserialize_np_arrays(mappings, path):
        data = []
        for entry in mappings:
            with open(normalize_path(path, entry)) as f:
                npz_file = np.load(f)
                src = npz_file['src']
                targ = npz_file['targ']
                label = npz_file['label']
            data.append((src, targ, label))
        return data
 

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

def print_text(iterator, max_examples_per_bucket=100):

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

def random_uuids(count=3):
    rand_id = lambda: uuid.uuid5(uuid.NAMESPACE_OID, str(time()))
    return list([str(rand_id()) for _ in range(count)])

# TODO: try serializing as npz rather than npy

def append_ext(name, ext='npz'):
    return name + '.' + ext

def serialize_list_tup_np_arr(data, extension='npz'):
    mappings = []
    for entry in data:
        src, targ, label = entry
        uuid = random_uuids()[0]
        mappings.append(uuid)
        with open(append_ext(uuid), 'wb') as f:
            np.savez_compressed(f, src=src, targ=targ, label=label)
#        for arr, name in zip([src, targ, label], uuids):
#            with open(append_ext(name), 'wb') as f:
#                np.save(f, arr)
    return mappings

def deserialize_np_arrays(mappings):
    data = []
    for entry in mappings:
        with open(append_ext(entry)) as f:
            npz_file = np.load(f)
            src = npz_file['src']
            targ = npz_file['targ']
            label = npz_file['label']    
            data.append((src, targ, label))
    return data            

if __name__ == '__main__':

    # Get rid of annoying Python deprecation warnings from built-in JSON encoder
    warnings.filterwarnings("ignore", category=DeprecationWarning)   
 
    start = time()

    dataset = get_s2s_data(
        src_train_path='./data/europarl-v7.es-en.en_train_small',
        src_valid_path='./data/europarl-v7.es-en.en_valid_small', # valid_small',
        targ_train_path='./data/europarl-v7.es-en.es_train_small',
        targ_valid_path='./data/europarl-v7.es-en.es_valid_small' # valid_small'
    )
   
    preproc_duration = time() - start
    print("\nPreprocessing data took %.4f seconds\n" % preproc_duration)

    min_len = 5

    max_len = 65
    increment = 5

    all_pairs = [(i, j) for i in xrange(
            min_len,max_len+increment,increment
        ) for j in xrange(
            min_len,max_len+increment,increment
        )]

    train_iter = Seq2SeqIter(dataset.src_train_sent, dataset.targ_train_sent, dataset.src_vocab, dataset.inv_src_vocab,
                     dataset.targ_vocab, dataset.inv_targ_vocab, layout='TN', batch_size=32, buckets=all_pairs)

    train_iter.bucketize()

    train_iter.reset()   

    train_iter.save('iterator.pkl')

#    bucketed_data = train_iter.bucketed_data
#    mappings = serialize_list_tup_np_arr(bucketed_data)
    
#    del bucketed_data
#    start = time()
#    bucketed_data = deserialize_np_arrays(mappings)
#    deser_duration = time() - start
  
#    print("Deserializing preprocessed NumPy arrays took %.4f seconds\n" % deser_duration) 

#    print("Speed-up from preprocessing: %.1f times\n" % (preproc_duration / deser_duration))
