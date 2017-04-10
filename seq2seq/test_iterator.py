import numpy as np
import mxnet as mx
import argparse

import re
from unidecode import unidecode

from common import tokenize_text, invert_dict, get_s2s_data, Dataset

from seq2seq_iterator import *

from attention_cell import AttentionEncoderCell, DotAttentionCell

parser = argparse.ArgumentParser(description="Train RNN on Penn Tree Bank",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default=False, action='store_true',
                    help='whether to do testing instead of training')
parser.add_argument('--model-prefix', type=str, default=None,
                    help='path to save/load model')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='load from epoch')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=200,
                    help='hidden layer size')
parser.add_argument('--num-embed', type=int, default=200,
                    help='embedding layer size')
parser.add_argument('--bidirectional', type=bool, default=False,
                    help='whether to use bidirectional layers')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=25,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
# When training a deep, complex model, it's recommended to stack fused RNN cells (one
# layer per cell) together instead of one with all layers. The reason is that fused RNN
# cells doesn't set gradients to be ready until the computation for the entire layer is
# completed. Breaking a multi-layer fused RNN cell into several one-layer ones allows
# gradients to be processed ealier. This reduces communication overhead, especially with
# multiple GPUs.
parser.add_argument('--stack-rnn', default=False,
                    help='stack fused RNN cells to reduce communication overhead')
parser.add_argument('--dropout', type=float, default='0.0',
                    help='dropout probability (1.0 - keep probability)')

#buckets = [32]
buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

start_label = 1
invalid_label = 0

def print_inferred_shapes(node, arg_shapes, aux_shapes, out_shapes):
    args = node.list_arguments()
    aux_states = node.list_auxiliary_states()
    outputs = node.list_outputs()
    print("\n================================================")
    print("\nNODE: %s" % node.name)
    print("\n============")
    print("args:")
    print("============")
    if len(arg_shapes) == 0:
        print("N/A")
    for i in range(len(arg_shapes)):
        print("%s: %s" % (args[i], arg_shapes[i]))
    print("\n=============")
    print("aux_states:")
    print("=============")
    if len(aux_shapes) == 0:
        print("N/A")
    for i in range(len(aux_states)):
        print("%s: %s" % (aux_states[i], aux_shapes[i]))
    print("\n=============")
    print("outputs:")
    print("==============")
    if len(out_shapes) == 0:
        print("N/A")
    for i in range(len(outputs)):
        print("%s: %s" % (outputs[i], out_shapes[i]))
    print("\n================================================")
    print("\n")

def _normalize_sequence(length, inputs, layout, merge, in_layout=None):
    from mxnet import symbol, init, ndarray, _symbol_internal

    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, symbol.Symbol):
        if merge is False:
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. Please convert " \
                "to list with list(inputs) first or let unroll handle splitting."
            inputs = list(symbol.split(inputs, axis=in_axis, num_outputs=length,
                                       squeeze_axis=1))
    else: 
        assert length is None or len(inputs) == length
        if merge is True:
            inputs = [symbol.expand_dims(i, axis=axis) for i in inputs]
            inputs = symbol.Concat(*inputs, dim=axis)
            in_axis = axis

    if isinstance(inputs, symbol.Symbol) and axis != in_axis:
        inputs = symbol.swapaxes(inputs, dim0=axis, dim1=in_axis)

    return inputs, axis

def get_data2(layout, buckets):

    train_dataset = get_s2s_data(
        src_path='./data/europarl-v7.es-en.en_train_small',
        targ_path='./data/europarl-v7.es-en.es_train_small',
        start_label=1,
        invalid_label=0
    )

    valid_dataset = get_s2s_data(
        src_path='./data/europarl-v7.es-en.en_valid_small',
        targ_path='./data/europarl-v7.es-en.es_valid_small',
        start_label=1,
        invalid_label=0
    )

    train_src_sent = train_dataset.src_sent
    train_targ_sent = train_dataset.targ_sent

    sent_len = lambda x: map(lambda y: len(y), x)
    max_len = lambda x: max(sent_len(x))
    min_len = lambda x: min(sent_len(x))

    min_len = 5 #min(min(sent_len(train_src_sent)), min(sent_len(train_targ_sent)))

    max_len = 65
    increment = 5

    all_pairs = [(i, j) for i in xrange(
            min_len,max_len+increment,increment
        ) for j in xrange(
            min_len,max_len+increment,increment
        )]

    train_iter = Seq2SeqIter(train_dataset, layout=layout, batch_size=32, buckets=buckets)
    valid_iter = Seq2SeqIter(valid_dataset, layout=layout, batch_size=32, buckets=buckets)
    train_iter.reset()
    valid_iter.reset()
    
    print("\nSize of src vocab: %d" % len(train_iter.src_vocab))
    print("Size of targ vocab: %d\n" % len(train_iter.targ_vocab))

    return train_iter, valid_iter, train_iter.src_vocab, train_iter.targ_vocab

def get_data(layout, buckets):
    source_data = "./data/europarl-v7.es-en.es_train_small"
    target_data = "./data/europarl-v7.es-en.en_train_small"
    src_sent, src_vocab = tokenize_text(source_data, start_label=start_label,
                                      invalid_label=invalid_label)
    targ_sent, targ_vocab = tokenize_text(target_data, vocab=None, start_label=start_label, # vocab, start_label=start_label,
                                invalid_label=invalid_label)
 
    data_src  = mx.rnn.BucketSentenceIter(src_sent, 32, buckets=buckets,
                                            invalid_label=0, layout='TNC')
    data_targ    = mx.rnn.BucketSentenceIter(targ_sent, 32, buckets=buckets,
                                            invalid_label=0, layout='TNC')
    return data_src, data_targ, src_vocab, targ_vocab


# WORK IN PROGRESS !!!
def decoder_unroll(decoder, target_embed, targ_vocab, unroll_length, go_symbol, begin_state=None, layout='TNC', merge_outputs=None):

        decoder.reset()

        if begin_state is None:
            begin_state = decoder.begin_state()

        print("first normalize sequence")
        print("decoder_unroll: type(target_embed) = %s" % str(type(target_embed)))
        inputs, _ = _normalize_sequence(unroll_length, target_embed, layout, False)

        # Need to use hidden state from attention model, but <GO> as input
        states = begin_state
        outputs = []

        # Replace this with a <GO> symbol
        feed = inputs[0]
        output, states = decoder(feed, states)

        pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden), name='output_reshape') 
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(targ_vocab), name='pred')
        output = mx.sym.argmax(pred, name='argmax') 

#        outputs, _ = _normalize_sequence(1, outputs, layout, merge_outputs)

        pred_word_idx = mx.sym.Variable('pred_word_idx')

        embed = mx.sym.Embedding(data=pred_word_idx, input_dim=len(targ_vocab),
            output_dim=args.num_embed, name='src_embed') 

        for i in range(0, unroll_length):
            # this works            
            output, states = decoder(inputs[i], states)
            outputs.append(output)


#            output, states = decoder(output, states)
#            outputs.append(output)

#            pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden), name='pred_reshape') 
#            pred = mx.sym.FullyConnected(data=pred, num_hidden=len(targ_vocab), name='loop_pred')
#            pred = mx.sym.Reshape(pred, shape=(-1,))
            # record actual probs for softmax, then get new token for embedding
#            outputs.append(pred)
#            output = mx.sym.argmax(pred)
#            result = output.eval().asnumpy()
#            new_word_idx = output.forward()
#            output = output.eval(contexts, data={'pred_word_idx': result})
#            pred_word_idx.bind(contexts, {'pred_word_idx': output})
            #print("\ntype(new_word): %s" % type(new_word_idx)) 
#            output = embed 

        print("second normalize sequence")
        print("len(outputs): %d" % len(outputs))
        print("unroll_length: %d" % unroll_length)
        outputs, _ = _normalize_sequence(unroll_length, outputs, layout, merge_outputs)

        return outputs, states

min_len = 5
max_len = 65
increment = 5

#buckets = [(i, j) for i in xrange(
#        min_len,max_len+increment,increment
#    ) for j in xrange(
#        min_len,max_len+increment,increment
#    )]

from time import time

buckets = list(range(min_len, max_len, increment))

#start = time()
#data_train1, data_val1, src_vocab1, targ_vocab1 = get_data('TNC', buckets)
#print(time() - start)


buckets = [(i, j) for i in xrange(
        min_len,max_len+increment,increment
    ) for j in xrange(
        min_len,max_len+increment,increment
    )]

start = time()
data_train2, data_val2, src_vocab2, targ_vocab2 = get_data2('TN', buckets)
print(time() - start)

#print(data_train2)
print(type(data_train2))

print(dir(data_train2))

#try:
#    while True:
#        print(data_train1.next())
#except StopIteration as e:
#    pass
#
#print("\n\ndata2\n\n")
#
#try:
#    while True:
#        print(data_train2.next())
#except StopIteration as e:
#    pass
#
#
#print("\n\n\n")
#
#print(data_train2.next())
