import numpy as np
import mxnet as mx
import argparse
import cPickle as pickle
#import dill as pickle
import math
import nltk

from mxnet.rnn import LSTMCell, SequentialRNNCell, FusedRNNCell, BidirectionalCell
#from rnn_cell import LSTMCell, SequentialRNNCell
from itertools import takewhile, dropwhile
from operator import itemgetter

from time import time
import re
from unidecode import unidecode

from utils import array_to_text, tokenize_text, invert_dict, get_s2s_data, Dataset

from seq2seq_iterator import *

# from attention_cell import AttentionEncoderCell, DotAttentionCell

parser = argparse.ArgumentParser(description="Train RNN on Penn Tree Bank",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--infer', default=False, action='store_true',
                    help='whether to do inference instead of training')
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
parser.add_argument('--bidirectional', action='store_true',
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
parser.add_argument('--max-grad-norm', type=float, default=5.0,
                    help='maximum gradient norm (larger values will be clipped')

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
parser.add_argument('--use-cudnn-cells', action='store_true',
                    help='Use CUDNN LSTM (mx.rnn.FusedRNNCell) for training instead of in-graph LSTM cells (mx.rnn.LSTMCell)')

parser.add_argument('--inference-unrolling-for-training', action='store_true',
                    help='Feed previous prediction (instead of previous ground truth) into the decoder input during training')
parser.add_argument('--seed', type=int, default=1234,
                    help='Set random seed for Python, NumPy and MxNet RNGs')

parser.add_argument('--remove-state-feed', action='store_true',
                    help='Remove direct state feeding from encoder to decoder (use when using attention)')


parser.add_argument('--input-feed', action='store_true',
                    help='Enable input feed (attention is fed into the decoder as input, rather than concatenated with output)')

parser.add_argument('--attention', action='store_true',
                    help='Use attention (dot attention is the currently implemented form')

#buckets = [32]
# buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

start_label = 1
invalid_label = 0

reserved_tokens={'<PAD>':0, '<UNK>':1, '<EOS>':2, '<GO>':3}

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

def get_data(layout, infer=False):

    start = time()

    print("\nUnpickling training iterator")

    if not infer:
        with open('./data/train_iterator.pkl', 'rb') as f: # _en_de.pkl
            train_iter = pickle.load(f)
 
        train_iter.initialize(curr_batch_size=args.batch_size)

        print("\nUnpickling validation iterator")

        with open('./data/valid_iterator.pkl', 'rb') as f: # _en_de.pkl
            valid_iter = pickle.load(f)
 
        valid_iter.initialize(curr_batch_size=args.batch_size)

    with open('./data/test_iterator.pkl', 'rb') as f:
        test_iter = pickle.load(f)

    test_iter.initialize(curr_batch_size=args.batch_size)

#    print("\nEncoded source language sentences:\n")
#    for i in range(5):
#        print(array_to_text(train_iter.src_sent[i], train_iter.inv_src_vocab))

#    print("\nEncoded target language sentences:\n")
#    for i in range(5):
#        print(array_to_text(train_iter.targ_sent[i], train_iter.inv_targ_vocab))
    
    duration = time() - start

    print("\nDataset deserialization time: %.2f seconds\n" % duration)
 
    if not infer:
        return train_iter, valid_iter, test_iter, train_iter.src_vocab, train_iter.targ_vocab, train_iter.inv_src_vocab, train_iter.inv_targ_vocab
    else:
        return test_iter, test_iter.src_vocab, test_iter.inv_src_vocab, test_iter.targ_vocab, test_iter.inv_targ_vocab

def attention_step(i, encoder_outputs, decoder_output):

    attention_state = mx.sym.zeros_like(encoder_outputs[-1], name='train_dec_unroll_attention_state')
    curr_att_input = mx.sym.expand_dims(decoder_output, axis=2, name='train_dec_unroll_expand_dims_%d_' % i)
    enc_len = len(encoder_outputs)
    dots = []
    concat_dots = None
    # loop over all the encoder periods to create weights for weighted state
    for j in range(enc_len):
        transposed = mx.sym.expand_dims(encoder_outputs[j], axis=2)
        transposed = mx.sym.transpose(transposed, axes=(0, 2, 1), name='train_decoder_transpose%d_' % i)
        dot = mx.sym.batch_dot(transposed, curr_att_input, name='train_decoder_batch_dot_%d_%d_' % (i, j))
        dot = mx.sym.exp(dot, name='train_decoder_exp_%d_%d' % (i, j))
        # The batch size shouldn't be an arg here anyway. We should just remove extra dimensions
        # and then transpose.
        dot = mx.sym.reshape(dot, shape=(1, args.batch_size / len(contexts)),
                             name='train_decoder_unroll_reshape_%d_%d' % (i, j))
        dots.append(dot)
        if not concat_dots:
            concat_dots = dot
        else:
            concat_dots = mx.sym.concat(concat_dots, dot)
    dot_sum = mx.sym.sum(concat_dots, axis=1)
    for j in range(enc_len):
        curr_dot = mx.sym.transpose(dots[j])
        attention_state += mx.sym.broadcast_mul(curr_dot, encoder_outputs[j],
                                                name='train_encoder_acc_attention_%d_%d_' % (i, j))

    attention_state = mx.sym.broadcast_div(attention_state, dot_sum)

    return attention_state


def train_decoder_unroll(decoder, encoder_outputs, target_embed, targ_vocab, unroll_length,
                         go_symbol, fc_weight, fc_bias, attention_fc_weight, attention_fc_bias, targ_em_weight,
                        begin_state=None, layout='TNC', merge_outputs=None):
    decoder.reset()
    if begin_state is None:
        begin_state = decoder.begin_state()
    inputs, _ = _normalize_sequence(unroll_length, target_embed, layout, False)
    # Need to use hidden state from attention model, but <GO> as input
    states = begin_state
    outputs = []

    #At the first time step there is no previous attention
    attention_state = None

    for i in range(unroll_length):
        if args.input_feed:
            # Copy previous attention output to concatenate with the embedding input
            prev_attention_state = attention_state if attention_state else mx.sym.zeros_like(encoder_outputs[-1], name='train_dec_unroll_prev_attention_state')
            decoder_feed = mx.sym.concat(inputs[i], prev_attention_state, name = 'decoder_feed_concat_%d_' % i)
        else:
            decoder_feed = inputs[i]
        dec_out, states = decoder(decoder_feed, states)

        if args.attention:
            # The attention receives as input all the encoder outputs and the current decoder output and return the vector
            # for this time step
            attention_state = attention_step(i, encoder_outputs, dec_out)
            # The attention output is combined with the decoder output for computing the next word
            concatenated = mx.sym.concat(dec_out, attention_state, name = 'train_decoder_concat_%d_' % i)
            attention_fc = mx.sym.FullyConnected(
                data=concatenated, weight=attention_fc_weight, bias=attention_fc_bias, num_hidden=args.num_hidden, name='attention_fc%d_' % i
            )
            curr_out = mx.sym.Activation(data = attention_fc, act_type='tanh', name = 'attention_tanh%d_' % i)
        else:
            # We avoid all the attention computation
            curr_out = dec_out
        outputs.append(curr_out)
    outputs, _ = _normalize_sequence(unroll_length, outputs, layout, merge_outputs)
    return outputs, states


def infer_decoder_unroll(decoder, encoder_outputs, target_embed, targ_vocab, unroll_length,
                  go_symbol, fc_weight, fc_bias, attention_fc_weight, attention_fc_bias, targ_em_weight,
                  begin_state=None, layout='TNC', merge_outputs=None):
    decoder.reset()
    if begin_state is None:
        begin_state = decoder.begin_state()
    inputs, _ = _normalize_sequence(unroll_length, target_embed, layout, False)
    # Need to use hidden state from attention model, but <GO> as input
    states = begin_state
    outputs = []
    embed = inputs[0]

    attention_state = None

    for i in range(unroll_length):
        if args.input_feed:
            # Copy previous attention output to concatenate with the embedding input
            prev_attention_state = attention_state if attention_state else mx.sym.zeros_like(encoder_outputs[-1],
                                        name='train_dec_unroll_prev_attention_state')
            decoder_feed = mx.sym.concat(embed, prev_attention_state, name='decoder_feed_concat_%d_' % i)
        else:
            decoder_feed = embed
        dec_out, states = decoder(decoder_feed, states)

        # Should this be dec_out or states as the first argument?
        if args.attention:
            attention_state = attention_step(i, encoder_outputs, dec_out)
            concatenated = mx.sym.concat(dec_out, attention_state, name = 'train_decoder_concat_%d_' % i)
            attention_fc = mx.sym.FullyConnected(
                data=concatenated, weight=attention_fc_weight, bias=attention_fc_bias, num_hidden=args.num_hidden, name='attention_fc%d_' % i
            )
            curr_out = mx.sym.Activation(data = attention_fc, act_type='tanh', name = 'attention_tanh%d_' % i)
        else:
            curr_out = dec_out
        outputs.append(curr_out)
        fc = mx.sym.FullyConnected(data=curr_out, weight=fc_weight, bias=fc_bias, num_hidden=len(targ_vocab), name='decoder_fc%d_'%i)
        am = mx.sym.argmax(data=fc, axis=1)
        embed = mx.sym.Embedding(data=am, weight=targ_em_weight, input_dim=len(targ_vocab), output_dim=args.num_embed, name='decoder_embed%d_'%i)

    outputs, _ = _normalize_sequence(unroll_length, outputs, layout, merge_outputs)
    return outputs, states

def train(args):

    from time import time

    data_train, data_val, _, src_vocab, targ_vocab, inv_src_vocab, inv_targ_vocab = get_data('TN')
    print "len(src_vocab) len(targ_vocab)", len(src_vocab), len(targ_vocab)

    attention_fc_weight = mx.sym.Variable('attention_fc_weight')
    attention_fc_bias = mx.sym.Variable('attention_fc_bias')

    fc_weight = mx.sym.Variable('fc_weight')
    fc_bias = mx.sym.Variable('fc_bias')
    targ_em_weight = mx.sym.Variable('targ_embed_weight')

    encoder = SequentialRNNCell()

    if args.use_cudnn_cells:
        encoder.add(mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, dropout=args.dropout,
            mode='lstm', prefix='lstm_encoder', bidirectional=args.bidirectional, get_next_state=True))
    else:
        for i in range(args.num_layers):
            if args.bidirectional:
                encoder.add(
                    BidirectionalCell(
                        LSTMCell(args.num_hidden // 2, prefix='rnn_encoder_f%d_' % i),
                        LSTMCell(args.num_hidden // 2, prefix='rnn_encoder_b%d_' % i)))
                if i < args.num_layers - 1 and args.dropout > 0.0:
                    encoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_encoder%d_' % i))
            else:
                encoder.add(
                    LSTMCell(args.num_hidden, prefix='rnn_encoder%d_' % i))
                if i < args.num_layers - 1 and args.dropout > 0.0:
                    encoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_encoder%d_' % i))

    decoder = mx.rnn.SequentialRNNCell()

    if args.use_cudnn_cells:
        decoder.add(mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, 
            mode='lstm', prefix='lstm_decoder', bidirectional=args.bidirectional, get_next_state=True))
    else:
        for i in range(args.num_layers):
            decoder.add(LSTMCell(args.num_hidden, prefix=('rnn_decoder%d_' % i)))
            if i < args.num_layers - 1 and args.dropout > 0.0:
                decoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_decoder%d_' % i))

    def sym_gen(seq_len):
        src_data = mx.sym.Variable('src_data')
        targ_data = mx.sym.Variable('targ_data')
        label = mx.sym.Variable('softmax_label')
 
        src_embed = mx.sym.Embedding(data=src_data, input_dim=len(src_vocab), 
                                 output_dim=args.num_embed, name='src_embed') 
        targ_embed = mx.sym.Embedding(data=targ_data, weight=targ_em_weight, input_dim=len(targ_vocab),    # data=data
                                 output_dim=args.num_embed, name='targ_embed')

        encoder.reset()
        decoder.reset()

        enc_seq_len, dec_seq_len = seq_len

        layout = 'TNC'
        encoder_outputs, encoder_states = encoder.unroll(enc_seq_len, inputs=src_embed, layout=layout)

        if args.remove_state_feed:
            encoder_states = None

        # This should be based on EOS or max seq len for inference, but here we unroll to the target length
        # TODO: fix <GO> symbol
        if args.inference_unrolling_for_training:
            outputs, _ = infer_decoder_unroll(decoder, encoder_outputs, targ_embed, targ_vocab, dec_seq_len, 0, fc_weight, fc_bias,
                             attention_fc_weight, attention_fc_bias, 
                             targ_em_weight, begin_state=encoder_states, layout='TNC', merge_outputs=True)
        else:
            outputs, _ = train_decoder_unroll(decoder, encoder_outputs, targ_embed, targ_vocab, dec_seq_len, 0, fc_weight, fc_bias,
                             attention_fc_weight, attention_fc_bias, 
                             targ_em_weight, begin_state=encoder_states, layout='TNC', merge_outputs=True)

        # NEW
        rs = mx.sym.Reshape(outputs, shape=(-1, args.num_hidden), name='sym_gen_reshape1')
        fc = mx.sym.FullyConnected(data=rs, weight=fc_weight, bias=fc_bias, num_hidden=len(targ_vocab), name='sym_gen_fc')
        label_rs = mx.sym.Reshape(data=label, shape=(-1,), name='sym_gen_reshape2')
        pred = mx.sym.SoftmaxOutput(data=fc, label=label_rs, name='sym_gen_softmax')

        return pred, ('src_data', 'targ_data',), ('softmax_label',)


#    foo, _, _ = sym_gen((1, 1))
#    print(type(foo))
#    mx.viz.plot_network(symbol=foo).save('./seq2seq.dot')


    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule( 
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            [encoder, decoder], args.model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    opt_params['clip_gradient'] = args.max_grad_norm

    start = time()

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = mx.metric.Perplexity(invalid_label),
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = opt_params, 
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = args.load_epoch,
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(batch_size=args.batch_size, frequent=args.disp_batches, auto_reset=True),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint([encoder, decoder], args.model_prefix, 1)
                              if args.model_prefix else None)

    train_duration = time() - start
    time_per_epoch = train_duration / args.num_epochs
    print("\n\nTime per epoch: %.2f seconds\n\n" % time_per_epoch)


def drop_sentinels(text_lst):
    sentinels = lambda x: x == reserved_tokens['<PAD>'] or x == reserved_tokens['<GO>']
    text_lst = dropwhile(lambda x: sentinels(x), text_lst)
    text_lst = takewhile(lambda x: not sentinels(x) and x != reserved_tokens['<EOS>'], text_lst)
    return list(text_lst)


class BleuScore(mx.metric.EvalMetric):
    def __init__(self, ignore_label, axis=-1):
        super(BleuScore, self).__init__('BleuScore')
        self.ignore_label = ignore_label
        self.axis = axis

    def update(self, labels, preds):
        assert len(labels) == len(preds)

        smoothing_fn = nltk.translate.bleu_score.SmoothingFunction().method3

        for label, pred in zip(labels, preds):
            maxed = mx.ndarray.argmax(data=pred, axis=1)
            pred_nparr = maxed.asnumpy()
            label_nparr = label.asnumpy().astype(np.int32) 
            sent_len, batch_size = np.shape(label_nparr)
            pred_nparr = pred_nparr.reshape(sent_len, batch_size).astype(np.int32)

            for i in range(batch_size):
                exp_lst = drop_sentinels(label_nparr[:, i].tolist())
                act_lst = drop_sentinels(pred_nparr[:, i].tolist())
                expected = exp_lst
                actual = act_lst
                bleu = nltk.translate.bleu_score.sentence_bleu(
                    references=[expected], hypothesis=actual, weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function = smoothing_fn 
                )
#                print("bleu: %f" % bleu)
                self.sum_metric += bleu
                self.num_inst += 1
            assert label.size == pred.size/pred.shape[-1], \
                "shape mismatch: %s vs. %s"%(label.shape, pred.shape)

    def get(self):
        num = self.num_inst if self.num_inst > 0 else float('nan')
        return (self.name, self.sum_metric/num)


def infer(args):
    assert args.model_prefix, "Must specifiy path to load from"

    data_test, src_vocab, inv_src_vocab, targ_vocab, inv_targ_vocab = get_data('TN', infer=True)

    print "len(src_vocab) len(targ_vocab)", len(src_vocab), len(targ_vocab)

    attention_fc_weight = mx.sym.Variable('attention_fc_weight')
    attention_fc_bias = mx.sym.Variable('attention_fc_bias')

    fc_weight = mx.sym.Variable('fc_weight')
    fc_bias = mx.sym.Variable('fc_bias')
    targ_em_weight = mx.sym.Variable('targ_embed_weight')

    if args.use_cudnn_cells:
        encoder = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, dropout=args.dropout,
            mode='lstm', prefix='lstm_encoder', bidirectional=args.bidirectional, get_next_state=True).unfuse()

    else:
        encoder = SequentialRNNCell()

        for i in range(args.num_layers):
            if args.bidirectional:
                encoder.add(
                    BidirectionalCell(
                        LSTMCell(args.num_hidden // 2, prefix='rnn_encoder_f%d_' % i),
                        LSTMCell(args.num_hidden // 2, prefix='rnn_encoder_b%d_' % i)))
                if i < args.num_layers - 1 and args.dropout > 0.0:
                    encoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_encoder%d_' % i))
            else:
                encoder.add(
                    LSTMCell(args.num_hidden, prefix='rnn_encoder%d_' % i))
                if i < args.num_layers - 1 and args.dropout > 0.0:
                    encoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_encoder%d_' % i))

    if args.use_cudnn_cells:
        decoder = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, 
            mode='lstm', prefix='lstm_decoder', bidirectional=args.bidirectional, get_next_state=True).unfuse()
 
    else:
        decoder = mx.rnn.SequentialRNNCell()

        for i in range(args.num_layers):
            decoder.add(LSTMCell(args.num_hidden, prefix=('rnn_decoder%d_' % i)))
            if i < args.num_layers - 1 and args.dropout > 0.0:
                decoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_decoder%d_' % i))

    def sym_gen(seq_len):
        src_data = mx.sym.Variable('src_data')
        targ_data = mx.sym.Variable('targ_data')
        label = mx.sym.Variable('softmax_label')
 
        src_embed = mx.sym.Embedding(data=src_data, input_dim=len(src_vocab), 
                                 output_dim=args.num_embed, name='src_embed') 
        targ_embed = mx.sym.Embedding(data=targ_data, input_dim=len(targ_vocab),
                                 weight = targ_em_weight,    # data=data
                                 output_dim=args.num_embed, name='targ_embed')

        encoder.reset()
        decoder.reset()

        enc_seq_len, dec_seq_len = seq_len

        layout = 'TNC'
        encoder_outputs, encoder_states = encoder.unroll(enc_seq_len, inputs=src_embed, layout=layout)

        # This should be based on EOS or max seq len for inference, but here we unroll to the target length
        # TODO: fix <GO> symbol
#        outputs, _ = decoder.unroll(dec_seq_len, targ_embed, begin_state=states, layout=layout, merge_outputs=True)
        outputs, _ = infer_decoder_unroll(decoder, encoder_outputs, targ_embed, targ_vocab, dec_seq_len, 0,
                     fc_weight, fc_bias, 
                     attention_fc_weight, attention_fc_bias,
                     targ_em_weight,
                     begin_state=encoder_states, layout='TNC', merge_outputs=True)

        # NEW

        rs = mx.sym.Reshape(outputs, shape=(-1, args.num_hidden), name='sym_gen_reshape1')
        fc = mx.sym.FullyConnected(data=rs, weight=fc_weight, bias=fc_bias, num_hidden=len(targ_vocab), name='sym_gen_fc')
        label_rs = mx.sym.Reshape(data=label, shape=(-1,), name='sym_gen_reshape2')
        pred = mx.sym.SoftmaxOutput(data=fc, label=label_rs, name='sym_gen_softmax')

#        rs = mx.sym.Reshape(outputs, shape=(-1, args.num_hidden), name='sym_gen_reshape1')
#        fc = mx.sym.FullyConnected(data=rs, num_hidden=len(targ_vocab), name='sym_gen_fc')
#        label_rs = mx.sym.Reshape(data=label, shape=(-1,), name='sym_gen_reshape2')
#        pred = mx.sym.SoftmaxOutput(data=fc, label=label_rs, name='sym_gen_softmax')

        return pred, ('src_data', 'targ_data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule( 
        sym_gen             = sym_gen,
        default_bucket_key  = data_test.default_bucket_key,
        context             = contexts)

    model.bind(data_test.provide_data, data_test.provide_label, for_training=False)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            [encoder, decoder], args.model_prefix, args.load_epoch)
#        print(arg_params)
        model.set_params(arg_params, aux_params)

    else:
        arg_params = None
        aux_params = None


    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    opt_params['clip_gradient'] = args.max_grad_norm

    start = time()

    # mx.metric.Perplexity
#    model.score(data_test, BleuScore(invalid_label), #mx.metric.Perplexity(invalid_label),
#                batch_end_callback=mx.callback.Speedometer(batch_size=args.batch_size, frequent=1, auto_reset=True))

    examples = []
    bleu_acc = 0.0
    num_inst = 0

    try:
        data_test.reset()

        smoothing_fn = nltk.translate.bleu_score.SmoothingFunction().method3

        while True:

            data_batch = data_test.next()
            model.forward(data_batch, is_train=None)
            source = data_batch.data[0]
            preds = model.get_outputs()[0]
            labels = data_batch.label[0]

            maxed = mx.ndarray.argmax(data=preds, axis=1)
            pred_nparr = maxed.asnumpy()
            src_nparr = source.asnumpy()
            label_nparr = labels.asnumpy().astype(np.int32)
            sent_len, batch_size = np.shape(label_nparr)
            pred_nparr = pred_nparr.reshape(sent_len, batch_size).astype(np.int32)

            for i in range(batch_size):

                src_lst = list(reversed(drop_sentinels(src_nparr[:, i].tolist())))
                exp_lst = drop_sentinels(label_nparr[:, i].tolist())
                act_lst = drop_sentinels(pred_nparr[:, i].tolist())

                expected = exp_lst
                actual = act_lst
                bleu = nltk.translate.bleu_score.sentence_bleu(
                    references=[expected], hypothesis=actual, weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function = smoothing_fn 
                )
                bleu_acc += bleu
                num_inst += 1
                examples.append((src_lst, exp_lst, act_lst, bleu))

    except StopIteration as se:
        pass
    
    bleu_acc /= num_inst

    # Find the top K best translations
    examples = sorted(examples, key=itemgetter(3), reverse=True) 

    num_examples = 20

    print("\nSample translations:\n")
    for i in range(min(num_examples, len(examples))):
        src_lst, exp_lst, act_lst, bleu = examples[i]
        src_txt = array_to_text(src_lst, data_test.inv_src_vocab)
        exp_txt = array_to_text(exp_lst, data_test.inv_targ_vocab) 
        act_txt = array_to_text(act_lst, data_test.inv_targ_vocab) 
        print("\n")
        print("Source text: %s" % src_txt)
        print("Expected translation: %s" % exp_txt)
        print("Actual translation: %s" % act_txt)
    print("\nTest set BLEU score (averaged over all examples): %.3f\n" % bleu_acc)

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()

    if args.input_feed:
        assert (args.attention == True), "--input-feed is legal only with --attention!"

    # set random seeds for Python, NumPy and MxNet
    import random
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)
    print("Using seed: %d" % seed)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)
   
    print("\n") 

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.infer:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        infer(args)
    else:
        if args.inference_unrolling_for_training:
            print("INFO: Using inference decoder unrolling for training")
        else:
            print("INFO: Using regular decoder unrolling for training")
        train(args)
