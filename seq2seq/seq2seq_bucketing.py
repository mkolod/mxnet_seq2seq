import numpy as np
import mxnet as mx
import argparse

import re
from unidecode import unidecode

from common import tokenize_text, invert_dict, get_s2s_data, Dataset

from seq2seq_iterator import *

from attention_cell import AttentionEncoderCell, DotAttentionCell

from seq2seq_bucketing_module import Seq2SeqBucketingModule

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

def get_data2(layout):

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

    train_iter = Seq2SeqIter(train_dataset, layout=layout, batch_size=args.batch_size, buckets=all_pairs)
    valid_iter = Seq2SeqIter(valid_dataset, layout=layout, batch_size=args.batch_size, buckets=all_pairs)
    train_iter.reset()
    valid_iter.reset()
    
    print("\nSize of src vocab: %d" % len(train_iter.src_vocab))
    print("Size of targ vocab: %d\n" % len(train_iter.targ_vocab))

    return train_iter, valid_iter, train_iter.src_vocab, train_iter.targ_vocab

def get_data(layout):
    source_data = "./data/europarl-v7.es-en.es_train_small"
    target_data = "./data/europarl-v7.es-en.en_train_small"
    train_sent, vocab = tokenize_text(source_data, start_label=start_label,
                                      invalid_label=invalid_label)
    val_sent, _ = tokenize_text(target_data, vocab=None, start_label=start_label, # vocab, start_label=start_label,
                                invalid_label=invalid_label)
 
    data_train  = mx.rnn.BucketSentenceIter(train_sent, args.batch_size, buckets=buckets,
                                            invalid_label=invalid_label, layout=layout)
    data_val    = mx.rnn.BucketSentenceIter(val_sent, args.batch_size, buckets=buckets,
                                            invalid_label=invalid_label, layout=layout)
    return data_train, data_val, vocab


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

def train(args):

    data_train, data_val, src_vocab, targ_vocab = get_data2('TN')

#    data_train, data_val, src_vocab = get_data('TNC') #TN')
#    targ_vocab = src_vocab

#    print(data_train.next())
#    print(data_val.next()) 

    print("Dict size: %d" % len(src_vocab))

    encoder = mx.rnn.SequentialRNNCell()

    for i in range(args.num_layers):
        encoder.add(mx.rnn.LSTMCell(args.num_hidden, prefix='rnn_encoder%d_' % i))
        if i < args.num_layers - 1 and args.dropout > 0.0:
            encoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_encoder%d_' % i))
    encoder.add(AttentionEncoderCell())

    decoder = mx.rnn.SequentialRNNCell()
    for i in range(args.num_layers):
        decoder.add(mx.rnn.LSTMCell(args.num_hidden, prefix=('rnn_decoder%d_' % i)))
        if i < args.num_layers - 1 and args.dropout > 0.0:
            decoder.add(mx.rnn.DropoutCell(args.dropout, prefix='rnn_decoder%d_' % i))
    decoder.add(DotAttentionCell())

    ##############################
    # Remove this after debugging
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    src_embed = mx.sym.Embedding(data=data, input_dim=len(src_vocab),
				 output_dim=args.num_embed, name='src_embed') 
    targ_embed = mx.sym.Embedding(data=label, input_dim=len(targ_vocab),
				 output_dim=args.num_embed, name='targ_embed')

    encoder.reset()
    decoder.reset()

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        src_embed = mx.sym.Embedding(data=data, input_dim=len(src_vocab),
                                 output_dim=args.num_embed, name='src_embed') 
        targ_embed = mx.sym.Embedding(data=label, input_dim=len(targ_vocab),
                                 output_dim=args.num_embed, name='targ_embed')

        encoder.reset()
        decoder.reset()

#        enc_seq_len = seq_len
#        dec_seq_len = seq_len
        enc_seq_len, dec_seq_len = seq_len

        layout = 'TNC'
        _, states = encoder.unroll(enc_seq_len, inputs=src_embed, layout=layout)


        outputs, _ = decoder.unroll(dec_seq_len, inputs=targ_embed, begin_state=states, layout=layout, merge_outputs=True)

        # This should be based on EOS or max seq len for inference, but here we unroll to the target length
        # TODO: fix <GO> symbol
        print("dec_seq_len: %d" % dec_seq_len)
#        outputs, _ = decoder_unroll(decoder, targ_embed, targ_vocab, dec_seq_len, 0, begin_state=states, layout='TNC', merge_outputs=True)

        pred = mx.sym.Reshape(outputs,
                shape=(-1, args.num_hidden)) # -1
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(targ_vocab), name='pred')
#        print(pred)
#        print(dir(pred))
#        print(pred.infer_shape_partial())

#        pred = mx.sym.Reshape(pred, shape=(enc_seq_len, 32, args.num_hidden))
 #       label = mx.sym.Reshape(label, shape=(enc_seq_len, 32))
#        pred = mx.sym.Reshape(data=pred, shape=(-1,))
        label = mx.sym.Reshape(data=label, shape=(-1,))

        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)


    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    # mx.mod.BucketingModule
    model = Seq2SeqBucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            cell, args.model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

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
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(decoder, args.model_prefix, 1)
                              if args.model_prefix else None)

def test(args):
    assert args.model_prefix, "Must specifiy path to load from"
    _, data_val, vocab = get_data('TN') # NT

    encoder = mx.rnn.SequentialRNNCell()
    encoder.add(mx.rnn.LSTMCell(args.num_hidden, prefix='rnn_encoder0_'))
    encoder.add(mx.rnn.AttentionEncoderCell())

    decoder = mx.rnn.SequentialRNNCell()
    decoder.add(mx.rnn.LSTMCell(args.num_hidden, prefix='rnn_decoder0_'))
    decoder.add(mx.rnn.DotAttentionCell())

#    encoder_data = mx.sym.Variable('encoder_data')
#    decoder_data = mx.sym.Variable('decoder_data')

    # assert outputs.list_outputs() == ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']
    # args, outs, auxs = outputs.infer_shape(encoder_data=(10, 3, 50), decoder_data=(10, 3, 50))

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        print(data.asnumpy())
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=args.num_embed, name='embed')

        encoder.reset()
        decoder.reset()

        _, states = encoder.unroll(seq_len, inputs=embed)
        outputs, _ = decoder.unroll(seq_len, inputs=embed, begin_state=states)
        outputs = mx.sym.Group(outputs)
        print(type(outputs[0]))

#        args, outs, auxs = outputs.infer_shape(encoder_data=(10, 3, 50), decoder_data=(10, 3, 50))
#        print("args: %s" % args)
#        print("outs: %s" % outs)
#        print("auxs: %s" % auxs)

        pred = mx.sym.Reshape(outputs,
                shape=(-1, args.num_hidden*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

#    model = mx.mod.BucketingModule(
#        sym_gen             = sym_gen,
#        default_bucket_key  = data_val.default_bucket_key,
#        context             = contexts)
#    model.bind(data_val.provide_data, data_val.provide_label, for_training=False)

    # note here we load using SequentialRNNCell instead of FusedRNNCell.
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, args.model_prefix, args.load_epoch)
    model.set_params(arg_params, aux_params)

    model.score(data_val, mx.metric.Perplexity(invalid_label),
                batch_end_callback=mx.callback.Speedometer(args.batch_size, 5))

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()
    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)
    

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.test:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        test(args)
    else:
        train(args)
