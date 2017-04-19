import numpy as np
import mxnet as mx
import argparse
import cPickle as pickle
#import dill as pickle

from time import time
import re
from unidecode import unidecode

from utils import array_to_text, tokenize_text, invert_dict, get_s2s_data, Dataset

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
    if len(args) == 0 or args[0] == "N/A":
        print("N/A")
    else:
        for i in range(len(args)):
            print("%s: %s" % (args[i], arg_shapes[i]))
    print("\n=============")
    print("aux_states:")
    print("=============")
    if len(aux_states) == 0 or aux_states[0] == "N/A":
        print("N/A")
    else:
        for i in range(min(len(aux_shapes), len(aux_states))):
            print("%s: %s" % (aux_states[i], aux_shapes[i]))
    print("\n=============")
    print("outputs:")
    print("==============")
    if len(out_shapes) == 0:
        print("N/A")
    else:
        for i in range(len(out_shapes)):
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
        print("length = %d, len(inputs) = %d" % (length, len(inputs))) 
        assert length is None or len(inputs) == length
        if merge is True:
            inputs = [symbol.expand_dims(i, axis=axis) for i in inputs]
            inputs = symbol.Concat(*inputs, dim=axis)
            in_axis = axis

    if isinstance(inputs, symbol.Symbol) and axis != in_axis:
        inputs = symbol.swapaxes(inputs, dim0=axis, dim1=in_axis)

    return inputs, axis

def get_data(layout):

    start = time()

    print("\nUnpickling training iterator")

    with open('./data/train_iterator.pkl', 'rb') as f:
        train_iter = pickle.load(f)
 
    train_iter.initialize()
    train_iter.batch_size = args.batch_size

    print("\nUnpickling validation iterator")

    with open('./data/valid_iterator.pkl', 'rb') as f:
        valid_iter = pickle.load(f)
 
    valid_iter.initialize()
    valid_iter.batch_size = args.batch_size

#    nextb = train_iter.next()
#    print(type(nextb))
#    print(dir(nextb))
#    print(nextb.data)
#    print(nextb.label)
    

    print("\nEncoded source language sentences:\n")
    for i in range(5):
        print(array_to_text(train_iter.src_sent[i], train_iter.inv_src_vocab))

    print("\nEncoded target language sentences:\n")
    for i in range(5):
        print(array_to_text(valid_iter.targ_sent[i], train_iter.inv_targ_vocab))
    

    duration = time() - start

    print("\nDataset deserialization time: %.2f seconds\n" % duration)



#    train_iter.save('./data/train_iterator.pkl')

#    train_iter = Seq2SeqIter.load('./data/train_iterator.pkl')

#    valid_iter = Seq2SeqIter.load('./data/valid_iterator.pkl')

#    print("source dict size: %d" % len(train_iter.src_dict))
#    print("targ dict size: %d" % len(train_iter.targ_dict))


#    duration = time() - start

#    print("\nDeserializing training and validation iterators took %.2f seconds\n" % duration)

#    print("\nSize of src vocab: %d" % len(train_iter.src_vocab))
#    print("Size of targ vocab: %d\n" % len(train_iter.targ_vocab))

    return train_iter, valid_iter, train_iter.src_vocab, train_iter.targ_vocab

# WORK IN PROGRESS !!!
def decoder_unroll(decoder, target_embed, targ_vocab, unroll_length, go_symbol, begin_state=None, layout='TNC', merge_outputs=None):

        decoder.reset()

        if begin_state is None:
            begin_state = decoder.begin_state()

        inputs, _ = _normalize_sequence(unroll_length, target_embed, layout, False)

        # Need to use hidden state from attention model, but <GO> as input
        outputs = []

        output, states = decoder(inputs[0], begin_state)
        pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden), name='output_reshape') 
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(targ_vocab), name='pred')
#        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
        output = mx.sym.argmax(pred, axis=1, keepdims=False, name='argmax')

        feed = mx.sym.var('targ_data')

        targ_embed_weight = mx.sym.var('targ_embedding_weight')
        embed1 = mx.sym.Embedding(data=feed, input_dim=len(targ_vocab), 
            output_dim=args.num_embed, weight=targ_embed_weight, name='infer_embed')

        outs2 = []
        states2 = []

#        out, state = decoder(feed, begin_state)
        out = None 

        for i in range(unroll_length): #unroll_length - 1):
            a = out if i > 0 else feed
            state = begin_state if i == 0 else state
            embed = mx.sym.Embedding(data=a, input_dim=len(targ_vocab), 
                output_dim=args.num_embed, weight=targ_embed_weight, name='infer_embed_%d' % i)
            out, state = decoder(embed, state)
            outs2.append(out)
            states2.append(state)
            pred = mx.sym.Reshape(out, shape=(-1, args.num_hidden), name='output_reshape') 
            pred = mx.sym.FullyConnected(data=pred, num_hidden=len(targ_vocab), name='pred')
#        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
            out = mx.sym.argmax(pred, axis=1, keepdims=False, name='argmax')

#        print(outs2)
#        var = outs2[-1]
#        arg_shapes, out_shapes, aux_shapes = var.infer_shape_partial() #targ_data=mx.nd.array(np.array([1, 2, 3])))
#        print_inferred_shapes(var, arg_shapes, out_shapes, aux_shapes)
       

#        src_embed = mx.sym.Embedding(data=src_data, input_dim=len(src_vocab), 
#                                 output_dim=args.num_embed, name='src_embed') 
#        targ_embed = mx.sym.Embedding(data=targ_data, input_dim=len(targ_vocab),    # data=data
#                                 output_dim=args.num_embed, name='targ_embed')







#        output, _ = _normalize_sequence(1, output2, layout, merge_outputs)

#        var = target_embed
#        arg_shapes, out_shapes, aux_shapes = var.infer_shape_partial(targ_data=(55,1)) #src_data=(55,1), targ_data=(55,1))
#        print_inferred_shapes(var, arg_shapes, out_shapes, aux_shapes)


 #       output2, states = decoder(output2, states)
  #      arg_shapes, out_shapes, aux_shapes = output2.infer_shape_partial(src_data=(55,1), targ_data=(55,1))
   #     print_inferred_shapes(output2, arg_shapes, out_shapes, aux_shapes)

#        outputs, _ = _normalize_sequence(1, outputs, layout, merge_outputs)

#        pred_word_idx = mx.sym.Variable('pred_word_idx')

#        embed = mx.sym.Embedding(data=pred_word_idx, input_dim=len(targ_vocab),
#            output_dim=args.num_embed, name='src_embed') 

#	embed_mat = mx.nd.array(np.random.randn(50004, 500))
 
#        for i in range(0, unroll_length):
#            print("i = %d" % i)
#            bound = target_embed.bind(mx.cpu(), {'targ_data': mx.nd.array([0]), 'targ_embed_weight': embed_mat})
#            output, states = (bound, states)
#            outputs.append(output)   

            # this works


        for i in range(unroll_length):
            output, states = decoder(inputs[i], states)
            outputs.append(output)
#                print(target_embed.infer_shape_partial())
#                src_data = mx.sym.Variable('src_data')
#                targ_data = mx.sym.Variable('targ_data')
#                label = mx.sym.Variable('softmax_label')


                 
#          arg_shapes, out_shapes, aux_shapes = output.infer_shape(src_data=)
#                print_inferred_shapes(output, arg_shapes, out_shapes, aux_shapes)

        outputs, _ = _normalize_sequence(unroll_length, outputs, layout, merge_outputs)
 
        outs2, _ = _normalize_sequence(unroll_length, outs2, layout, merge_outputs)       
 
        return outs2, states2
 
#        return outputs, states

def train(args):

    from time import time

    data_train, data_val, src_vocab, targ_vocab = get_data('TN')

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

    def sym_gen(seq_len):
        src_data = mx.sym.Variable('src_data')
        targ_data = mx.sym.Variable('targ_data') #'targ_data')
        label = mx.sym.Variable('softmax_label')
 
        src_embed = mx.sym.Embedding(data=src_data, input_dim=len(src_vocab), 
                                 output_dim=args.num_embed, name='src_embed') 
        targ_embed = mx.sym.Embedding(data=targ_data, input_dim=len(targ_vocab),    # data=data
                                 output_dim=args.num_embed, name='targ_embed')

#        x = targ_embed.bind(mx.cpu(), {'targ_data': mx.nd.array([0, 1, 2]), 'targ_embed_weight': mx.nd.array(np.random.randn(50004, 500))})
#        y = x.forward()
#        z = y[0].asnumpy()
#        print(np.shape(z))

        encoder.reset()
        decoder.reset()

        enc_seq_len, dec_seq_len = seq_len

        layout = 'TNC'
        _, states = encoder.unroll(enc_seq_len, inputs=src_embed, layout=layout)


        # decoder.unroll
        #decoder_unroll(decoder, target_embed, targ_vocab, unroll_length, go_symbol, begin_state=None, layout='TNC', merge_outputs=None):
        outputs, _ = decoder_unroll(decoder, targ_embed, targ_vocab, dec_seq_len, '<GO>', begin_state=states, layout=layout, merge_outputs=True)

#        outputs, _ = decoder_unroll(dec_seq_len, inputs=targ_embed, begin_state=states, layout=layout, merge_outputs=True)

        # This should be based on EOS or max seq len for inference, but here we unroll to the target length
        # TODO: fix <GO> symbol
#        outputs2, _ = decoder_unroll(decoder, targ_embed, targ_vocab, dec_seq_len, 0, begin_state=states, layout='TNC', merge_outputs=True)
      


        pred = mx.sym.Reshape(outputs,
                shape=(-1, args.num_hidden)) # -1
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(targ_vocab), name='pred')
#        print(pred)
#        print(dir(pred))
#        partial = pred.infer_shape_partial()
#        print(partial)
#        print(dir(partial))
#        print(type(partial))

        label = mx.sym.Reshape(data=label, shape=(-1,))

        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('src_data', 'targ_data',), ('softmax_label',)


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
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(decoder, args.model_prefix, 1)
                              if args.model_prefix else None)

    train_duration = time() - start
    time_per_epoch = train_duration / args.num_epochs
    print("\n\nTime per epoch: %.2f seconds\n\n" % time_per_epoch)

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
