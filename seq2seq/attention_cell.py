# Eric Xie's (@piiswrong's) implementation

import mxnet as mx
from mxnet import symbol

def _normalize_sequence(length, inputs, layout, merge, in_layout=None):
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
        inputs = symbol.swapaxes(inputs, dim1=axis, dim2=in_axis)

    return inputs, axis


class AttentionEncoderCell(mx.rnn.BaseRNNCell):
    """Place holder cell that prepare input for attention decoders"""
    def __init__(self, prefix='encode_', params=None):
        super(AttentionEncoderCell, self).__init__(prefix, params=params)

    @property
    def state_shape(self):
        return []

    @property
    def state_info(self):
        return [{'shape': (), '__layout__': 'NT'}]

    def __call__(self, inputs, states):
        return inputs, states + [symbol.expand_dims(inputs, axis=1)]

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        outputs = _normalize_sequence(length, inputs, layout, merge_outputs)
        if merge_outputs is True:
            states = outputs
        else:
            states = inputs

        # attention cell always use NTC layout for states
        states, _ = _normalize_sequence(None, states, 'NTC', True, layout)
        return outputs, [states]



def _attention_pooling(source, scores):
    # source: (batch_size, seq_len, encoder_num_hidden)
    # scores: (batch_size, seq_len, 1)
    probs = symbol.softmax(scores, axis=1)
    output = symbol.batch_dot(source, probs, transpose_a=True)
    return symbol.reshape(output, shape=(0, 0))


class BaseAttentionCell(mx.rnn.BaseRNNCell):
    """Base class for attention cells"""
    def __init__(self, prefix='att_', params=None):
        super(BaseAttentionCell, self).__init__(prefix, params=params)

    @property
    def state_shape(self):
        return [(0, 0, 0)]

    def __call__(self, inputs, states):
        raise NotImplementedError


class DotAttentionCell(BaseAttentionCell):
    """Dot attention"""
    def __init__(self, prefix='dotatt_', params=None):
        super(DotAttentionCell, self).__init__(prefix, params=params)

    def __call__(self, inputs, states):
        # inputs: (batch_size, decoder_num_hidden)
        # for dot attention decoder_num_hidden must equal encoder_num_hidden
        if len(states) > 1:
            states = [symbol.concat(*states, dim=1)]

        # source: (batch_size, seq_len, encoder_num_hidden)
        source = states[0]
        # (batch_size, decoder_num_hidden, 1)
        inputs = symbol.expand_dims(inputs, axis=2)
        # (batch_size, seq_len, 1)
        scores = symbol.batch_dot(source, inputs)
        # (batch_size, encoder_num_hidden)
        return _attention_pooling(source, scores), states

    @property
    def state_shape(self):
        return []

    @property
    def state_info(self):
        return [{'shape': (), '__layout__': 'NT'}]

