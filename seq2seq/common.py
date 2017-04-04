import mxnet as mx
import re
from unidecode import unidecode
from collections import namedtuple

Dataset = namedtuple(
    'Dataset',
    ['src_sent', 'src_vocab', 'inv_src_vocab', 'targ_sent', 'targ_vocab', 'inv_targ_vocab'])

# Decode text as UTF-8
# Remove diacritical signs and convert to Latin alphabet
# Separate punctuation as separate "words"
def tokenize_text(fname, vocab=None, invalid_label=0, start_label=1, sep_punctuation=True):
    lines = unidecode(open(fname).read().decode('utf-8')).split('\n')
    lines = map(lambda x: re.findall(r"\w+|[^\w\s]", x, re.UNICODE), lines)    
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)
    return sentences, vocab

def invert_dict(d):
    return {v: k for k, v in d.iteritems()}

def get_s2s_data(src_path, targ_path, start_label=1, invalid_label=0, pad_symbol='<PAD>'):
	src_sent, src_vocab = tokenize_text(src_path, start_label=start_label,
								invalid_label=invalid_label)
		
	src_vocab[pad_symbol] = invalid_label
	inv_src_vocab = invert_dict(src_vocab)

	targ_sent, targ_vocab = tokenize_text(targ_path, start_label=start_label, 
										  invalid_label=invalid_label)
		
	targ_vocab[pad_symbol] = invalid_label
	inv_targ_vocab = invert_dict(targ_vocab)
		
	return Dataset(
		src_sent=src_sent, src_vocab=src_vocab, inv_src_vocab=inv_src_vocab,
		targ_sent=targ_sent, targ_vocab=targ_vocab, inv_targ_vocab=inv_targ_vocab)

