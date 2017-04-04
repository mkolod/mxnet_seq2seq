import mxnet as mx
import re
from unidecode import unidecode

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
