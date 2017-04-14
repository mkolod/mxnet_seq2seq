from os import linesep
import random
from tqdm import tqdm

if __name__ == '__main__':

    random.seed(42)
    train_split = 0.8
    valid_split = 1.0 - train_split

    directory = './data/wmt15-de-en/'
    src_file = directory + 'all.en'
    targ_file = directory + 'all.de'

    def num_lines(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    num_lines_src = num_lines(src_file)
    num_lines_targ = num_lines(targ_file)
    assert num_lines_src == num_lines_targ, "Source and target language files have to have the same number of lines"

    fmt = lambda x: '{0:,}'.format(x)
    print("\nTotal example count: %s" % fmt(num_lines_src))
    num_train = int(num_lines_src * train_split)
    num_valid = int(num_lines_src * valid_split)
    print("A %.2f / %.2f split will give us %s training and %s validation examples\n" % (
        train_split, valid_split, fmt(num_train), fmt(num_valid)))

    indices = list(range(num_lines_src))
    random.shuffle(indices)

    train_indices = set(indices[:num_train])
    valid_indices = set(indices[num_train:])
 
    with open(src_file, 'r') as src_file, open(targ_file, 'r') as targ_file,        \
        open(directory + 'src_train', 'w') as src_train, open(directory + 'targ_train', 'w') as targ_train, \
        open(directory + 'src_valid', 'w') as src_valid, open(directory + 'targ_valid', 'w') as targ_valid:

        for idx in tqdm(range(num_lines_src), desc='Line of source/target data'):
            src_line = src_file.readline()
            targ_line = targ_file.readline()
            shuffled_idx = indices[idx]
            if shuffled_idx in train_indices:
                src_train.write(src_line + linesep)
                targ_train.write(targ_line + linesep)
            else:
                src_valid.write(src_line + linesep)
                targ_valid.write(targ_line +linesep)

print("\nDone\n")
