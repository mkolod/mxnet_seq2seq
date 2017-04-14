from os import linesep
import random
from tqdm import tqdm
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Split training and validation data",
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-src-data-path', type=str, help='path to concatenated source corpus', required=True)
    parser.add_argument('--in-targ-data-path', type=str, help='path to concatenated target corpus', required=True)
    parser.add_argument('--out-src-train-path', type=str, help='path to source training corpus (to be created)', required=True)
    parser.add_argument('--out-targ-train-path', type=str, help='path to target training corpus (to be created)', required=True)
    parser.add_argument('--out-src-valid-path', type=str, help='path to source validation corpus (to be created)', required=True)
    parser.add_argument('--out-targ-valid-path', type=str, help='path to target validation corpus (to be created)', required=True)
    parser.add_argument('--validation-fraction', type=float, help='fraction of data to reserve for validation, > 0.0 and < 1.0', default=0.2)
    parser.add_argument('--shuffle-seed', type=int, help='random number generator seed for shuffling corpora before train/valid split (int)', default=42)
 
    args = parser.parse_args()


    valid_split = args.validation_fraction
    train_split = 1.0 - valid_split

    src_file = args.in_src_data_path
    targ_file = args.in_targ_data_path

    src_train = args.out_src_train_path
    src_valid = args.out_src_valid_path
    targ_train = args.out_targ_train_path
    targ_valid = args.out_targ_valid_path

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
        open(src_train, 'w') as src_train, open(targ_train, 'w') as targ_train, \
        open(src_valid, 'w') as src_valid, open(targ_valid, 'w') as targ_valid:

        for idx in tqdm(range(num_lines_src), desc='Line of source/target data'):
            src_line = src_file.readline()
            targ_line = targ_file.readline()
            shuffled_idx = indices[idx]
            if shuffled_idx in train_indices:
                src_train.write(src_line + linesep)
                targ_train.write(targ_line + linesep)
            else:
                src_valid.write(src_line + linesep)
                targ_valid.write(targ_line + linesep)

    print("\n")
