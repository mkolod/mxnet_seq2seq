import random

if __name__ == '__main__':

    random.seed(42)
    train_valid_split = 0.2

    ct_file = 'europarl-v7.es-en.en'
    with open(ct_file, 'r') as f:
        lines = f.read().split('\n') 
    count = len(lines) - 1
    print("Total example count: %d" % ct)

    # create list to be shuffled
    indices = list(range(count))
    random.shuffle(indices)

    # use NumPy here to get logical indexing
    train_examples = list(range(int(count * train_valid_split)))

    valid_examples = 
