"""
Reads synthetic tree data and splits it into training, validation and testing sets.
"""
from argparse import ArgumentError, ArgumentParser
from pathlib import Path

import numpy as np

from syn_net.utils.tree import SyntheticTreeSet

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-st", "--synthetic-trees", type=Path)
    parser.add_argument("--split", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    parser.add_argument("-o", "--output", help="the base filename to save train/val/test data to. Each respective split will be saved to a file with the name 'OUTPUT_{train,val,test}.json.gz'")
    args = parser.parse_args()
    
    if not np.isclose(sum(args.split), 1):
        raise ArgumentError(f"train/val/test splits must sum to 1! got splits: {args.split}")

    print('Reading data from ', args.synthetic_trees)
    sts = SyntheticTreeSet.load(args.synthetic_trees)
    N = len(sts)
    print(f"Total synthetic trees: {len(sts)}")

    num_train = int(args.split[0] * N)
    num_valid = int(args.split[1] * N)
    num_test = N - num_train - num_valid

    sts_train = sts[:num_train]
    sts_valid = sts[num_train: num_train + num_valid]
    sts_test = sts[num_train + num_valid: ]

    print("Saving training dataset: ", len(sts_train))
    tree_set = SyntheticTreeSet(sts_train)
    tree_set.save(f'{args.output}_train.json.gz')

    print("Saving validation dataset: ", len(sts_valid))
    tree_set = SyntheticTreeSet(sts_valid)
    tree_set.save(f'{args.output}_valid.json.gz')

    print("Saving testing dataset: ", len(sts_test))
    tree_set = SyntheticTreeSet(sts_test)
    tree_set.save(f'{args.output}_test.json.gz')

    print("Finished!")
