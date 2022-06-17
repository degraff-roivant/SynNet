"""
This file generates synthetic tree data in a sequential fashion.
"""
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from syn_net.utils.tree import SyntheticTreeSet
from syn_net.utils.prep import generate_synthetic_tree
from syn_net.utils.utils import Action


if __name__ == '__main__':
    # path_reaction_file = '/home/whgao/shared/Data/scGen/reactions_pis.pickle.gz'
    # path_to_building_blocks = '/home/whgao/shared/Data/scGen/enamine_building_blocks_nochiral_matched.csv.gz'

    parser = ArgumentParser()
    parser.add_argument("-r", "--reactions-pkl", type=Path)
    parser.add_argument("-bb", "--building-blocks", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    np.random.seed(6)

    building_blocks = pd.read_csv(args.building_blocks)['SMILES'].tolist()
    rxns = pickle.loads(args.reactions_pkl.read_bytes())
    # with gzip.open(path_reaction_file, 'rb') as f:
    #     rxns = pickle.load(f)

    T = 5
    MAX_STEPS = 15

    num_finish = 0
    num_error = 0
    num_unfinish = 0
    trees = []
    for _ in tqdm(range(T)):
        tree, action = generate_synthetic_tree(building_blocks, rxns, MAX_STEPS)
        if action == Action.END:
            trees.append(tree)
            num_finish += 1
        elif action == -1:
            num_error += 1
        else:
            num_unfinish += 1
    
    m = 20
    print(f'{"Total trials":>{m}}: {T:>4}')
    print(f'{"num of finished trees":>{m}}: {num_finish:>4}')
    print(f'{"num of unfinished tree":>{m}}: {num_unfinish:>4}')
    print(f'{"num of error processes":>{m}}: {num_error:>4}')

    synthetic_tree_set = SyntheticTreeSet(trees)
    synthetic_tree_set.save(args.output)

    # data_file = gzip.open('st_data.pickle.gz', 'wb')
    # pickle.dump(trees, data_file)
    # data_file.close()
