"""
This file generates synthetic tree data in a sequential fashion.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import time

import dill as pickle
import numpy as np
import pandas as pd
import ray

from tqdm import tqdm
from syn_net.utils.tree import SyntheticTreeSet
from syn_net.utils.prep import generate_synthetic_tree
from syn_net.utils.utils import Action


@ray.remote
def generate_synthetic_tree_(building_blocks, rxns, max_steps: int = 15):
    return generate_synthetic_tree(building_blocks, rxns, max_steps)


if __name__ == '__main__':
    try:
        ray.init("auto")
    except ConnectionError:
        ray.init()

    parser = ArgumentParser()
    parser.add_argument("-r", "--reactions-pkl", type=Path)
    parser.add_argument("-bb", "--building-blocks", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    np.random.seed(6)

    building_blocks = pd.read_csv(args.building_blocks)['SMILES'].tolist()
    rxns = pickle.loads(args.reactions_pkl.read_bytes())

    MAX_STEPS = 15
    NUM_TRIALS = 600000

    building_blocks = ray.put(building_blocks)
    rxns = ray.put(rxns)
    
    t = time()
    refs = [
        generate_synthetic_tree_.remote(building_blocks, rxns, MAX_STEPS) for _ in range(NUM_TRIALS)
    ]
    results = [ray.get(r) for r in tqdm(refs)]
    print('Time: ', time() - t, 's')

    trees = [t for t, a in results if a == Action.END]
    actions = [a for _, a in results]

    num_finish = actions.count(Action.END)
    num_error = actions.count(-1)
    num_unfinish = NUM_TRIALS - num_finish - num_error

    m = 20
    print(f'{"Total trials":>{m}}: {NUM_TRIALS:>4}')
    print(f'{"num of finished trees":>{m}}: {num_finish:>4}')
    print(f'{"num of unfinished tree":>{m}}: {num_unfinish:>4}')
    print(f'{"num of error processes":>{m}}: {num_error:>4}')

    tree_set = SyntheticTreeSet(trees)
    tree_set.save(args.output)