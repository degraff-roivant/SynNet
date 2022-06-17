"""
Filters the synthetic trees by the QEDs of the root molecules.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from tdc import Oracle, oracles
import numpy as np
import pandas as pd
from syn_net.utils.data_utils import *

from syn_net.utils.tree import SyntheticTreeSet

def canonicalize(smi: str) -> Optional[str]:
    """canonicalize the input SMILES string. Return None if the SMILES string is invalid"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, isomericSmiles=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-st", "--synthetic-trees", type=Path)
    parser.add_argument("-r", "--reactions-pkl", type=Path)
    parser.add_argument("-bb", "--building-blocks", type=Path)
    parser.add_argument("--objective", choices=oracles.oracle_names, default="qed")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--filtered-smis", type=Path)
    args = parser.parse_args()
    
    # data_path = '/pool001/whgao/data/synth_net/st_pis/st_data.json.gz'
    sts = SyntheticTreeSet.load(args.synthetic_trees)
    # st_set.load(data_path)
    print(f'Finish reading, in total {len(sts)} synthetic trees.')

    obj = Oracle(args.objective)

    scores_all = []
    generated_smiles = []
    trees_filtered = []
    scores_filtered = []

    threshold = 0.5

    for t in sts:
        smi = canonicalize(t.root.smiles)
        if smi is None or smi in generated_smiles:
            continue

        score = obj(smi)
        scores_all.append(score)

        if score > args.threshold or np.random.random() < (score / args.threshold):
            generated_smiles.append(smi)
            trees_filtered.append(t)
            scores_filtered.append(score)

    print(f'Finish sampling, remaining {len(trees_filtered)} synthetic trees.')

    sts = SyntheticTreeSet(trees_filtered)
    sts.save(args.output)

    df = pd.DataFrame({'SMILES': generated_smiles, f'{obj.name}': scores_filtered})
    df.to_csv(args.filtered_smis, compression='gzip', index=False)

    print('Finish!')
