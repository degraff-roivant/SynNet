"""
Filters out purchasable building blocks which don't match a single template.
"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from syn_net.utils.reaction import ReactionSet

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-r", "--reaction-set", type=Path)
    parser.add_argument("-bb", "--building-blocks", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="the filepath to store the filtered building blocks to")
    args = parser.parse_args()

    rs = ReactionSet.load(args.reaction_set)
    matched_mols = set.union(*[set(reactants) for r in rs for reactants in r.available_reactions])
    # r_path = '/pool001/whgao/data/synth_net/st_pis/reactions_pis.json.gz'
    # bb_path = '/home/whgao/scGen/synth_net/data/enamine_us.csv.gz'
    # rs.load(r_path)
    # matched_mols = set()
    # for r in tqdm(rs.rxns):
        # matched_mols |= set.union(*[set(reactants) for reactants in r.available_reactants])
        # for reactants in r.available_reactants:
        #     matched_mols = matched_mols | set(a_list)

    original_mols = pd.read_csv(args.building_blocks)['SMILES'].tolist()

    print('Total building blocks number:', len(original_mols))
    print('Matched building blocks number:', len(matched_mols))

    df = pd.DataFrame({'SMILES': list(matched_mols)})
    df.to_csv(args.output, compression='gzip')
