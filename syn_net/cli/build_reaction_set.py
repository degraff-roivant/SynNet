"""
This file processes a set of reaction templates and finds applicable
reactants from a list of purchasable building blocks.

Usage:
    python process_rxn_mp.py
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from time import time

import pandas as pd
import ray
import shutup

from syn_net.utils.reaction import Reaction, ReactionSet

shutup.please()


@ray.remote
def set_available_reactants(rxn: Reaction, building_blocks: list[str]):
    rxn.set_available_reactants(building_blocks)
    return rxn


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", "--templates", type=Path)
    parser.add_argument("-bb", "--building-blocks", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="where to save the output ReactionSet")
    args = parser.parse_args()

    try:
        ray.init("auto")
    except ConnectionError:
        ray.init()

    # name = 'pis'
    # path_to_rxn_templates = '/home/whgao/scGen/synth_net/data/rxn_set_' + name + '.txt'

    # p_templates = Path(path_to_rxn_templates)
    reactions = [l.split("|")[1] for l in args.templates.read_text().splitlines()]

    # path_to_building_blocks = '/home/whgao/scGen/synth_net/data/enamine_us.csv.gz'
    building_blocks = pd.read_csv(args.building_blocks)['SMILES'].tolist()

    # reactions = []
    # for line in open(path_to_rxn_templates, 'rt'):
    #     reaction = Reaction(line.split('|')[1].strip())
    #     reactions.append(reaction)

    t = time()
    refs = [set_available_reactants.remote(reaction, building_blocks) for reaction in reactions]
    reactions = ray.get(refs)
    print('Time: ', time() - t, 's')

    R = ReactionSet(reactions)
    R.save(args.output)
