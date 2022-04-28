"""
This file contains a function to decode a single synthetic tree.
"""
import pandas as pd
import numpy as np
from dgllife.model import load_pretrained
from syn_net.utils.data_utils import ReactionSet
from syn_net.utils.predict_utils import synthetic_tree_decoder, tanimoto_similarity, load_modules_from_checkpoint
from syn_net.parameters.args import paths, parameters


# define model to use for molecular embedding
model_type   = 'gin_supervised_contextpred'
device       = 'cpu'
mol_embedder = load_pretrained(model_type).to(device)
mol_embedder.eval()

# load the purchasable building block embeddings
bb_emb = np.load(paths['bb_emb'])

# define path to the reaction templates and purchasable building blocks
path_to_reaction_file   = paths['reaction_file']
path_to_building_blocks = paths['building_blocks']

# define paths to pretrained modules
path_to_act = paths['to_act']
path_to_rt1 = paths['to_rt1']
path_to_rxn = paths['to_rxn']
path_to_rt2 = paths['to_rt2']

# load the purchasable building block SMILES to a dictionary
building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
bb_dict         = {building_blocks[i]: i for i in range(len(building_blocks))}

# load the reaction templates as a ReactionSet object
rxn_set = ReactionSet()
rxn_set.load(path_to_reaction_file)
rxns    = rxn_set.rxns

# load the pre-trained modules
print("-------------------------")
print("parameters:", parameters)
print("paths:", paths)
act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
    path_to_act=path_to_act,
    path_to_rt1=path_to_rt1,
    path_to_rxn=path_to_rxn,
    path_to_rt2=path_to_rt2,
    featurize=parameters['featurize'],
    rxn_template=parameters['rxn_template'],
    out_dim=parameters['out_dim'],
    nbits=parameters['nbits'],
    ncpu=parameters['ncpu'],
)

def func(emb):
    """
    Generates the synthetic tree for the input molecular embedding.

    Args:
        emb (np.ndarray): Molecular embedding to decode.

    Returns:
        str: SMILES for the final chemical node in the tree.
        SyntheticTree: The generated synthetic tree.
    """
    emb = emb.reshape((1, -1))
    try:
        tree, action = synthetic_tree_decoder(z_target=emb,
                                              building_blocks=building_blocks,
                                              bb_dict=bb_dict,
                                              reaction_templates=rxns,
                                              mol_embedder=mol_embedder,
                                              action_net=act_net,
                                              reactant1_net=rt1_net,
                                              rxn_net=rxn_net,
                                              reactant2_net=rt2_net,
                                              bb_emb=bb_emb,
                                              rxn_template=parameters['rxn_template'],
                                              n_bits=parameters['nbits'],
                                              max_step=15)
    except Exception as e:
        print(e)
        action = -1
    if action != 3:
        return None, None
    else:
        scores = np.array(
            tanimoto_similarity(emb, [node.smiles for node in tree.chemicals])
        )
        max_score_idx = np.where(scores == np.max(scores))[0][0]
        return tree.chemicals[max_score_idx].smiles, tree
