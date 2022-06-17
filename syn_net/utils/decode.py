from dgl.nn.pytorch.glob import AvgPooling
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
import numpy as np
from rdkit.Chem import AllChem as Chem
from sklearn.neighbors import BallTree
import torch
from syn_net.models.v2.utils import cosine_distance
from syn_net.utils.predict_utils import get_reaction_mask

from syn_net.utils.tree import SyntheticTree
from syn_net.utils.utils import Action


def mol_fp(smi, radius=2, length=4096):
    """
    Computes the Morgan fingerprint for the input SMILES.

    Args:
        smi (str): SMILES for molecule to compute fingerprint for.
        radius (int, optional): Fingerprint radius to use. Defaults to 2.
        length (int, optional): Length of fingerprint. Defaults to 4096.

    Returns:
        features (np.ndarray): For valid SMILES, this is the fingerprint.
            Otherwise, if the input SMILES is bad, this will be a zero vector.
    """
    x = np.zeros(length)

    if smi is None:
        return x

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return x

    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius, length)
    Chem.DataStructs.ConvertToNumpyArray(fp, x)

    return x


def nn_search(x, tree):
    """find the nearest neighbor to `x` in the input tree.

    Args:
        x (np.ndarray): An array of shape `1 x d` containing a specific point in the dataset.
        tree (BallTree): the tree to search.

    Returns:
        float: The distance to the nearest neighbor.
        int: The indices of the nearest neighbor.
    """
    dist, ind = tree.query(x)
    return dist[0][0], ind[0][0]


def one_hot(i: int, d: int):
    """Create a one-hot encoded array of shape `1 x d` with a single non-zero element at index `i`.

    Args:
        i (int): Non-zero bit in one-hot vector.
        d (int): Length of one-hot encoded vector.

    Returns:
        vec (np.ndarray): the one-hot array.
    """
    vec = np.zeros((1, d))
    vec[0, i] = 1
    return vec


def can_react(state, rxns):
    """
    Determines if two molecules can react using any of the input reactions.

    Args:
        state (np.ndarray): The current state in the synthetic tree.
        rxns (list of Reaction objects): Contains available reaction templates.

    Returns:
        np.ndarray: The sum of the reaction mask tells us how many reactions are
             viable for the two molecules.
        np.ndarray: The reaction mask, which masks out reactions which are not
            viable for the two molecules.
    """
    mol1 = state.pop()
    mol2 = state.pop()
    reaction_mask = [int(rxn.run_reaction([mol1, mol2]) is not None) for rxn in rxns]
    return sum(reaction_mask), reaction_mask


def get_action_mask(state, rxns):
    """
    Determines which actions can apply to a given state in the synthetic tree
    and returns a mask for which actions can apply.

    Args:
        state (np.ndarray): The current state in the synthetic tree.
        rxns (list of Reaction objects): Contains available reaction templates.

    Raises:
        ValueError: There is an issue with the input state.

    Returns:
        np.ndarray: The action mask. Masks out unviable actions from the current
            state using 0s, with 1s at the positions corresponding to viable
            actions.
    """
    if len(state) == 0:
        return np.array([1, 0, 0, 0])

    if len(state) == 1:
        return np.array([1, 1, 0, 1])

    if len(state) == 2:
        can_react_, _ = can_react(state, rxns)
        if can_react_:
            return np.array([0, 1, 1, 0])

        return np.array([0, 1, 0, 0])

    raise ValueError("Problem with state.")


def get_mol_embedding(smi, model, device="cpu", readout=AvgPooling()):
    """
    Computes the molecular graph embedding for the input SMILES.

    Args:
        smi (str): SMILES of molecule to embed.
        model (dgllife.model, optional): Pre-trained NN model to use for
            computing the embedding.
        device (str, optional): Indicates the device to run on. Defaults to 'cpu'.
        readout (dgl.nn.pytorch.glob, optional): Readout function to use for
            computing the graph embedding. Defaults to readout.

    Returns:
        torch.Tensor: Learned embedding for the input molecule.
    """
    mol = Chem.MolFromSmiles(smi)
    g = mol_to_bigraph(
        mol,
        add_self_loop=True,
        node_featurizer=PretrainAtomFeaturizer(),
        edge_featurizer=PretrainBondFeaturizer(),
        canonical_atom_order=False,
    )
    bg = g.to(device)
    nfeats = [
        bg.ndata.pop("atomic_number").to(device),
        bg.ndata.pop("chirality_type").to(device),
    ]
    efeats = [
        bg.edata.pop("bond_type").to(device),
        bg.edata.pop("bond_direction_type").to(device),
    ]
    with torch.no_grad():
        node_repr = model(bg, nfeats, efeats)
    return readout(bg, node_repr).detach().cpu().numpy()[0]


def set_embedding(z_target, state, nbits, _mol_embedding=get_mol_embedding):
    """
    Computes embeddings for all molecules in the input space.

    Args:
        z_target (np.ndarray): Embedding for the target molecule.
        state (list): Contains molecules in the current state, if not the
            initial state.
        nbits (int): Length of fingerprint.
        _mol_embedding (Callable, optional): Function to use for computing the
            embeddings of the first and second molecules in the state. Defaults
            to `get_mol_embedding`.

    Returns:
        np.ndarray: Embedding consisting of the concatenation of the target
            molecule with the current molecules (if available) in the input state.
    """
    if len(state) == 0:
        return np.concatenate([np.zeros((1, 2 * nbits)), z_target], axis=1)

    e1 = np.expand_dims(_mol_embedding(state[0]), axis=0)
    if len(state) == 1:
        e2 = np.zeros((1, nbits))
    else:
        e2 = _mol_embedding(state[1])
    return np.concatenate([e1, e2, z_target], axis=1)


def synthetic_tree_decoder(
    z_target,
    building_blocks,
    bb_dict,
    reaction_templates,
    mol_embedder,
    action_net,
    reactant1_net,
    rxn_net,
    reactant2_net,
    bb_emb,
    rxn_template,
    n_bits,
    max_step=15,
):
    """
    Computes the synthetic tree given an input molecule embedding, using the
    Action, Reaction, Reactant1, and Reactant2 networks and a greedy search

    Args:
        z_target (np.ndarray): Embedding for the target molecule
        building_blocks (list of str): Contains available building blocks
        bb_dict (dict): Building block dictionary
        reaction_templates (list of Reactions): Contains reaction templates
        mol_embedder (dgllife.model.gnn.gin.GIN): GNN to use for obtaining
            molecular embeddings
        action_net (synth_net.models.mlp.MLP): The action network
        reactant1_net (synth_net.models.mlp.MLP): The reactant1 network
        rxn_net (synth_net.models.mlp.MLP): The reaction network
        reactant2_net (synth_net.models.mlp.MLP): The reactant2 network
        bb_emb (list): Contains purchasable building block embeddings.
        rxn_template (str): Specifies the set of reaction templates to use.
        n_bits (int): Length of fingerprint.
        max_step (int, optional): Maximum number of steps to include in the
            synthetic tree

    Returns:
        tree (SyntheticTree): The final synthetic tree.
        act (int): The final action (to know if the tree was "properly"
            terminated).
    """
    tree = SyntheticTree()
    kdtree = BallTree(bb_emb, metric="cosine")
    mol_recent = None

    for _ in range(max_step):
        # Encode current state
        state = tree.get_state()  # a set
        z_state = set_embedding(z_target, state, nbits=n_bits, _mol_embedding=mol_fp)

        # Predict action type, masked selection
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        action_proba = action_net(torch.tensor(z_state))
        action_proba = action_proba.squeeze().detach().numpy() + 1e-10
        action_mask = get_action_mask(tree.get_state(), reaction_templates)
        act = np.argmax(action_proba * action_mask)

        z_mol1 = reactant1_net(torch.tensor(z_state))
        z_mol1 = z_mol1.detach().numpy()

        if act == Action.END:
            break
        elif act == Action.ADD:
            _, ind = nn_search(z_mol1, tree=kdtree)
            mol1 = building_blocks[ind]
        else:
            mol1 = mol_recent

        z_mol1 = mol_fp(mol1)

        # Select reaction
        try:
            reaction_proba = rxn_net(
                torch.tensor(np.concatenate([z_state, z_mol1], axis=1))
            )
        except:
            z_mol1 = np.expand_dims(z_mol1, axis=0)
            reaction_proba = rxn_net(
                torch.tensor(np.concatenate([z_state, z_mol1], axis=1))
            )
        reaction_proba = reaction_proba.squeeze().detach().numpy() + 1e-10

        if act != Action.MERGE:
            reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
        else:
            _, reaction_mask = can_react(tree.get_state(), reaction_templates)
            available_list = [[] for _ in reaction_templates]

        if reaction_mask is None:
            if len(state) == 1:
                act = 3
                break
            else:
                break

        rxn_id = np.argmax(reaction_proba * reaction_mask)
        rxn = reaction_templates[rxn_id]

        if rxn.num_reactant == 2:
            if act == Action.MERGE:
                temp = set(state) - set([mol1])
                mol2 = temp.pop()
            else:
                # Add or Expand
                if rxn_template == "hb":
                    num_rxns = 91
                elif rxn_template == "pis":
                    num_rxns = 4700
                else:
                    num_rxns = 3  # unit testing uses only 3 reaction templates
                reactant2_net_input = torch.tensor(
                    np.concatenate([z_state, z_mol1, one_hot(rxn_id, num_rxns)], 1)
                )
                z_mol2 = reactant2_net(reactant2_net_input)
                z_mol2 = z_mol2.detach().numpy()

                available = available_list[rxn_id]
                available = [bb_dict[available[i]] for i in range(len(available))]

                temp_emb = bb_emb[available]
                available_tree = BallTree(temp_emb, metric=cosine_distance)

                _, ind = nn_search(z_mol2, tree=available_tree)
                mol2 = building_blocks[available[ind]]
        else:
            mol2 = None

        # Run reaction
        mol_product = rxn.run_reaction([mol1, mol2])
        if mol_product is None or Chem.MolFromSmiles(mol_product) is None:
            if len(tree.get_state()) == 1:
                act = Action.END
                break
            else:
                break

        # Update
        tree.update(act, int(rxn_id), mol1, mol2, mol_product)
        mol_recent = mol_product

    if act != 3:
        tree = tree
    else:
        tree.update(act, None, None, None, None)

    return tree, act
