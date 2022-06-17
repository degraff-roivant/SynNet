"""
This file contains various utils for data preparation and preprocessing.
"""
from typing import Optional

from dgllife.model import load_pretrained
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from tdc.chem_utils import MolConvert
from syn_net.utils.data_utils import Reaction

from syn_net.utils.tree import SyntheticTree
from syn_net.utils.predict_utils import (
    can_react,
    get_action_mask,
    get_reaction_mask,
    mol_fp,
    get_mol_embedding,
)
from syn_net.utils.utils import Action


def rdkit2d_embedding(smi: Optional[str]) -> np.ndarray:
    """
    Computes an embedding using RDKit 2D descriptors.

    Args:
        smi (str): SMILES string.

    Returns:
        np.ndarray: A molecular embedding corresponding to the input molecule.
    """
    if smi is None:
        return np.zeros(200).reshape(-1)
    
    return MolConvert("SMILES", "RDKit2D")(smi).reshape(-1)


def organize(
    st: SyntheticTree,
    d_mol: int = 300,
    target_embedding: str = "fp",
    radius: int = 2,
    nBits: int = 4096,
    output_embedding: str = "gin",
):
    """
    Organizes the states and steps from the input synthetic tree into sparse
    matrices.

    Args:
        st (SyntheticTree): The input synthetic tree to organize.
        d_mol (int, optional): The molecular embedding size. Defaults to 300.
        target_embedding (str, optional): Indicates what embedding type to use
            for the input target (Morgan fingerprint --> 'fp' or GIN --> 'gin').
            Defaults to 'fp'.
        radius (int, optional): Morgan fingerprint radius to use. Defaults to 2.
        nBits (int, optional): Number of bits to use in the Morgan fingerprints.
            Defaults to 4096.
        output_embedding (str, optional): Indicates what type of embedding to
            use for the output node states. Defaults to 'gin'.

    Raises:
        ValueError: Raised if target embedding not supported.

    Returns:
        sparse.csc_matrix: Node states pulled from the tree.
        sparse.csc_matrix: Actions pulled from the tree.
    """
    # define model to use for molecular embedding
    model_type = "gin_supervised_contextpred"
    device = "cpu"
    model = load_pretrained(model_type).to(device)
    model.eval()

    states = []
    steps = []

    if output_embedding == "gin":
        d_mol = 300
    elif output_embedding == "fp_4096":
        d_mol = 4096
    elif output_embedding == "fp_256":
        d_mol = 256
    elif output_embedding == "rdkit2d":
        d_mol = 200

    if target_embedding == "fp":
        target = mol_fp(st.root.smiles, radius, nBits).tolist()
    elif target_embedding == "gin":
        target = get_mol_embedding(st.root.smiles, model).tolist()
    else:
        raise ValueError("Target embedding only supports fp and gin.")

    most_recent_mol = None
    other_root_mol = None
    for i, action in enumerate(st.actions):

        most_recent_mol_embedding = mol_fp(most_recent_mol, radius, nBits).tolist()
        other_root_mol_embedding = mol_fp(other_root_mol, radius, nBits).tolist()
        state = most_recent_mol_embedding + other_root_mol_embedding + target

        if action == Action.END:
            step = [3] + [0] * d_mol + [-1] + [0] * d_mol + [0] * nBits

        else:
            r = st.reactions[i]
            mol1 = r.child[0]
            if len(r.child) == 2:
                mol2 = r.child[1]
            else:
                mol2 = None

            if output_embedding == "gin":
                step = (
                    [action]
                    + get_mol_embedding(mol1, model).tolist()
                    + [r.rxn_id]
                    + get_mol_embedding(mol2, model).tolist()
                    + mol_fp(mol1, radius, nBits).tolist()
                )
            elif output_embedding == "fp_4096":
                step = (
                    [action]
                    + mol_fp(mol1, 2, 4096).tolist()
                    + [r.rxn_id]
                    + mol_fp(mol2, 2, 4096).tolist()
                    + mol_fp(mol1, radius, nBits).tolist()
                )
            elif output_embedding == "fp_256":
                step = (
                    [action]
                    + mol_fp(mol1, 2, 256).tolist()
                    + [r.rxn_id]
                    + mol_fp(mol2, 2, 256).tolist()
                    + mol_fp(mol1, radius, nBits).tolist()
                )
            elif output_embedding == "rdkit2d":
                step = (
                    [action]
                    + rdkit2d_embedding(mol1).tolist()
                    + [r.rxn_id]
                    + rdkit2d_embedding(mol2).tolist()
                    + mol_fp(mol1, radius, nBits).tolist()
                )

        if action == Action.MERGE:
            most_recent_mol = r.parent
            other_root_mol = None

        elif action == Action.EXPAND:
            most_recent_mol = r.parent

        elif action == Action.ADD:
            other_root_mol = most_recent_mol
            most_recent_mol = r.parent

        states.append(state)
        steps.append(step)

    return sparse.csc_matrix(np.array(states)), sparse.csc_matrix(np.array(steps))


def generate_synthetic_tree(
    building_blocks: list[str], reaction_templates: list[Reaction], max_step: int = 15
) -> tuple[SyntheticTree, Action]:
    """
    Generates a synthetic tree from the available building blocks and reaction
    templates. Used in preparing the training/validation/testing data.

    Args:
        building_blocks (list): Contains SMILES strings for purchasable building
            blocks.
        reaction_templates (list): Contains `Reaction` objects.
        max_step (int, optional): Indicates the maximum number of reaction steps
            to use for building the synthetic tree data. Defaults to 15.

    Returns:
        tree (SyntheticTree): The built up synthetic tree.
        action (Action): the final action.
    """
    tree = SyntheticTree()
    mol_recent = None

    try:
        for _ in range(max_step):
            p_action = np.random.rand(len(Action))
            action_mask = get_action_mask(tree.state, reaction_templates)
            action = np.argmax(p_action * action_mask)

            if action == Action.END:
                break
            elif action == Action.ADD:
                mol1 = np.random.choice(building_blocks)
            else:
                mol1 = mol_recent

            p_reaction = np.random.rand(len(reaction_templates))

            if action != Action.MERGE:
                rxn_mask, available = get_reaction_mask(mol1, reaction_templates)
            else:
                _, rxn_mask = can_react(tree.state, reaction_templates)
                available = [[] for _ in reaction_templates]

            if rxn_mask is None:
                if len(tree.state) == 1:
                    action = Action.END
                break

            rxn_id = np.argmax(p_reaction * rxn_mask)
            rxn = reaction_templates[rxn_id]

            if rxn.num_reactant == 2:
                if action == Action.MERGE:
                    temp = set(tree.state) - set([mol1])
                    mol2 = temp.pop()
                else:
                    mol2 = np.random.choice(available[rxn_id])
            else:
                mol2 = None

            mol_product = rxn.run_reaction([mol1, mol2])

            tree.update(action, int(rxn_id), mol1, mol2, mol_product)
            mol_recent = mol_product

    except Exception as e:
        print(e)
        action = -1
        tree = None

    if action != Action.END:
        tree = None
    else:
        tree.update(action, None, None, None, None)

    return tree, action


def prep_data(main_dir, num_rxn, out_dim):
    """
    Loads the states and steps from preprocessed *.npz files and saves data
    specific to the Action, Reactant 1, Reaction, and Reactant 2 networks in
    their own *.npz files.

    Args:
        main_dir (str): The path to the directory containing the *.npz files.
        num_rxn (int): Number of reactions in the dataset.
        out_dim (int): Size of the output feature vectors.
    """

    for dataset in ["train", "valid", "test"]:

        print("Reading " + dataset + " data ......")
        states_list = []
        steps_list = []
        for i in range(1):
            states_list.append(sparse.load_npz(f"{main_dir}states_{i}_{dataset}.npz"))
            steps_list.append(sparse.load_npz(f"{main_dir}steps_{i}_{dataset}.npz"))

        states = sparse.csc_matrix(sparse.vstack(states_list))
        steps = sparse.csc_matrix(sparse.vstack(steps_list))

        # extract Action data
        X = states
        y = steps[:, 0]
        sparse.save_npz(f"{main_dir}X_act_{dataset}.npz", X)
        sparse.save_npz(f"{main_dir}y_act_{dataset}.npz", y)

        states = sparse.csc_matrix(
            states.A[
                (steps[:, 0].A != 3).reshape(
                    -1,
                )
            ]
        )
        steps = sparse.csc_matrix(
            steps.A[
                (steps[:, 0].A != 3).reshape(
                    -1,
                )
            ]
        )

        # extract Reaction data
        X = sparse.hstack([states, steps[:, (2 * out_dim + 2) :]])
        y = steps[:, out_dim + 1]
        sparse.save_npz(f"{main_dir}X_rxn_{dataset}.npz", X)
        sparse.save_npz(f"{main_dir}y_rxn_{dataset}.npz", y)

        states = sparse.csc_matrix(
            states.A[
                (steps[:, 0].A != 2).reshape(
                    -1,
                )
            ]
        )
        steps = sparse.csc_matrix(
            steps.A[
                (steps[:, 0].A != 2).reshape(
                    -1,
                )
            ]
        )

        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit([[i] for i in range(num_rxn)])
        # import ipdb; ipdb.set_trace(context=9)

        # extract Reactant 2 data
        X = sparse.hstack(
            [
                states,
                steps[:, (2 * out_dim + 2) :],
                sparse.csc_matrix(
                    enc.transform(steps[:, out_dim + 1].A.reshape((-1, 1))).toarray()
                ),
            ]
        )
        y = steps[:, (out_dim + 2) : (2 * out_dim + 2)]
        sparse.save_npz(f"{main_dir}X_rt2_{dataset}.npz", X)
        sparse.save_npz(f"{main_dir}y_rt2_{dataset}.npz", y)

        states = sparse.csc_matrix(
            states.A[
                (steps[:, 0].A != 1).reshape(
                    -1,
                )
            ]
        )
        steps = sparse.csc_matrix(
            steps.A[
                (steps[:, 0].A != 1).reshape(
                    -1,
                )
            ]
        )

        # extract Reactant 1 data
        X = states
        y = steps[:, 1 : (out_dim + 1)]
        sparse.save_npz(f"{main_dir}X_rt1_{dataset}.npz", X)
        sparse.save_npz(f"{main_dir}y_rt1_{dataset}.npz", y)

        return None
