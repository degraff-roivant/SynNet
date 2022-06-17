from __future__ import annotations

from dataclasses import dataclass, asdict
import gzip
import json
from os import PathLike
from typing import Iterator, Optional, Sequence

from syn_net.utils.utils import Action, ReactionType


@dataclass
class ChemicalNode:
    smiles: str
    is_leaf: bool
    is_root: bool
    parent: int
    child: int
    depth: int
    index: int


@dataclass
class ReactionNode:
    rxn_id: int
    rtype: int
    parent: Optional[list[str]]
    child: list[str]
    depth: float
    index: int


class SyntheticTree:
    def __init__(
        self,
        chemicals: Optional[list[ChemicalNode]] = None,
        reactions: Optional[list[ReactionNode]] = None,
        root: Optional[ChemicalNode] = None,
        depth: int = 0,
        actions: Optional[list[Action]] = None,
        rxn_id2type: Optional[dict[int, int]] = None,
    ):
        self.chemicals = chemicals or []
        self.reactions = reactions or []
        self.root = root
        self.depth = depth
        self.actions = actions or []
        self.rxn_id2type = rxn_id2type

    @classmethod
    def from_dict(cls, data: dict) -> SyntheticTree:
        chemicals = [ChemicalNode(**d_chemical) for d_chemical in data["chemicals"]]
        reactions = [ReactionNode(**d_rxn) for d_rxn in data["reactions"]]
        root = ChemicalNode(**data["root"])
        depth = data["depth"]
        actions = data["actions"]
        rxn_id2type = data["rxn_id2type"]

        return cls(chemicals, reactions, root, depth, actions, rxn_id2type)

    def to_dict(self) -> dict:
        return {
            "chemicals": [asdict(m) for m in self.chemicals],
            "reactions": [asdict(r) for r in self.reactions],
            "root": asdict(self.root),
            "depth": self.depth,
            "actions": self.actions,
            "rxn_id2type": self.rxn_id2type,
        }

    def get_node_index(self, smi: str) -> Optional[int]:
        """get the index of the node with the input SMILES. If there is no such node, return None"""
        for node in self.chemicals:
            if smi == node.smiles:
                return node.index

        return None

    @property
    def state(self) -> list[str]:
        """the state of the synthetic tree. The most recent root node has 0 as its index.

        Returns:
            state (list[str]): A list containing all root node molecules.
        """
        return [node.smiles for node in self.chemicals if node.is_root][::-1]

    def update(
        self, action: Action, rxn_id: int, rct_1: str, rct_2: Optional[str], pdt: str
    ):
        """updates the synthetic tree with the given action and input.

        Args:
            action (Action): the action to apply.
            rxn_id (int): Index of the reaction occured, where the index can be
               anything in the range [0, len(template_list)-1].
            rct_1 (str): SMILES string of the first reactant.
            rct_2 (str): SMILES string of the second reactant.
            pdt (str): SMILES string of the product.
        """
        self.actions.append(action)

        if action == Action.END:
            self.root = self.chemicals[-1]
            self.depth = self.root.depth

        elif action == Action.MERGE:
            node_mol1 = self.chemicals[self.get_node_index(rct_1)]
            node_mol2 = self.chemicals[self.get_node_index(rct_2)]

            node_rxn = ReactionNode(
                rxn_id,
                ReactionType.BIMOLECULAR,
                None,
                [node_mol1.smiles, node_mol2.smiles],
                max(node_mol1.depth, node_mol2.depth) + 0.5,
                len(self.reactions),
            )
            node_product = ChemicalNode(
                pdt,
                False,
                True,
                None,
                node_rxn.rxn_id,
                node_rxn.depth + 0.5,
                len(self.chemicals),
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False
            node_mol2.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == Action.EXPAND:
            if rct_2 is None:
                node_mol1 = self.chemicals[self.get_node_index(rct_1)]
                node_rxn = ReactionNode(
                    rxn_id,
                    ReactionType.UNIMOLECULAR,
                    None,
                    [node_mol1.smiles],
                    node_mol1.depth + 0.5,
                    len(self.reactions),
                )
                node_product = ChemicalNode(
                    pdt,
                    False,
                    True,
                    None,
                    node_rxn.rxn_id,
                    node_rxn.depth + 0.5,
                    len(self.chemicals),
                )

                node_rxn.parent = node_product.smiles
                node_mol1.parent = node_rxn.rxn_id
                node_mol1.is_root = False

                self.chemicals.append(node_product)
                self.reactions.append(node_rxn)

            else:
                node_mol1 = self.chemicals[self.get_node_index(rct_1)]
                node_mol2 = ChemicalNode(
                    rct_2,
                    True,
                    False,
                    None,
                    None,
                    0,
                    len(self.chemicals),
                )
                node_rxn = ReactionNode(
                    rxn_id,
                    ReactionType.BIMOLECULAR,
                    None,
                    [node_mol1.smiles, node_mol2.smiles],
                    max(node_mol1.depth, node_mol2.depth) + 0.5,
                    len(self.reactions),
                )
                node_product = ChemicalNode(
                    pdt,
                    False,
                    True,
                    None,
                    node_rxn.rxn_id,
                    node_rxn.depth + 0.5,
                    len(self.chemicals) + 1,
                )

                node_rxn.parent = node_product.smiles
                node_mol1.parent = node_rxn.rxn_id
                node_mol2.parent = node_rxn.rxn_id
                node_mol1.is_root = False

                self.chemicals.append(node_mol2)
                self.chemicals.append(node_product)
                self.reactions.append(node_rxn)

        elif action == Action.ADD:
            if rct_2 is None:
                node_mol1 = ChemicalNode(
                    rct_1, True, False, None, None, 0, len(self.chemicals)
                )
                node_rxn = ReactionNode(
                    rxn_id,
                    ReactionType.UNIMOLECULAR,
                    None,
                    [node_mol1.smiles],
                    0.5,
                    len(self.reactions),
                )
                node_product = ChemicalNode(
                    pdt, False, True, None, node_rxn.rxn_id, 1, len(self.chemicals) + 1
                )

                node_rxn.parent = node_product.smiles
                node_mol1.parent = node_rxn.rxn_id

                self.chemicals.append(node_mol1)
                self.chemicals.append(node_product)
                self.reactions.append(node_rxn)
            else:
                node_mol1 = ChemicalNode(
                    rct_1,
                    True,
                    False,
                    None,
                    None,
                    0,
                    len(self.chemicals),
                )
                node_mol2 = ChemicalNode(
                    rct_2,
                    True,
                    False,
                    None,
                    None,
                    0,
                    len(self.chemicals) + 1,
                )
                node_rxn = ReactionNode(
                    rxn_id,
                    ReactionType.BIMOLECULAR,
                    None,
                    [node_mol1.smiles, node_mol2.smiles],
                    0.5,
                    len(self.reactions),
                )
                node_product = ChemicalNode(
                    pdt,
                    False,
                    True,
                    None,
                    node_rxn.rxn_id,
                    1,
                    len(self.chemicals) + 2,
                )

                node_rxn.parent = node_product.smiles
                node_mol1.parent = node_rxn.rxn_id
                node_mol2.parent = node_rxn.rxn_id

                self.chemicals.append(node_mol1)
                self.chemicals.append(node_mol2)
                self.chemicals.append(node_product)
                self.reactions.append(node_rxn)

        else:
            raise ValueError("Check input")

    def debug(self):
        BORDER = "*"
        WIDTH = 48
        print(f"{' Stored Molecules ':{BORDER}^{WIDTH}}")
        for node in self.chemicals:
            print(node.smiles, node.is_root)
        print(f"{' Stored Reactions ':{BORDER}^{WIDTH}}")
        for node in self.reactions:
            print(node.rxn_id, node.rtype)
        print(f"{' Followed Actions ':{BORDER}^{WIDTH}}")
        print(self.actions)


class SyntheticTreeSet:
    """A set of synthetic trees, for saving and loading purposes.

    Arritbute:
        sts (list): Contains `SyntheticTree`s. One can initialize the class with
            either a list of synthetic trees or None, in which case an empty
            list is created.
    """

    def __init__(self, trees: Optional[Sequence[SyntheticTree]] = None):
        self.sts = trees or []

    def __len__(self):
        return len(self.sts)
    
    def __iter__(self) -> Iterator[SyntheticTree]:
        return iter(self.sts)
        
    @classmethod
    def load(cls, path: PathLike) -> SyntheticTreeSet:
        """
        A function that loads a JSON-formatted synthetic tree file.

        Args:
            json_file (str): The path to the stored synthetic tree file.
        """
        with gzip.open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))

        trees = [SyntheticTree.from_dict(d) if d is not None else None for d in data]

        return cls(trees)

    def save(self, path: PathLike):
        """
        A function that saves the synthetic tree set to a JSON-formatted file.

        Args:
            json_file (str): The path to the stored synthetic tree file.
        """
        sts = [st.to_dict() if st is not None else None for st in self.sts]
        with gzip.open(path, "wb") as f:
            f.write(json.dumps(sts).encode("utf-8"))


    def debug(self, x: int = 3):
        for st in self.sts[:x]:
            print(st.to_dict())
