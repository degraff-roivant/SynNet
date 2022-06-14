from __future__ import annotations

import gzip
import json
from os import PathLike
from pathlib import Path
from typing import Iterable, Optional

from rdkit.Chem import Draw
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm

from syn_net.utils.utils import InvalidSmirksError, MolLike


class Reaction:
    """
    This class models a chemical reaction based on a SMARTS transformation.

    Args:
        template (str): SMARTS string representing a chemical reaction.
        rxnname (str): The name of the reaction for downstream analysis.
        smiles: (str): A reaction SMILES string that macthes the SMARTS pattern.
        reference (str): Reference information for the reaction.
    """

    def __init__(
        self,
        smirks: str = None,
        rxnname: str = None,
        smiles: str = None,
        reference: str = None,
    ):
        self.smirks = smirks
        if self.smirks is None:
            return

        self.rxnname = rxnname
        self.smiles = smiles
        self.reference = reference

    def to_dict(self) -> dict[str, str]:
        return {
            "smirks": self.smirks,
            "rxnname": self.rxnname,
            "smiles": self.smiles,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(
        cls,
        smirks,
        num_reactant,
        num_agent,
        num_product,
        reactant_template,
        product_template,
        agent_template,
        available_reactants,
        rxnname,
        smiles,
        reference,
        **kwargs,
    ) -> Reaction:
        rxn = cls(None)
        rxn.smirks = smirks
        rxn.num_reactant = num_reactant
        rxn.num_agent = num_agent
        rxn.num_product = num_product
        rxn.reactant_template = list(reactant_template)
        rxn.product_template = product_template
        rxn.agent_template = agent_template
        rxn.available_reactants = list(available_reactants)
        rxn.rxnname = rxnname
        rxn.smiles = smiles
        rxn.reference = reference

    @property
    def smirks(self) -> str:
        return self.__smirks

    @smirks.setter
    def smirks(self, smirks: Optional[str]):
        self.__smirks = smirks

        if smirks is None:
            return

        try:
            self.rxn = Chem.ReactionFromSmarts(self.smirks)
        except ValueError:
            raise ValueError(
                f"arg `smirks` must contain either 1 or 2 reactants! got: {self.num_reactant}"
            )

        try:
            reactants_template, agents_template, products_template = self.smirks.split(
                ">"
            )
        except ValueError:
            raise InvalidSmirksError

        if self.num_reactant == 1:
            self.reactant_template = [reactants_template]
        else:
            self.reactant_template = reactants_template.split(".")[:2]

        self.agent_template = agents_template
        self.product_template = products_template

    @property
    def rxn(self):
        return self.__rxn

    @rxn.setter
    def rxn(self, rxn):
        self.__rxn = rxn
        rdChemReactions.ChemicalReaction.Initialize(self.rxn)

        self.num_reactant = self.rxn.GetNumReactantTemplates()
        self.num_agent = self.rxn.GetNumAgentTemplates()
        self.num_product = self.rxn.GetNumProductTemplates()

        if self.num_reactant not in range(1, 3):
            raise ValueError(
                f"arg `rxn` must contain either 1 or 2 reactants! got: {self.num_reactant}"
            )

    @property
    def available_reactants(self) -> list[list[str]]:
        return self.__available_reactants

    @available_reactants.setter
    def available_reactants(self, building_blocks: list[str]):
        self.__available_reactants = list(self.filter_reactants(building_blocks))

    def visualize(self, path: PathLike = "./reaction1_highlight.o.png"):
        """plot the chemical translation into a PNG figure.

        Example
        -------
        >>> from IPython.display import Image
        >>> rxn: Reaction
        >>> path: PathLike
        >>> rxn.visualize(path)
        ...
        >>> Image(path)

        Args:
            path (str): The path to the figure.

        Returns:
            path (str): The path to the figure.
        """
        rxn = Chem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(800, 300)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        png = d2d.GetDrawingText()
        Path(path).write_text(png)

        return path

    def is_reactant(self, smi):
        """
        A function that checks if a molecule is a reactant of the reaction
        defined by the `Reaction` object.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            result (bool): Indicates if the molecule is a reactant of the reaction.
        """
        # rxn = self.get_rxnobj()
        smi = self.get_mol(smi)
        result = self.rxn.IsMoleculeReactant(smi)

        return result

    def is_agent(self, mol: MolLike):
        """
        A function that checks if a molecule is an agent in the reaction defined
        by the `Reaction` object.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            result (bool): Indicates if the molecule is an agent in the reaction.
        """
        # rxn = self.get_rxnobj()
        mol = self.get_mol(mol)
        result = self.rxn.IsMoleculeAgent(mol)

        return result

    def is_product(self, mol: MolLike):
        """
        A function that checks if a molecule is the product in the reaction defined
        by the `Reaction` object.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            result (bool): Indicates if the molecule is the product in the reaction.
        """
        # rxn = self.get_rxnobj()
        mol = self.get_mol(mol)
        result = self.rxn.IsMoleculeProduct(mol)

        return result

    def is_reactant_first(self, mol: MolLike):
        """
        A function that checks if a molecule is the first reactant in the reaction
        defined by the `Reaction` object, where the order of the reactants is
        determined by the SMARTS pattern.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            result (bool): Indicates if the molecule is the first reactant in
                the reaction.
        """
        mol = self.get_mol(mol)
        if mol.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(0))):
            return True
        else:
            return False

    def is_reactant_second(self, mol: MolLike):
        """
        A function that checks if a molecule is the second reactant in the reaction
        defined by the `Reaction` object, where the order of the reactants is
        determined by the SMARTS pattern.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            result (bool): Indicates if the molecule is the second reactant in
                the reaction.
        """
        mol = self.get_mol(mol)
        if mol.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(1))):
            return True
        else:
            return False

    def get_rxnobj(self):
        """
        A function that returns the RDKit Reaction object.

        Returns:
            rxn (rdChem.Reactions.ChemicalReaction): RDKit reaction object.
        """
        rxn = Chem.ReactionFromSmarts(self.smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def get_reactant_template(self, idx: int = 0):
        """
        A function that returns the SMARTS pattern which represents the specified
        reactant.

        Args:
            idx (int): The index of the reactant. Defaults to 0.

        Returns:
            reactant_template (str): SMARTS pattern representing the reactant.
        """
        return self.reactant_template[idx]

    def run_reaction(self, reactants: Iterable[MolLike], keep_main: bool = True):
        """
        A function that transform the reactants into the corresponding product.

        Args:
            reactants (list): Contains SMILES strings for the reactants.
            keep_main (bool): Indicates whether to return only the main product,
                or all possible products. Defaults to True.

        Returns:
            uniqps (str): SMILES string representing the product.
        """
        rxn = self.rxn  # self.get_rxnobj()

        if self.num_reactant == 1:
            if isinstance(reactants, (str, Chem.Mol)):
                r = self.get_mol(reactants)
            elif isinstance(reactants, Iterable):
                if len(reactants) == 1:
                    r = self.get_mol(reactants[0])
                elif len(reactants) == 2 and reactants[1] is None:
                    r = self.get_mol(reactants[0])
                else:
                    return None
            else:
                raise TypeError(
                    "The input of a uni-molecular reaction should "
                    "be a SMILES, an rdkit.Chem.Mol object, or a "
                    "tuple/list of length 1 or 2."
                )

            if not self.is_reactant(r):
                return None

            ps = rxn.RunReactants((r,))
        else:
            if not (isinstance(reactants, (tuple, list)) and len(reactants) == 2):
                raise TypeError(
                    "The input of a bi-molecular reaction should "
                    "be a tuple/list of length 2."
                )

            r1 = self.get_mol(reactants[0])
            r2 = self.get_mol(reactants[1])

            if self.is_reactant_first(r1) and self.is_reactant_second(r2):
                pass
            elif self.is_reactant_first(r2) and self.is_reactant_second(r1):
                r1, r2 = r2, r1
            else:
                return None

            ps = rxn.RunReactants((r1, r2))

        uniqps = list({Chem.MolToSmiles(p[0]) for p in ps})

        if len(uniqps) < 1:
            raise RuntimeError("Reaction produced no products!")

        if keep_main:
            return uniqps[0]

        return uniqps

    def filter_reactants(self, mols: Iterable[MolLike]):
        """Filter invalid reactants from the list of molecules.

        Args:
            smis (list[MolLike]): the molecules to filter

        Returns:
            tuple[list[str], list[str]]: a tuple of list(s) of SMILES which match either the first
                reactant, or, if applicable, the second reactant.

        """
        if self.num_reactant == 1:
            smi_w_patt = [mol for mol in tqdm(mols) if self.is_reactant_first(mol)]
            reactants = (smi_w_patt,)

        else:
            smi_w_patt1 = []
            smi_w_patt2 = []
            for mol in tqdm(mols):
                if self.is_reactant_first(mol):
                    smi_w_patt1.append(mol)
                if self.is_reactant_second(mol):
                    smi_w_patt2.append(mol)
            reactants = (smi_w_patt1, smi_w_patt2)

        return reactants

    def set_available_reactants(self, building_blocks):
        """
        A function that finds the applicable building blocks from a list of
        purchasable building blocks.

        Args:
            building_block_list (list): The list of purchasable building blocks,
                where building blocks are represented as SMILES strings.
        """
        self.available_reactants = list(self.filter_reactants(building_blocks))

    @staticmethod
    def get_mol(mol: MolLike) -> Chem.Mol:
        if isinstance(mol, str):
            return Chem.MolFromSmiles(mol)
        elif isinstance(mol, Chem.Mol):
            return mol
        else:
            raise TypeError(
                f"arg `mol` should have type `str` or `RDKit.Chem.Mol`! got: {type(mol)}"
            )


class ReactionSet:
    """
    A class representing a set of reactions, for saving and loading purposes.

    Arritbutes:
        rxns (list or None): Contains `Reaction` objects. One can initialize the
            class with a list or None object, the latter of which is used to
            define an empty list.
    """

    def __init__(self, rxns: Optional[list[Reaction]] = None):
        self.rxns = rxns or []

    def __len__(self) -> int:
        return len(self.rxns)

    @classmethod
    def load(cls, path: PathLike) -> ReactionSet:
        """load reactions from a JSON file.

        Args:
            json_file (PathLike): The path to the stored reaction file.
        """
        with gzip.open(path, "r") as f:
            data = json.loads(f.read().decode("utf-8"))

        return cls([Reaction.from_dict(**r_dict) for r_dict in data])

    def save(self, path: PathLike):
        """save the reaction set to a JSON file.

        Args:
            json_file (str): The path to the stored reaction file.
        """
        rxns = [r.to_dict() for r in self.rxns]
        with gzip.open(path, "w") as f:
            f.write(json.dumps(rxns).encode("utf-8"))

    def debug(self, x: int = 3):
        for r in self.rxns[:x]:
            print(r.__dict__)
