from enum import Enum, auto
from typing import Union

from rdkit import Chem

MolLike = Union[str, Chem.Mol]

class Action(Enum):
    ADD = auto()
    EXPAND = auto()
    MERGE = auto()
    END = auto()


class ReactionType(Enum):
    UNIMOLECULAR = auto()
    BIMOLECULAR = auto()


class InvalidSmirksError(ValueError):
    pass