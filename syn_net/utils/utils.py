from enum import Enum, auto
from typing import Union

from rdkit import Chem

MolLike = Union[str, Chem.Mol]

class Action(Enum):
    ADD = 0
    EXPAND = 1
    MERGE = 2
    END = 3


class Embedding(Enum):
    GIN = 300
    FP_4096 = 4096
    FP_256 = 256
    RDKIT_2D = 200


class ReactionType(Enum):
    UNIMOLECULAR = auto()
    BIMOLECULAR = auto()


class InvalidSmirksError(ValueError):
    pass