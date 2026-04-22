"""
Scaffold-based train/test splitting for molecular datasets.
"""
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_split(smiles: list, seed: int, test_ratio: float = 0.2) -> tuple:
    """
    Split a list of SMILES into train and test indices using Murcko scaffold split.

    Args:
        smiles:     List of SMILES strings.
        seed:       Random seed for reproducibility.
        test_ratio: Fraction of data to use as test set.

    Returns:
        train_idx, test_idx: Lists of integer indices.
    """
    random.seed(seed)

    scaffolds = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.setdefault(scaf, []).append(i)

    groups = list(scaffolds.values())
    random.shuffle(groups)

    train_idx, test_idx = [], []
    test_size = int(len(smiles) * test_ratio)

    for g in groups:
        if len(test_idx) + len(g) <= test_size:
            test_idx.extend(g)
        else:
            train_idx.extend(g)

    return train_idx, test_idx


def scaffold_train_val_test_split(smiles: list, seed: int,
                                   test_ratio: float = 0.2,
                                   val_ratio: float = 0.1) -> tuple:
    """
    Three-way scaffold split: train / val / test.
    All splits are scaffold-based to prevent any leakage.

    Args:
        smiles:     List of SMILES strings.
        seed:       Random seed for reproducibility.
        test_ratio: Fraction for test set.
        val_ratio:  Fraction for validation set.

    Returns:
        train_idx, val_idx, test_idx: Lists of integer indices.
    """
    random.seed(seed)

    scaffolds = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.setdefault(scaf, []).append(i)

    groups = list(scaffolds.values())
    random.shuffle(groups)

    test_size = int(len(smiles) * test_ratio)
    val_size  = int(len(smiles) * val_ratio)

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(test_idx) + len(g) <= test_size:
            test_idx.extend(g)
        elif len(val_idx) + len(g) <= val_size:
            val_idx.extend(g)
        else:
            train_idx.extend(g)

    return train_idx, val_idx, test_idx
