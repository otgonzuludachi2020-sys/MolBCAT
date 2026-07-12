"""
Molecular graph construction for GIN and MolBCAT models.
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import rdchem

from .smiles import encode_smiles


HYBRIDIZATION_MAP = {
    rdchem.HybridizationType.SP:          0,
    rdchem.HybridizationType.SP2:         1,
    rdchem.HybridizationType.SP3:         2,
    rdchem.HybridizationType.SP3D:        3,
    rdchem.HybridizationType.SP3D2:       4,
    rdchem.HybridizationType.OTHER:       5,
    rdchem.HybridizationType.UNSPECIFIED: 5,
}

CHIRAL_MAP = {
    rdchem.ChiralType.CHI_UNSPECIFIED:       0,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW:    1,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:   2,
    rdchem.ChiralType.CHI_OTHER:             3,
}


def _atom_features(atom) -> list:
    """Extract 9-dimensional atom feature vector."""
    return [
        atom.GetAtomicNum()   / 100.0,
        atom.GetDegree()      / 10.0,
        float(atom.GetIsAromatic()),
        atom.GetFormalCharge() / 10.0,
        atom.GetTotalNumHs()  / 8.0,
        HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5) / 5.0,
        float(atom.IsInRing()),
        CHIRAL_MAP.get(atom.GetChiralTag(), 0) / 3.0,
        atom.GetTotalValence() / 6.0,
    ]


def smiles_to_graph(smiles: str, label: float) -> Data | None:
    """
    Convert a SMILES string to a PyG Data object.

    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.zeros((2, 0), dtype=torch.long)
    )
    feats = [_atom_features(atom) for atom in mol.GetAtoms()]

    return Data(
        x=torch.tensor(feats, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.float),
    )


def build_graph_list(smiles_list: list, label_list: list) -> list:
    """Build a list of PyG Data objects, skipping invalid SMILES."""
    return [
        g for s, l in zip(smiles_list, label_list)
        if (g := smiles_to_graph(s, l)) is not None
    ]


class CrossAttnDataset(Dataset):
    """
    Dataset for MolBCAT: returns (smiles_tensor, graph, label) triplets.
    Used for both classification and regression.
    """

    def __init__(self, smiles_list: list, label_list: list,
                 vocab: dict, max_len: int):
        self.items = []
        for s, l in zip(smiles_list, label_list):
            g = smiles_to_graph(s, l)
            if g is not None:
                smiles_t = torch.tensor(
                    encode_smiles(s, vocab, max_len, use_cls=True),
                    dtype=torch.long
                )
                self.items.append((smiles_t, g, l))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def crossattn_collate(batch: list) -> tuple:
    """Collate function for CrossAttnDataset."""
    smiles_list, graph_list, label_list = zip(*batch)
    return (
        torch.stack(smiles_list),
        Batch.from_data_list(graph_list),
        torch.tensor(label_list, dtype=torch.float32),
    )
