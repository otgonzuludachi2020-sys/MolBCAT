"""
SMILES encoding and dataset utilities.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from rdkit import Chem


def filter_valid(smiles_raw: list, labels_raw: list, task: str = 'cls') -> tuple:
    """
    Filter out invalid SMILES and missing labels.

    Args:
        smiles_raw: Raw list of SMILES strings.
        labels_raw: Raw list of labels.
        task:       'cls' for classification (int labels),
                    'reg' for regression (float labels).

    Returns:
        smiles, labels: Filtered lists.
    """
    valid_smiles, valid_labels = [], []
    for s, l in zip(smiles_raw, labels_raw):
        if l is None:
            continue
        if isinstance(l, float) and np.isnan(l):
            continue
        if Chem.MolFromSmiles(s) is None:
            continue
        try:
            label = int(l) if task == 'cls' else float(l)
        except (ValueError, TypeError):
            continue
        valid_smiles.append(s)
        valid_labels.append(label)

    n_removed = len(smiles_raw) - len(valid_smiles)
    print(f"  Valid samples: {len(valid_smiles)} / {len(smiles_raw)}"
          + (f" ({n_removed} removed)" if n_removed else ""))
    return valid_smiles, valid_labels


def encode_smiles(s: str, vocab: dict, max_len: int, use_cls: bool = False) -> list:
    """
    Encode a SMILES string into a list of token IDs.

    Args:
        s:       SMILES string.
        vocab:   Token-to-ID vocabulary dict.
        max_len: Maximum sequence length.
        use_cls: If True, prepend <CLS> token (used for MolBCAT).

    Returns:
        List of integer token IDs, padded to max_len.
    """
    unk_id = vocab.get('<UNK>', 3)
    pad_id = vocab.get('<PAD>', 0)
    cls_id = vocab.get('<CLS>', 2)

    if use_cls:
        seq = [cls_id] + [vocab.get(c, unk_id) for c in s][:max_len - 1]
    else:
        seq = [vocab.get(c, unk_id) for c in s][:max_len]

    seq += [pad_id] * (max_len - len(seq))
    return seq


class SMILESDataset(Dataset):
    """Dataset for GRU-based models (sequence input only)."""

    def __init__(self, smiles_list: list, label_list: list,
                 vocab: dict, max_len: int, use_cls: bool = False):
        self.data = [
            (torch.tensor(encode_smiles(s, vocab, max_len, use_cls), dtype=torch.long),
             torch.tensor(l, dtype=torch.float32))
            for s, l in zip(smiles_list, label_list)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_gru_loaders(X_tr, y_tr, X_val, y_val,
                     vocab: dict, max_len: int,
                     batch_size: int = 64,
                     num_workers: int = 2) -> tuple:
    """Build DataLoaders for GRU training.

    Note: use_cls=False here (no CLS token) — this is correct for standalone
    GRU models. MolBCAT uses CrossAttnDataset which calls encode_smiles with
    use_cls=True separately.
    """
    X_tr_t  = torch.tensor([encode_smiles(s, vocab, max_len) for s in X_tr])
    X_val_t = torch.tensor([encode_smiles(s, vocab, max_len) for s in X_val])
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    import torch.cuda
    pin = torch.cuda.is_available()

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader
