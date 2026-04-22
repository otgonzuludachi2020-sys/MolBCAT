"""
GRU-based encoder and classifier/regressor for SMILES sequences.

Supports three training modes (configured externally):
  - Random:   trained from scratch
  - Frozen:   pretrained weights loaded, encoder frozen
  - Finetune: pretrained weights loaded, full fine-tuning
"""
import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    GRU encoder for SMILES sequences.
    Used as the sequence branch inside MolBCAT.
    """

    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns full GRU output sequence. Shape: (B, L, H)."""
        return self.gru(self.emb(x))[0]


class GRUModel(nn.Module):
    """
    GRU model with a classification or regression head.
    Works for GRU_Random, GRU_Frozen, and GRU_Finetune variants.
    The training mode (frozen / pretrained) is configured in train scripts.
    """

    def __init__(self, vocab_size: int,
                 emb_dim: int = 256,
                 hidden_dim: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.emb     = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru     = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape (B, L)
        Returns:
            Logits / predictions, shape (B, 1)
        """
        _, h = self.gru(self.emb(x))
        return self.fc(self.dropout(h[-1]))
