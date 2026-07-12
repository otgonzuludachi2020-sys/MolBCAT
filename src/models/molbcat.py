"""
MolBCAT — Bidirectional Cross-Modal Attention model.
Combines a pretrained GRU encoder (SMILES) with a GIN encoder (graph)
via bidirectional cross-modal attention and a gated fusion mechanism.
"""
import torch
import torch.nn as nn
from .gru import GRUEncoder
from .gin import GINEncoder


class MolBCAT(nn.Module):
    """
    MolBCAT: Self-supervised GRU pretraining with bidirectional
    cross-modal attention for molecular property prediction.

    Works for both classification and regression tasks.
    The task is determined by the loss function used during training.
    """

    def __init__(self,
                 vocab_size:    int,
                 emb_dim:       int = 256,
                 hidden_dim:    int = 512,
                 gin_hidden:    int = 128,
                 node_feat_dim: int = 9,
                 nhead:         int = 4,
                 dropout:       float = 0.2):
        super().__init__()

        self.pad_id = 0  # updated after vocab is loaded

        # Encoders
        self.gru_encoder = GRUEncoder(vocab_size, emb_dim, hidden_dim)
        self.gin_encoder = GINEncoder(node_feat_dim, gin_hidden)
        self.gin_proj    = nn.Linear(gin_hidden, hidden_dim)

        # Bidirectional cross-modal attention
        self.attn_sg = nn.MultiheadAttention(hidden_dim, nhead,
                                              dropout=dropout, batch_first=True)
        self.attn_gs = nn.MultiheadAttention(hidden_dim, nhead,
                                              dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Sequence gate: fuses CLS token and mean pooling
        self.seq_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Gated fusion
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        # Classification / regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _encode(self, smiles_x: torch.Tensor,
                graph_data) -> torch.Tensor:
        """
        Core encoding logic shared by forward() and get_embedding().

        Args:
            smiles_x:   Token IDs, shape (B, L)
            graph_data: PyG Batch object

        Returns:
            Fused representation h_out, shape (B, hidden_dim)
        """
        device = smiles_x.device

        # --- Sequence branch ---
        gru_out = self.gru_encoder(smiles_x)          # (B, L, H)
        h_cls   = gru_out[:, 0, :]                    # CLS token
        mask    = (smiles_x != self.pad_id).float()
        h_mean  = (
            (gru_out * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp(min=1)
        )
        h_seq = torch.sigmoid(
            self.seq_gate(torch.cat([h_cls, h_mean], dim=1))
        ).unsqueeze(1)                                 # (B, 1, H)

        # --- Graph branch ---
        atom_emb, bvec = self.gin_encoder(graph_data)  # (N_atoms, G)
        atom_emb = self.gin_proj(atom_emb)              # (N_atoms, H)

        batch_size = smiles_x.size(0)
        max_atoms  = int(torch.bincount(bvec).max().item())
        max_atoms  = max(max_atoms, 1)

        H_atom_pad   = torch.zeros(batch_size, max_atoms, atom_emb.size(-1), device=device)
        atom_counts  = torch.zeros(batch_size, dtype=torch.long, device=device)
        key_pad_mask = torch.ones(batch_size, max_atoms, dtype=torch.bool, device=device)

        for i in range(batch_size):
            atoms_i = atom_emb[bvec == i]
            n = atoms_i.size(0)
            atom_counts[i]       = n
            H_atom_pad[i, :n]    = atoms_i
            key_pad_mask[i, :n]  = False

        # --- Bidirectional cross-modal attention ---
        # Sequence attends to graph
        h_sg, _ = self.attn_sg(
            h_seq, H_atom_pad, H_atom_pad,
            key_padding_mask=key_pad_mask
        )
        h_sg = self.norm(h_sg.squeeze(1))             # (B, H)

        # Graph attends to sequence
        h_gs_all, _ = self.attn_gs(
            H_atom_pad, gru_out, gru_out,
            key_padding_mask=(smiles_x == self.pad_id)
        )
        h_gs_all = h_gs_all * (1 - key_pad_mask.unsqueeze(-1).float())
        h_gs = self.norm(
            h_gs_all.sum(1) / atom_counts.float().unsqueeze(1).clamp(min=1)
        )                                              # (B, H)

        # --- Gated fusion ---
        gate  = torch.sigmoid(self.gate_fc(torch.cat([h_sg, h_gs], dim=1)))
        h_out = gate * h_sg + (1 - gate) * h_gs       # (B, H)

        return h_out

    def forward(self, smiles_x: torch.Tensor,
                graph_data) -> torch.Tensor:
        """
        Args:
            smiles_x:   Token IDs, shape (B, L)
            graph_data: PyG Batch object

        Returns:
            Logits / predictions, shape (B, 1)
        """
        return self.head(self._encode(smiles_x, graph_data))

    def get_embedding(self, smiles_x: torch.Tensor,
                      graph_data) -> torch.Tensor:
        """
        Return the fused representation before the prediction head.
        Useful for downstream tasks or visualization.

        Returns:
            Embeddings, shape (B, hidden_dim)
        """
        return self._encode(smiles_x, graph_data)
