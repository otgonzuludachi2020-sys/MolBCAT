"""
Graph Isomorphism Network (GIN) encoder and standalone model.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm


class GINEncoder(nn.Module):
    """
    GIN encoder — used as the graph branch inside MolBCAT.
    Returns per-atom embeddings and batch assignment vector.
    """

    def __init__(self, node_feat_dim: int = 9,
                 hidden_dim: int = 128,
                 num_layers: int = 5):
        super().__init__()

        def make_mlp(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d), nn.ReLU(), nn.Linear(out_d, out_d)
            )

        self.convs       = nn.ModuleList([GINConv(make_mlp(node_feat_dim, hidden_dim))])
        self.batch_norms = nn.ModuleList([BatchNorm(hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(make_mlp(hidden_dim, hidden_dim)))
            self.batch_norms.append(BatchNorm(hidden_dim))

    def forward(self, data) -> tuple:
        """
        Returns:
            x:     Per-atom embeddings, shape (N_atoms, hidden_dim)
            batch: Batch assignment vector, shape (N_atoms,)
        """
        x = data.x
        for conv, bn in zip(self.convs, self.batch_norms):
            x = torch.relu(bn(conv(x, data.edge_index)))
        return x, data.batch


class GINModel(nn.Module):
    """
    Standalone GIN model with a classification or regression head.
    """

    def __init__(self, node_feat_dim: int = 9,
                 hidden_dim: int = 128,
                 num_layers: int = 5):
        super().__init__()

        def make_mlp(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d), nn.ReLU(), nn.Linear(out_d, out_d)
            )

        self.convs       = nn.ModuleList([GINConv(make_mlp(node_feat_dim, hidden_dim))])
        self.batch_norms = nn.ModuleList([BatchNorm(hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(make_mlp(hidden_dim, hidden_dim)))
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: PyG Batch object
        Returns:
            Logits / predictions, shape (B, 1)
        """
        x = data.x
        for conv, bn in zip(self.convs, self.batch_norms):
            x = torch.relu(bn(conv(x, data.edge_index)))
        return self.fc(global_mean_pool(x, data.batch))
