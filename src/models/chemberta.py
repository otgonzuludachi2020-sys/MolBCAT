"""
ChemBERTa encoder and standalone model.
"""
import torch.nn as nn
from transformers import AutoModel

CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"


class ChemBERTaModel(nn.Module):
    """
    Standalone ChemBERTa model with a classification or regression head.
    """

    def __init__(self, dropout=0.2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(CHEMBERTA_MODEL)

        hidden_size = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)