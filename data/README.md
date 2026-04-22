# Datasets

All datasets are loaded automatically from HuggingFace using the `datasets` library.
No manual download is required.

## Classification Datasets

| Dataset     | Task              | HuggingFace Path                                        | Label Column |
|-------------|-------------------|---------------------------------------------------------|--------------|
| BBBP        | Blood-brain barrier penetration | `scikit-fingerprints/MoleculeNet_BBBP`   | `label`      |
| HIV         | HIV replication inhibition      | `scikit-fingerprints/MoleculeNet_HIV`    | `label`      |
| ClinTox     | Clinical toxicity               | `scikit-fingerprints/MoleculeNet_ClinTox`| `CT_TOX`     |
| Tox21_NR_AR | Nuclear receptor activity       | `scikit-fingerprints/MoleculeNet_Tox21`  | `NR-AR`      |

## Regression Datasets

| Dataset        | Task                        | HuggingFace Path                                             | Label Column |
|----------------|-----------------------------|--------------------------------------------------------------|--------------|
| ESOL           | Aqueous solubility (log mol/L) | `scikit-fingerprints/MoleculeNet_ESOL`                    | `label`      |
| Lipophilicity  | Lipophilicity (log D)       | `scikit-fingerprints/MoleculeNet_Lipophilicity`              | `label`      |

## Pretraining Dataset

| Dataset  | Purpose                       | HuggingFace Path         |
|----------|-------------------------------|--------------------------|
| ZINC250k | GRU pretraining (250k SMILES) | `edmanft/zinc250k`       |

## Usage

Datasets are loaded automatically in the training scripts:

```python
from datasets import load_dataset
ds = load_dataset('scikit-fingerprints/MoleculeNet_BBBP')['train']
```

HuggingFace caches datasets at `~/.cache/huggingface` after the first download.
All datasets are publicly available via HuggingFace and correspond to the MoleculeNet benchmark, a widely used standard for molecular property prediction.