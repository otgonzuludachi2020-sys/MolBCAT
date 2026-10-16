# MolBCAT: Self-Supervised GRU Pretraining with Bidirectional Cross-Modal Attention for Molecular Property Prediction

Official implementation of the paper:

> **MolBCAT: Self-supervised GRU pretraining with bidirectional cross-modal attention for molecular property prediction**

**Otgonzul Zorigt, Soualihou Ngnamsie Njimbouom, Candra Zonyfar, Jeong-Dong Kim**

---

## Overview

MolBCAT is a multimodal molecular property prediction framework that integrates:

- **Self-supervised pretrained GRU encoder** for SMILES sequences
- **Graph Isomorphism Network (GIN)** for molecular graphs
- **Bidirectional cross-modal attention** for dynamic interaction between SMILES and graph representations
- **Gated fusion mechanism** for adaptive multimodal representation learning

The repository also includes baseline implementations of:

- GRU
- GIN
- ChemBERTa

These models are evaluated under the same experimental protocol as MolBCAT.

---

## Repository Structure

```text
molbcat/
│
├── main.py                         # Primary CLI entry point
│
├── configs/
│   ├── classification.yaml         # Classification configuration
│   ├── regression.yaml             # Regression configuration
│   └── pretrain.yaml               # GRU pretraining configuration
│
├── data/
│   └── README.md                   # Dataset information
│
├── src/
│   ├── models/
│   │   ├── gru.py                  # GRU encoder and models
│   │   ├── gin.py                  # GIN encoder and model
│   │   ├── chemberta.py            # ChemBERTa model
│   │   └── molbcat.py              # MolBCAT model
│   │
│   ├── dataset/
│   │   ├── smiles.py               # SMILES encoding and datasets
│   │   ├── graph.py                # Molecular graph construction
│   │   └── split.py                # Scaffold-based splitting
│   │
│   ├── checkpoint.py               # Fold-level checkpoint and resume
│   ├── trainer.py                  # Unified classification/regression trainer
│   ├── evaluation.py               # Evaluation metrics
│   └── utils.py                    # General utility functions
│
├── scripts/
│   ├── pretrain.py                 # GRU pretraining on ZINC250K
│   ├── train_cls.py                # Classification experiments
│   ├── train_reg.py                # Regression experiments
│   ├── train_ablation.py           # Ablation experiments
│   └── predict.py                  # Molecular property prediction
│
├── checkpoints/                    # Saved experiment checkpoints
├── results/                        # Experimental results
├── weights/                        # Pretrained/fine-tuned model weights
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install torch==2.10.0

pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

pip install torch-sparse \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

pip install torch-geometric

pip install -r requirements.txt
```

> **Note:** The commands above assume CUDA 12.8 (`cu128`).
> For CPU-only or other CUDA versions, replace the corresponding
> PyTorch Geometric wheel with the appropriate version.

---

## Pretrained Weights

Download the pretrained weights and place them in the `weights/` directory.

Directory Structure

weights/
├── vocab.json
├── pretrained_encoder_epoch10.pt
├── ESOL/
│   ├── fold_1_MolBCAT.pt
│   ├── fold_2_MolBCAT.pt
│   ├── fold_3_MolBCAT.pt
│   ├── fold_4_MolBCAT.pt
│   └── fold_5_MolBCAT.pt
└── Lipophilicity/
    ├── fold_1_MolBCAT.pt
    ├── fold_2_MolBCAT.pt
    ├── fold_3_MolBCAT.pt
    ├── fold_4_MolBCAT.pt
    └── fold_5_MolBCAT.pt

Available Checkpoints

ESOL: Five fold-level MolBCAT checkpoints.
Lipophilicity: Five fold-level MolBCAT checkpoints.
Pretrained encoder: pretrained_encoder_epoch10.pt
Vocabulary: vocab.json

Download pretrained weights:

**Google Drive:**  
https://drive.google.com/drive/u/0/folders/1WtWhfcbG1HVKLm0ZiR6H8VHdiGX7alp8

> **ChemBERTa:** The publicly available
> `DeepChem/ChemBERTa-77M-MTR` pretrained model is loaded automatically
> from Hugging Face. Fine-tuned ChemBERTa checkpoints used in the
> experiments can also be provided in the `weights/` directory.

---

# Usage

## Step 1: GRU Pretraining

GRU pretraining on ZINC250K is optional when pretrained weights are already available.

```bash
python main.py pretrain
```

The pretrained GRU encoder is used by:

- GRU_Frozen
- GRU_Finetune
- MolBCAT

ChemBERTa uses its own pretrained language model.

---

## Step 2: Classification Experiments

Run all classification datasets:

```bash
python main.py train_cls
```

Run a single dataset:

```bash
python main.py train_cls --dataset BBBP
```

### Resume an interrupted experiment

Completed folds are skipped when `--resume` is specified:

```bash
python main.py train_cls --resume
```

For a single dataset:

```bash
python main.py train_cls --dataset BBBP --resume
```

The checkpoint mechanism stores completed fold results so that interrupted
experiments do not require completed folds to be recomputed.

---

## Step 3: Regression Experiments

Run all regression datasets:

```bash
python main.py train_reg
```

Run a single dataset:

```bash
python main.py train_reg --dataset ESOL
```

Resume an interrupted regression experiment:

```bash
python main.py train_reg --resume
```

---


## Step 4: Ablation Experiments

```bash
python main.py train_ablation
```

---

## Prediction

Prediction is currently supported for the released MolBCAT
regression checkpoints for ESOL and Lipophilicity.

### Single SMILES

Predict the ESOL property:

```bash
python scripts/predict.py \
    --dataset ESOL \
    --model MolBCAT \
    --fold 1 \
    --smiles "CCO"

Predict the Lipophilicity property:

python scripts/predict.py \
    --dataset Lipophilicity \
    --model MolBCAT \
    --fold 1 \
    --smiles "CCO"


CSV Input
Prepare a CSV file named molecules.csv with a smiles column:

smiles
CCO
CC(=O)O
c1ccccc1

Run prediction on the ESOL dataset:

python scripts/predict.py \
    --dataset ESOL \
    --input molecules.csv \
    --output results_esol.csv \
    --model MolBCAT \
    --fold 1

Run prediction on Lipophilicity:

python scripts/predict.py \
    --dataset Lipophilicity \
    --input molecules.csv \
    --output results_lipophilicity.csv \
    --model MolBCAT \
    --fold 1

Selecting a Fold

Use --fold to select a checkpoint from folds 1 to 5. The default is fold 1.

The checkpoints must be available in the corresponding weights/ESOL/ or weights/Lipophilicity/ directory. The weights/vocab.json file is also required.

Note: Prediction is currently supported only for the released MolBCAT regression checkpoints for ESOL and Lipophilicity. Checkpoint availability for other datasets and model architectures is not implied.

# Datasets

The repository uses the following molecular property prediction benchmarks.

| Dataset       | Task           | Metric  |
|---------------|----------------|---------|
| BBBP          | Classification | ROC-AUC |
| HIV           | Classification | ROC-AUC |
| ClinTox       | Classification | ROC-AUC |
| Tox21_NR_AR   | Classification | ROC-AUC |
| ESOL          | Regression     | RMSE    |
| Lipophilicity | Regression     | RMSE    |

Additional classification metrics are reported for the revised evaluation protocol.

---

# Experimental Protocol

## Classification

Classification experiments use a **5-fold scaffold-aware outer cross-validation** protocol.

For each outer fold:

```text
Full dataset
     │
     ├── Development
     │       │
     │       └── Inner scaffold-disjoint validation
     │
     └── Untouched test fold
```

The outer test fold is kept completely separate from model and
hyperparameter selection.

### Scaffold separation

Bemis–Murcko scaffolds are used to construct the splits.

The following scaffold overlaps are checked:

- Train vs. validation
- Train vs. test
- Validation vs. test

The expected scaffold intersection is zero for all pairs.

### Hyperparameter tuning

Hyperparameters are selected **within each outer fold** using only the
training and validation data of that fold.

The outer test fold is not used for hyperparameter selection.

### Classification threshold

The classification threshold is selected using the validation set only.

Candidate thresholds are evaluated over:

```text
0.05, 0.06, ..., 0.94, 0.95
```

The threshold that maximizes validation F1 is selected and then frozen
before evaluating the outer test fold.

The test set is therefore not used to determine the classification
threshold.

### Classification metrics

The following metrics are reported:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- MCC

ROC-AUC and PR-AUC are threshold-independent.
Precision, Recall, F1, and MCC use the threshold selected on the
validation set.

---

# Regression

Regression experiments use the same **5-fold scaffold-aware outer
cross-validation** framework.

For each outer fold:

```text
Full dataset
     │
     ├── Development
     │       │
     │       └── Inner scaffold-disjoint validation
     │
     └── Untouched test fold
```

Hyperparameters are selected using only the training and validation
data within each outer fold.

The test fold is evaluated only after model selection is complete.

### Regression metrics

- RMSE
- MAE
- R²

---

# Statistical Analysis

Performance is evaluated across the outer folds using paired comparisons
between models on the same test folds.

For classification model comparisons, p-values are corrected using the
**Holm multiple-comparison correction** across the prespecified
dataset-level comparisons.

Effect sizes are also reported where applicable.

Mean and standard deviation are calculated using the fold-level results.

---

# Reproducibility

The revised evaluation protocol uses:

- 5-fold scaffold-aware outer cross-validation
- Inner scaffold-disjoint validation
- Fold-specific hyperparameter selection
- Validation-only threshold selection for classification
- Untouched outer test folds
- Incremental result/checkpoint saving
- Resume support for interrupted experiments

Random seeds are controlled through the utility functions in
`src/utils.py`.

---

# Checkpoints and Resume

Experiment-level checkpoint handling is implemented in:

```text
src/checkpoint.py
```

The checkpoint system is separate from the general utility functions in
`src/utils.py`.

Example:

```text
Fold 1  ✓ completed
Fold 2  ✓ completed
Fold 3  ✓ completed
Fold 4  ✗ interrupted
Fold 5  -
```

Running:

```bash
python main.py train_cls --resume
```

will skip completed folds and continue the remaining experiment.

---

# Data Splitting

All final classification and regression experiments use
**scaffold-aware splitting** based on Bemis–Murcko scaffolds.

The outer test fold is scaffold-disjoint from the development data.

The inner validation split is also scaffold-disjoint from the training
data.

This protocol is intended to reduce structural similarity between
training and evaluation molecules and provide a more stringent estimate
of generalization to structurally distinct compounds.

---

# License

This project is licensed under the MIT License.
