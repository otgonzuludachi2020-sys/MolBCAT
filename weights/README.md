# Pretrained Model Weights

All trained model weights are publicly available via Google Drive:

[Download Weights (Google Drive)](https://drive.google.com/drive/u/0/folders/1JcMC1v_mQCMoa0IX91l1UF57_nnXz7Ob)

---

## Google Drive Structure

```
Weights/
├── Classification/
│   ├── Baselines weights/
│   │   ├── BBBP.zip
│   │   ├── ClinTox.zip
│   │   ├── HIV.zip
│   │   └── Tox21_NR_AR.zip
│   ├── MolBCAT weights/
│   │   ├── BBBP.zip
│   │   ├── ClinTox.zip
│   │   ├── HIV.zip
│   │   └── Tox21_NR_AR.zip
│   └── ChemBERTa weights/
│       ├── BBBP.zip
│       ├── ClinTox.zip
│       ├── HIV.zip
│       └── Tox21_NR_AR.zip
├── Pretrained/
│   ├── pretrained_encoder_epoch10.pt
│   └── vocab.json
└── Regression/
    ├── ESOL.zip
    ├── Lipophilicity.zip
    ├── ESOL.ChemBERTa.zip
    └── Lipophilicity.ChemBERTa.zip
```

---

## Setup Instructions

### Step 1: Download pretrained encoder
Download from `Pretrained/` folder and place in `weights/`:
```
weights/
├── vocab.json
└── pretrained_encoder_epoch10.pt
```

### Step 2: Download model weights

**Classification** — download from:

- Classification/Baselines weights/
- Classification/MolBCAT weights/
- Classification/ChemBERTa weights/
```
weights/
├── GRU_GIN/
│   └── {DATASET}/
│       └── seed{1..10}/
│           ├── GRU_Random.pt
│           ├── GRU_Frozen.pt
│           ├── GRU_Finetune.pt
│           └── GIN.pt
├── MolBCAT/
│   └── {DATASET}/
│       └── seed{1..10}/
│           └── MolBCAT.pt
└── ChemBERTa/
    └── {DATASET}/
        └── seed{1..10}/
            └── ChemBERTa.pt       
```

**Regression** — download from `Regression/`, extract into:
```
weights/
└── Regression/
    └── {DATASET}/
        └── seed{1..10}/
            ├── GRU_Random.pt
            ├── GRU_Frozen.pt
            ├── GRU_Finetune.pt
            ├── GIN.pt
            ├── ChemBERTa.pt
            └── MolBCAT_Reg.pt
```

Where `{DATASET}` is one of: `BBBP`, `HIV`, `ClinTox`, `Tox21_NR_AR`, `ESOL`, `Lipophilicity`

---

## Final weights/ folder structure

```
weights/
├── vocab.json
├── pretrained_encoder_epoch10.pt
├── GRU_GIN/
│   └── BBBP/seed1/GRU_Random.pt ...
├── MolBCAT/
│   └── BBBP/seed1/MolBCAT.pt ...
├── ChemBERTa/
│   └── BBBP/seed1/ChemBERTa.pt ...
└── Regression/
    └── ESOL/seed1/
        ├── GRU_Random.pt
        ├── GRU_Frozen.pt
        ├── GRU_Finetune.pt
        ├── GIN.pt
        ├── ChemBERTa.pt
        └── MolBCAT_Reg.pt
```

---

## Notes

- Due to file size limitations, weights are not stored in this repository
- All results in the paper are mean ± std over 10 random seeds (seed 1–10)
- ChemBERTa uses the publicly available pretrained model (DeepChem/ChemBERTa-77M-MTR) from Hugging Face. The Google Drive provides the fine-tuned ChemBERTa checkpoints used in our experiments.
