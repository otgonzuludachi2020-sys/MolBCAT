"""
Inference script — predict molecular properties using pretrained weights.

Usage:
    # Single SMILES
    python scripts/predict.py \
        --weights_dir weights \
        --dataset BBBP \
        --smiles "CC(=O)Oc1ccccc1C(=O)O"

    # CSV file (must have a 'smiles' column)
    python scripts/predict.py \
        --weights_dir weights \
        --dataset BBBP \
        --input molecules.csv \
        --output results.csv \
        --model MolBCAT \
        --seed 1

Available datasets:
    Classification: BBBP, HIV, ClinTox, Tox21_NR_AR
    Regression:     ESOL, Lipophilicity
"""
import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, load_vocab, load_json
from src.models import GRUModel, GINModel, MolBCAT


# Checkpoint path templates matching the weights/ folder structure
CLS_DATASETS = {'BBBP', 'HIV', 'ClinTox', 'Tox21_NR_AR'}
REG_DATASETS = {'ESOL', 'Lipophilicity'}

GRU_GIN_MODELS = {'GRU_Random', 'GRU_Frozen', 'GRU_Finetune', 'GIN'}


def parse_args():
    parser = argparse.ArgumentParser(description='MolBCAT inference')
    parser.add_argument('--weights_dir', type=str, default='./weights',
                        help='Path to weights folder (downloaded from Google Drive)')
    parser.add_argument('--dataset',     type=str, required=True,
                        choices=sorted(CLS_DATASETS | REG_DATASETS),
                        help='Dataset / task to predict for')
    parser.add_argument('--model',       type=str, default='MolBCAT',
                        choices=['MolBCAT', 'GRU_Finetune', 'GRU_Frozen',
                                 'GRU_Random', 'GIN'],
                        help='Which model to use (default: MolBCAT)')
    parser.add_argument('--seed',        type=int, default=1,
                        help='Which seed checkpoint to load (1-10, default: 1)')
    parser.add_argument('--smiles',      type=str, default=None,
                        help='Single SMILES string to predict')
    parser.add_argument('--input',       type=str, default=None,
                        help='CSV file with a "smiles" column')
    parser.add_argument('--output',      type=str, default=None,
                        help='Save results to this CSV path')
    return parser.parse_args()


def _get_ckpt_path(weights_dir: str, model_name: str,
                   dataset: str, seed: int) -> str:
    """Resolve checkpoint path from the weights/ folder structure."""
    if dataset in REG_DATASETS:
        ckpt_name = 'MolBCAT_Reg.pt' if model_name == 'MolBCAT' else f'{model_name}.pt'
        return os.path.join(weights_dir, 'Regression',
                            dataset, f'seed{seed}', ckpt_name)
    if model_name == 'MolBCAT':
        return os.path.join(weights_dir, 'MolBCAT',
                            dataset, f'seed{seed}', 'MolBCAT.pt')
    return os.path.join(weights_dir, 'GRU_GIN',
                        dataset, f'seed{seed}', f'{model_name}.pt')


def load_model(args, vocab: dict, device: str):
    """Load the appropriate model and checkpoint."""
    ckpt_path = _get_ckpt_path(args.weights_dir, args.model, args.dataset, args.seed)

    if not os.path.exists(ckpt_path):
        print(f"\nERROR: Checkpoint not found:\n  {ckpt_path}")
        print("\nExpected weights/ folder structure:")
        print("  weights/")
        print("  ├── vocab.json")
        print("  ├── pretrained_encoder_epoch10.pt")
        print("  ├── GRU_GIN/{dataset}/seed{N}/{model}.pt")
        print("  ├── MolBCAT/{dataset}/seed{N}/MolBCAT.pt")
        print("  └── Regression/{dataset}/seed{N}/{model}.pt")
        print("\nDownload from Google Drive — see README.md")
        sys.exit(1)

    vocab_size = len(vocab)

    if args.model == 'MolBCAT':
        model = MolBCAT(vocab_size)
        model.pad_id = vocab.get('<PAD>', 0)
    elif args.model == 'GIN':
        model = GINModel()
    else:
        # hidden size를 checkpoint에서 자동 감지
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        hidden_dim = ckpt['fc.weight'].shape[1]
        model = GRUModel(vocab_size, hidden_dim=hidden_dim)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded : {args.model} | {args.dataset} | seed{args.seed}")
    print(f"Device : {device}")
    return model


def predict(smiles_list: list, model, vocab: dict,
            model_name: str, device: str, dataset: str,
            max_len: int = 120) -> np.ndarray:
    """
    Run inference on a list of SMILES strings.

    Returns:
        Array of predicted probabilities (cls) or values (reg).
        NaN for invalid SMILES.
    """
    from rdkit import Chem
    from src.dataset import encode_smiles, CrossAttnDataset, crossattn_collate
    from src.dataset import build_graph_list
    from torch.utils.data import DataLoader, TensorDataset
    from torch_geometric.loader import DataLoader as GeoLoader
    from src.models import GINModel, MolBCAT
    from src.trainer import infer_gru, infer_gin, infer_molbcat

    task = 'cls' if dataset in CLS_DATASETS else 'reg'

    # Separate valid from invalid
    valid_smiles, valid_idx = [], []
    for i, s in enumerate(smiles_list):
        if Chem.MolFromSmiles(str(s)) is not None:
            valid_smiles.append(s)
            valid_idx.append(i)
        else:
            print(f"  WARNING: Invalid SMILES (idx {i}): {s}")

    if not valid_smiles:
        print("ERROR: No valid SMILES found.")
        sys.exit(1)

    probs = np.full(len(smiles_list), np.nan)

    if model_name == 'MolBCAT':
        dummy_labels = [0.0] * len(valid_smiles)
        test_ds  = CrossAttnDataset(valid_smiles, dummy_labels, vocab, max_len)
        loader   = DataLoader(test_ds, batch_size=32,
                               collate_fn=crossattn_collate, num_workers=0)
        p, _     = infer_molbcat(model, loader, task, device)

    elif model_name == 'GIN':
        graph_list = build_graph_list(valid_smiles, [0.0] * len(valid_smiles))
        p, _       = infer_gin(model, graph_list, task, 32, device)

    else:  # GRU variants
        X = torch.tensor([encode_smiles(s, vocab, max_len) for s in valid_smiles])
        p = infer_gru(model, X, task, 256, device)

    for i, prob in zip(range(len(valid_smiles)), p):
        probs[valid_idx[i]] = prob

    return probs


def print_results(smiles_list: list, probs: np.ndarray,
                  dataset: str, model_name: str,
                  threshold: float = 0.5) -> None:
    is_cls = dataset in CLS_DATASETS
    print(f"\n{'─'*60}")
    print(f"  Model: {model_name}  |  Dataset: {dataset}")
    if is_cls:
        print(f"  Threshold: {threshold}")
    print(f"{'─'*60}")
    print(f"  {'SMILES':<38} {'Value':>8}  Label")
    print(f"{'─'*60}")
    for s, p in zip(smiles_list, probs):
        if np.isnan(p):
            label, val_str = 'INVALID', '     N/A'
        elif is_cls:
            label   = 'POSITIVE' if p >= threshold else 'negative'
            val_str = f'{p:.4f}'
        else:
            label   = f'{p:.4f}'
            val_str = f'{p:.4f}'
        display = s if len(s) <= 38 else s[:35] + '...'
        print(f"  {display:<38} {val_str:>8}  {label}")
    print(f"{'─'*60}\n")


def main(args=None):
    if args is None:
        args = parse_args()

    if args.smiles is None and args.input is None:
        print("ERROR: Provide --smiles or --input.")
        sys.exit(1)

    device = get_device()
    vocab  = load_vocab(os.path.join(args.weights_dir, 'vocab.json'))
    model  = load_model(args, vocab, device)

    # Read max_len from config (dataset type determines which config to use)
    import yaml
    _cfg_file = ('configs/regression.yaml'
                 if args.dataset in REG_DATASETS
                 else 'configs/classification.yaml')
    try:
        with open(_cfg_file) as f:
            _cfg = yaml.safe_load(f)
        max_len = _cfg['model']['max_len']
    except Exception:
        max_len = 120  # safe fallback

    # Collect SMILES
    if args.smiles:
        smiles_list = [args.smiles]
    else:
        df_in = pd.read_csv(args.input)
        if 'smiles' not in df_in.columns:
            print("ERROR: Input CSV must have a 'smiles' column.")
            sys.exit(1)
        smiles_list = df_in['smiles'].tolist()

    probs = predict(smiles_list, model, vocab, args.model, device, args.dataset, max_len)
    print_results(smiles_list, probs, args.dataset, args.model)

    # Save CSV
    if args.output:
        is_cls = args.dataset in CLS_DATASETS
        pd.DataFrame({
            'smiles':      smiles_list,
            'value':       probs,
            'prediction':  [
                ('positive' if p >= 0.5 else 'negative') if is_cls and not np.isnan(p)
                else 'invalid' if np.isnan(p)
                else f'{p:.4f}'
                for p in probs
            ],
        }).to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
