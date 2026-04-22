"""
Data efficiency experiment.
Evaluates GRU_Random, GRU_Finetune, and MolBCAT across
training data ratios: 10%, 20%, 40%, 60%, 80%, 100%.

Datasets: BBBP (classification), Lipophilicity (regression)
Seeds: 1-10 | Scaffold split

Usage:
    python scripts/train_dataeff.py --config configs/classification.yaml \
                                    --reg_config configs/regression.yaml
"""
import os
import sys
import copy
import argparse
import numpy as np
import torch
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_vocab, load_json, save_json, load_pretrained_encoder
from src.evaluation import safe_auc, calc_reg_metrics, make_pos_weight
from src.dataset import filter_valid, encode_smiles, make_gru_loaders, scaffold_split
from src.dataset import CrossAttnDataset, crossattn_collate
from src.models import GRUModel, MolBCAT
from src.trainer import train_gru, train_molbcat, infer_gru, infer_molbcat


RATIOS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS  = list(range(1, 11))


def parse_args():
    parser = argparse.ArgumentParser(description='Data efficiency experiments')
    parser.add_argument('--cls_config', type=str, default='configs/classification.yaml')
    parser.add_argument('--reg_config', type=str, default='configs/regression.yaml')
    parser.add_argument('--out_dir',    type=str, default='./outputs/data_efficiency')
    return parser.parse_args()


def _get_best_hps(cfg, dataset_name: str, task: str) -> tuple:
    """
    Load best HPs from Phase 1 results.
    Falls back to notebook defaults if Phase 1 has not been run.
    """
    out_dir  = os.path.join(cfg['output']['save_dir'], 'phase1')
    hp_path  = os.path.join(out_dir, f'{dataset_name}_best_hps.json')
    best_hps = load_json(hp_path)

    # Fallback defaults matching original notebook values
    if task == 'reg':
        default_lr_random   = 1e-3
        default_lr_finetune = 5e-5
        default_molbcat     = {'lr_encoder': 1e-5, 'lr_head': 1e-3}
    else:
        default_lr_random   = 1e-4
        default_lr_finetune = 1e-4
        default_molbcat     = {'lr_encoder': 1e-5, 'lr_head': 5e-4}

    lr_random   = best_hps.get('GRU_Random',   {}).get('lr',       default_lr_random)
    lr_finetune = best_hps.get('GRU_Finetune', {}).get('lr',       default_lr_finetune)
    molbcat_hp  = best_hps.get('MolBCAT',      default_molbcat)

    if not best_hps:
        print(f"  Phase 1 not found for {dataset_name} — using default HPs")

    return lr_random, lr_finetune, molbcat_hp


def run_data_efficiency(dataset_cfg: dict, cfg: dict,
                        vocab: dict, device: str,
                        task: str, out_dir: str) -> dict:
    """
    Run data efficiency experiment for one dataset.

    Args:
        dataset_cfg: Dict with name, hf, col.
        cfg:         Full config dict.
        vocab:       Vocabulary dict.
        device:      'cuda' or 'cpu'.
        task:        'cls' or 'reg'.
        out_dir:     Directory to save results.
    """
    name = dataset_cfg['name']
    print(f"\n{'='*55}\nDATA EFFICIENCY — {name}\n{'='*55}")

    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f'{name}_results.json')
    results      = load_json(results_path)
    if results:
        print(f"  Resume: {len(results)} combinations done")

    lr_random, lr_finetune, molbcat_hp = _get_best_hps(cfg, name, task)

    # Use data_efficiency-specific training settings
    de_cfg     = cfg.get('data_efficiency', cfg['training'])
    epochs     = de_cfg['epochs']
    early_stop = de_cfg['early_stop']

    ds = load_dataset(dataset_cfg['hf'])['train']
    smiles, labels = filter_valid(ds['SMILES'], ds[dataset_cfg['col']], task=task)
    vocab_size     = len(vocab)

    for seed in SEEDS:
        set_seed(seed)
        train_idx, test_idx = scaffold_split(smiles, seed, cfg['split']['test_ratio'])
        X_train = [smiles[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test  = [smiles[i] for i in test_idx]
        y_test  = [labels[i] for i in test_idx]

        X_tr_full, X_val, y_tr_full, y_val = train_test_split(
            X_train, y_train, test_size=cfg['split']['val_ratio'],
            random_state=seed)

        for ratio in RATIOS:
            key = f'seed{seed}_ratio{ratio}'
            if (key in results and
                    all(m in results[key]
                        for m in ['GRU_Random', 'GRU_Finetune', 'MolBCAT'])):
                print(f"  Skip {key}")
                continue

            # Subsample training data
            n   = max(int(len(X_tr_full) * ratio), 16)
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(X_tr_full))[:n]
            X_tr = [X_tr_full[i] for i in idx]
            y_tr = [y_tr_full[i] for i in idx]

            print(f"\n  Seed={seed} Ratio={ratio:.0%} "
                  f"(train={n}, val={len(X_val)}, test={len(X_test)})")

            pos_weight = make_pos_weight(y_tr, device, max_weight=10.0) if task == 'cls' else None
            train_loader, val_loader = make_gru_loaders(
                X_tr, y_tr, X_val, y_val, vocab,
                cfg['model']['max_len'], cfg['training']['batch_size'])
            X_test_t = torch.tensor([encode_smiles(s, vocab, cfg['model']['max_len'])
                                      for s in X_test])

            combo_results = results.get(key, {})

            # GRU_Random
            if 'GRU_Random' not in combo_results:
                try:
                    set_seed(seed)
                    m = GRUModel(vocab_size, cfg['model']['emb_dim'],
                                  cfg['model']['hidden_dim']).to(device)
                    m, _ = train_gru(m, train_loader, val_loader,
                                     pos_weight, lr_random,
                                     epochs, early_stop,
                                     task, device)
                    preds = infer_gru(m, X_test_t, task, 256, device)
                    combo_results['GRU_Random'] = (
                        float(safe_auc(y_test, preds)) if task == 'cls'
                        else calc_reg_metrics(y_test, preds)['RMSE']
                    )
                except Exception as e:
                    print(f"    GRU_Random failed: {e}")
                    combo_results['GRU_Random'] = float('nan')

            # GRU_Finetune
            if 'GRU_Finetune' not in combo_results:
                try:
                    set_seed(seed)
                    m = GRUModel(vocab_size, cfg['model']['emb_dim'],
                                  cfg['model']['hidden_dim']).to(device)
                    load_pretrained_encoder(m, cfg['weights']['pretrained_encoder'], device)
                    m, _ = train_gru(m, train_loader, val_loader,
                                     pos_weight, lr_finetune,
                                     epochs, early_stop,
                                     task, device)
                    preds = infer_gru(m, X_test_t, task, 256, device)
                    combo_results['GRU_Finetune'] = (
                        float(safe_auc(y_test, preds)) if task == 'cls'
                        else calc_reg_metrics(y_test, preds)['RMSE']
                    )
                except Exception as e:
                    print(f"    GRU_Finetune failed: {e}")
                    combo_results['GRU_Finetune'] = float('nan')

            # MolBCAT
            if 'MolBCAT' not in combo_results:
                try:
                    train_ds = CrossAttnDataset(X_tr,   y_tr,   vocab, cfg['model']['max_len'])
                    val_ds   = CrossAttnDataset(X_val,  y_val,  vocab, cfg['model']['max_len'])
                    test_ds  = CrossAttnDataset(X_test, y_test, vocab, cfg['model']['max_len'])

                    if len(train_ds) == 0 or len(test_ds) == 0:
                        raise ValueError("Empty dataset")

                    tl      = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                                         shuffle=True, collate_fn=crossattn_collate)
                    vl      = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'],
                                         collate_fn=crossattn_collate)
                    tl_test = DataLoader(test_ds,  batch_size=cfg['training']['batch_size'],
                                         collate_fn=crossattn_collate)

                    set_seed(seed)
                    m = MolBCAT(vocab_size, cfg['model']['emb_dim'],
                                 cfg['model']['hidden_dim'], cfg['model']['gin_hidden_dim'],
                                 cfg['model']['node_feat_dim'], cfg['model']['nhead']).to(device)
                    m.pad_id = vocab.get('<PAD>', 0)
                    load_pretrained_encoder(m.gru_encoder,
                                             cfg['weights']['pretrained_encoder'], device)
                    m, _, _, _ = train_molbcat(
                        m, tl, vl, pos_weight,
                        molbcat_hp['lr_encoder'], molbcat_hp['lr_head'],
                        epochs, early_stop,
                        task, device)

                    preds, trues = infer_molbcat(m, tl_test, task, device)
                    combo_results['MolBCAT'] = (
                        float(safe_auc(trues, preds)) if task == 'cls'
                        else calc_reg_metrics(y_test, preds)['RMSE']
                    )
                except Exception as e:
                    print(f"    MolBCAT failed: {e}")
                    combo_results['MolBCAT'] = float('nan')

            results[key] = combo_results
            save_json(results_path, results)

            metric = 'AUC' if task == 'cls' else 'RMSE'
            print(f"    {metric} — "
                  f"Random:{combo_results['GRU_Random']:.4f}  "
                  f"Finetune:{combo_results['GRU_Finetune']:.4f}  "
                  f"MolBCAT:{combo_results['MolBCAT']:.4f}")

    return results


def aggregate(results: dict) -> dict:
    """Compute mean ± std across seeds for each ratio and model."""
    models  = ['GRU_Random', 'GRU_Finetune', 'MolBCAT']
    summary = {}
    for ratio in RATIOS:
        vals = {m: [] for m in models}
        for seed in SEEDS:
            key = f'seed{seed}_ratio{ratio}'
            if key not in results:
                continue
            for m in models:
                v = results[key].get(m, float('nan'))
                if not np.isnan(v):
                    vals[m].append(v)
        summary[ratio] = {}
        for m in models:
            v = vals[m]
            summary[ratio][f'{m}_mean'] = float(np.mean(v)) if v else float('nan')
            summary[ratio][f'{m}_std']  = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return summary


def plot_data_efficiency(bbbp_summary: dict, lipo_summary: dict, out_dir: str) -> None:
    """Plot data efficiency curves for BBBP and Lipophilicity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ratios    = sorted(bbbp_summary.keys())
    pct       = [int(r * 100) for r in ratios]
    models    = ['GRU_Random', 'GRU_Finetune', 'MolBCAT']
    styles    = {'GRU_Random': ('o--', '#888780'), 'GRU_Finetune': ('s--', '#EF9F27'),
                 'MolBCAT':    ('D-',  '#185FA5')}

    for ax, summary, title, ylabel in [
        (axes[0], bbbp_summary,   'BBBP (↑ higher is better)',         'ROC-AUC'),
        (axes[1], lipo_summary,   'Lipophilicity (↓ lower is better)', 'RMSE'),
    ]:
        for m in models:
            means = [summary[r].get(f'{m}_mean', float('nan')) for r in ratios]
            stds  = [summary[r].get(f'{m}_std',  0.0)          for r in ratios]
            marker, color = styles[m]
            ax.plot(pct, means, marker, color=color, label=m, linewidth=2)
            ax.fill_between(pct,
                             [mu - s for mu, s in zip(means, stds)],
                             [mu + s for mu, s in zip(means, stds)],
                             alpha=0.15, color=color)
        ax.set_title(title)
        ax.set_xlabel('Training Data Size (%)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(pct)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'data_efficiency.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def main(args=None):
    if args is None:
        args = parse_args()

    with open(args.cls_config) as f:
        cls_cfg = yaml.safe_load(f)
    with open(args.reg_config) as f:
        reg_cfg = yaml.safe_load(f)

    device = get_device()
    vocab  = load_vocab(cls_cfg['weights']['vocab'])

    os.makedirs(args.out_dir, exist_ok=True)

    # BBBP — classification
    bbbp_cfg = next(d for d in cls_cfg['data']['datasets'] if d['name'] == 'BBBP')
    bbbp_res = run_data_efficiency(bbbp_cfg, cls_cfg, vocab, device,
                                    task='cls', out_dir=args.out_dir)
    bbbp_summary = aggregate(bbbp_res)

    # Lipophilicity — regression
    lipo_cfg = next(d for d in reg_cfg['data']['datasets']
                    if d['name'] == 'Lipophilicity')
    lipo_res = run_data_efficiency(lipo_cfg, reg_cfg, vocab, device,
                                    task='reg', out_dir=args.out_dir)
    lipo_summary = aggregate(lipo_res)

    # Print summary table
    print(f"\n{'='*60}")
    print("DATA EFFICIENCY SUMMARY")
    print(f"{'='*60}")
    for ratio in RATIOS:
        pct = int(ratio * 100)
        b   = bbbp_summary[ratio]
        l   = lipo_summary[ratio]
        print(f"\n  {pct}% training data:")
        print(f"    BBBP (AUC)  — "
              f"Random:{b['GRU_Random_mean']:.4f}  "
              f"Finetune:{b['GRU_Finetune_mean']:.4f}  "
              f"MolBCAT:{b['MolBCAT_mean']:.4f}")
        print(f"    Lipo (RMSE) — "
              f"Random:{l['GRU_Random_mean']:.4f}  "
              f"Finetune:{l['GRU_Finetune_mean']:.4f}  "
              f"MolBCAT:{l['MolBCAT_mean']:.4f}")

    # Save summary CSV
    rows = []
    for ratio in RATIOS:
        row = {'ratio': f'{int(ratio*100)}%'}
        for m in ['GRU_Random', 'GRU_Finetune', 'MolBCAT']:
            row[f'BBBP_{m}_mean'] = bbbp_summary[ratio].get(f'{m}_mean', float('nan'))
            row[f'BBBP_{m}_std']  = bbbp_summary[ratio].get(f'{m}_std',  float('nan'))
            row[f'Lipo_{m}_mean'] = lipo_summary[ratio].get(f'{m}_mean', float('nan'))
            row[f'Lipo_{m}_std']  = lipo_summary[ratio].get(f'{m}_std',  float('nan'))
        rows.append(row)

    out_csv = os.path.join(args.out_dir, 'data_efficiency_summary.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSummary saved to {out_csv}")

    # Plot
    plot_data_efficiency(bbbp_summary, lipo_summary, args.out_dir)


if __name__ == '__main__':
    main()
