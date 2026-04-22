"""
Classification experiment runner.
Runs Phase 1 (HP tuning) and Phase 2 (multi-seed evaluation)
for GRU_Random, GRU_Frozen, GRU_Finetune, GIN, and MolBCAT.

Usage:
    python scripts/train_cls.py --config configs/classification.yaml
    python scripts/train_cls.py --config configs/classification.yaml --dataset BBBP
"""
import os
import sys
import copy
import argparse
import numpy as np
import torch
import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PIN = torch.cuda.is_available()
_NW  = 2  # num_workers for DataLoader

from src.utils import set_seed, get_device, load_vocab, load_json, save_json, load_pretrained_encoder
from src.evaluation import calc_cls_metrics, make_pos_weight, safe_auc, cohens_d_paired
from src.dataset import filter_valid, encode_smiles, make_gru_loaders, scaffold_split
from src.dataset import build_graph_list, CrossAttnDataset, crossattn_collate
from src.models import GRUModel, GINModel, MolBCAT
from src.trainer import train_gru, train_molbcat, infer_gru, infer_gin, infer_molbcat


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification experiments')
    parser.add_argument('--config',  type=str, default='configs/classification.yaml')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset name or "all" (default: all)')
    return parser.parse_args()


def _build_gru(cfg, vocab_size, hidden, device, mode):
    """Build and optionally load pretrained weights for a GRU variant."""
    model = GRUModel(vocab_size, cfg['model']['emb_dim'], hidden).to(device)
    if mode in ('frozen', 'finetune'):
        load_pretrained_encoder(model, cfg['weights']['pretrained_encoder'], device)
    if mode == 'frozen':
        for p in model.emb.parameters(): p.requires_grad = False
        for p in model.gru.parameters(): p.requires_grad = False
    return model


# ── Phase 1 — Hyperparameter tuning ──────────────────────────────────────
def run_phase1(dataset_cfg: dict, cfg: dict, vocab: dict, device: str) -> dict:
    name  = dataset_cfg['name']
    print(f"\n{'='*55}\nPHASE 1 — {name}\n{'='*55}")

    out_dir  = os.path.join(cfg['output']['save_dir'], 'phase1')
    os.makedirs(out_dir, exist_ok=True)
    hp_path  = os.path.join(out_dir, f'{name}_best_hps.json')
    best_hps = load_json(hp_path)

    all_keys = ['GRU_Random', 'GRU_Frozen', 'GRU_Finetune', 'GIN', 'MolBCAT']
    if all(k in best_hps for k in all_keys):
        print("  Already tuned. Skipping.")
        return best_hps

    set_seed(cfg['split']['phase1_seed'])
    ds = load_dataset(dataset_cfg['hf'])['train']
    smiles, labels = filter_valid(ds['SMILES'], ds[dataset_cfg['col']], task='cls')

    train_idx, _ = scaffold_split(smiles, cfg['split']['phase1_seed'],
                                   cfg['split']['test_ratio'])
    X_tr_all = [smiles[i] for i in train_idx]
    y_tr_all = [labels[i] for i in train_idx]

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_all, y_tr_all, test_size=cfg['split']['val_ratio'],
            stratify=y_tr_all, random_state=cfg['split']['phase1_seed'])
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_all, y_tr_all, test_size=cfg['split']['val_ratio'],
            random_state=cfg['split']['phase1_seed'])

    if not X_tr or not X_val:
        print("  Skipping: empty train or val split"); return best_hps

    pos_weight   = make_pos_weight(y_tr, device)
    if pos_weight is None:
        print("  Skipping: degenerate labels"); return best_hps

    train_loader, val_loader = make_gru_loaders(
        X_tr, y_tr, X_val, y_val, vocab,
        cfg['model']['max_len'], cfg['training']['batch_size'])

    vocab_size = len(vocab)

    # GRU variants
    for key, mode, grid_key in [
        ('GRU_Random',   'random',   'gru_random'),
        ('GRU_Frozen',   'frozen',   'gru_frozen'),
        ('GRU_Finetune', 'finetune', 'gru_finetune'),
    ]:
        if key in best_hps:
            continue
        best_trial, best_val = None, -1
        for trial in cfg['grids'][grid_key]:
            set_seed(cfg['split']['phase1_seed'])
            try:
                m = _build_gru(cfg, vocab_size,
                                trial.get('hidden', cfg['model']['hidden_dim']),
                                device, mode)
                _, val_auc = train_gru(m, train_loader, val_loader,
                                       pos_weight, trial['lr'],
                                       cfg['training']['epochs'],
                                       cfg['training']['early_stop'],
                                       'cls', device)
                print(f"  {key} lr={trial['lr']} → {val_auc:.4f}")
                if val_auc > best_val:
                    best_val, best_trial = val_auc, copy.deepcopy(trial)
            except Exception as e:
                print(f"  {key} trial failed: {e}")
        best_hps[key] = best_trial
        save_json(hp_path, best_hps)

    # GIN
    if 'GIN' not in best_hps:
        train_g = build_graph_list(X_tr, y_tr)
        val_g   = build_graph_list(X_val, y_val)
        best_trial, best_val = None, -1
        for trial in cfg['grids']['gin']:
            set_seed(cfg['split']['phase1_seed'])
            try:
                m   = GINModel(cfg['model']['node_feat_dim'],
                               cfg['model']['gin_hidden_dim']).to(device)
                tl  = GeoLoader(train_g, batch_size=cfg['training']['batch_size'], shuffle=True)
                vl  = GeoLoader(val_g,   batch_size=cfg['training']['batch_size'])
                _, val_auc = train_gru(m, tl, vl, pos_weight, trial['lr'],
                                       cfg['training']['epochs'],
                                       cfg['training']['early_stop'], 'cls', device)
                print(f"  GIN lr={trial['lr']} → {val_auc:.4f}")
                if val_auc > best_val:
                    best_val, best_trial = val_auc, copy.deepcopy(trial)
            except Exception as e:
                print(f"  GIN trial failed: {e}")
        best_hps['GIN'] = best_trial
        save_json(hp_path, best_hps)

    # MolBCAT
    if 'MolBCAT' not in best_hps:
        tl = DataLoader(CrossAttnDataset(X_tr,  y_tr,  vocab, cfg['model']['max_len']),
                        batch_size=cfg['training']['batch_size'],
                        shuffle=True, collate_fn=crossattn_collate,
                        num_workers=_NW, pin_memory=_PIN)
        vl = DataLoader(CrossAttnDataset(X_val, y_val, vocab, cfg['model']['max_len']),
                        batch_size=cfg['training']['batch_size'],
                        collate_fn=crossattn_collate,
                        num_workers=_NW, pin_memory=_PIN)
        best_trial, best_val = None, -1
        for trial in cfg['grids']['molbcat']:
            set_seed(cfg['split']['phase1_seed'])
            try:
                m = MolBCAT(vocab_size, cfg['model']['emb_dim'],
                            cfg['model']['hidden_dim'], cfg['model']['gin_hidden_dim'],
                            cfg['model']['node_feat_dim'], cfg['model']['nhead']).to(device)
                m.pad_id = vocab.get('<PAD>', 0)
                load_pretrained_encoder(m.gru_encoder, cfg['weights']['pretrained_encoder'], device)
                _, val_auc, _, _ = train_molbcat(
                    m, tl, vl, pos_weight,
                    trial['lr_encoder'], trial['lr_head'],
                    cfg['training'].get('molbcat_epochs', cfg['training']['epochs']), cfg['training']['early_stop'],
                    'cls', device)
                print(f"  MolBCAT lr_enc={trial['lr_encoder']} lr_head={trial['lr_head']} → {val_auc:.4f}")
                if val_auc > best_val:
                    best_val, best_trial = val_auc, copy.deepcopy(trial)
            except Exception as e:
                print(f"  MolBCAT trial failed: {e}")
        best_hps['MolBCAT'] = best_trial
        save_json(hp_path, best_hps)

    print(f"Phase 1 done: {name}")
    return best_hps


# ── Phase 2 — Multi-seed evaluation ──────────────────────────────────────
def run_phase2(dataset_cfg: dict, cfg: dict, vocab: dict,
               best_hps: dict, device: str) -> dict:
    name = dataset_cfg['name']
    print(f"\n{'='*55}\nPHASE 2 — {name}\n{'='*55}")

    out_dir  = os.path.join(cfg['output']['save_dir'], 'phase2')
    os.makedirs(out_dir, exist_ok=True)
    p2_path  = os.path.join(out_dir, f'{name}.json')
    p2       = load_json(p2_path)
    seeds    = cfg['split']['phase2_seeds']
    remaining = [s for s in seeds if str(s) not in p2]
    print(f"  Remaining seeds: {remaining}")

    if remaining:
        ds = load_dataset(dataset_cfg['hf'])['train']
        smiles, labels = filter_valid(ds['SMILES'], ds[dataset_cfg['col']], task='cls')

        for seed in remaining:
            print(f"\n  Seed {seed}")
            set_seed(seed)
            train_idx, test_idx = scaffold_split(smiles, seed, cfg['split']['test_ratio'])
            X_train = [smiles[i] for i in train_idx]
            y_train = [labels[i] for i in train_idx]
            X_test  = [smiles[i] for i in test_idx]
            y_test  = [labels[i] for i in test_idx]

            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=cfg['split']['val_ratio'],
                    stratify=y_train, random_state=seed)
            except ValueError:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=cfg['split']['val_ratio'],
                    random_state=seed)

            if not X_tr or not X_val or not X_test:
                print(f"  Seed {seed}: empty split, skipping.")
                continue

            pos_weight = make_pos_weight(y_tr, device)
            if pos_weight is None:
                continue

            train_loader, val_loader = make_gru_loaders(
                X_tr, y_tr, X_val, y_val, vocab,
                cfg['model']['max_len'], cfg['training']['batch_size'])
            X_test_t     = torch.tensor([encode_smiles(s, vocab, cfg['model']['max_len'])
                                         for s in X_test])
            seed_results = {}
            vocab_size   = len(vocab)

            ckpt_gru_gin = os.path.join(cfg['weights']['save_dir']['gru_gin'],
                                         name, f'seed{seed}')
            ckpt_molbcat = os.path.join(cfg['weights']['save_dir']['molbcat'],
                                         name, f'seed{seed}')
            os.makedirs(ckpt_gru_gin, exist_ok=True)
            os.makedirs(ckpt_molbcat, exist_ok=True)

            # GRU variants
            for key, mode, hp_key, ckpt_name in [
                ('GRU_Random',   'random',   'GRU_Random',   'GRU_Random.pt'),
                ('GRU_Frozen',   'frozen',   'GRU_Frozen',   'GRU_Frozen.pt'),
                ('GRU_Finetune', 'finetune', 'GRU_Finetune', 'GRU_Finetune.pt'),
            ]:
                try:
                    h = best_hps.get(hp_key, {'hidden': cfg['model']['hidden_dim'], 'lr': 1e-3})
                    set_seed(seed)
                    m = _build_gru(cfg, vocab_size, h.get('hidden', cfg['model']['hidden_dim']),
                                   device, mode)
                    m, _ = train_gru(m, train_loader, val_loader, pos_weight, h['lr'],
                                     cfg['training']['epochs'], cfg['training']['early_stop'],
                                     'cls', device)
                    seed_results[key] = calc_cls_metrics(
                        y_test, infer_gru(m, X_test_t, 'cls', 256, device))
                    torch.save(m.state_dict(), os.path.join(ckpt_gru_gin, ckpt_name))
                except Exception as e:
                    print(f"    {key} failed: {e}")
                    seed_results[key] = {k: float('nan') for k in
                                         ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']}

            # GIN
            try:
                gin_hp  = best_hps.get('GIN', {'lr': 1e-3})
                train_g = build_graph_list(X_tr, y_tr)
                val_g   = build_graph_list(X_val, y_val)
                test_g  = build_graph_list(X_test, y_test)
                if train_g and test_g:
                    set_seed(seed)
                    m_gin = GINModel(cfg['model']['node_feat_dim'],
                                     cfg['model']['gin_hidden_dim']).to(device)
                    tl_g  = GeoLoader(train_g, batch_size=cfg['training']['batch_size'], shuffle=True)
                    vl_g  = GeoLoader(val_g,   batch_size=cfg['training']['batch_size'])
                    m_gin, _ = train_gru(m_gin, tl_g, vl_g, pos_weight, gin_hp['lr'],
                                         cfg['training']['epochs'], cfg['training']['early_stop'],
                                         'cls', device)
                    gin_preds, gin_labels = infer_gin(m_gin, test_g, 'cls', 32, device)
                    seed_results['GIN'] = calc_cls_metrics(gin_labels, gin_preds)
                    torch.save(m_gin.state_dict(), os.path.join(ckpt_gru_gin, 'GIN.pt'))
                else:
                    seed_results['GIN'] = {k: float('nan') for k in
                                           ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']}
            except Exception as e:
                print(f"    GIN failed: {e}")
                seed_results['GIN'] = {k: float('nan') for k in
                                       ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']}

            # MolBCAT
            try:
                ca_hp = best_hps.get('MolBCAT', {'lr_encoder': 1e-5, 'lr_head': 1e-3})
                train_ds = CrossAttnDataset(X_tr,   y_tr,   vocab, cfg['model']['max_len'])
                val_ds   = CrossAttnDataset(X_val,  y_val,  vocab, cfg['model']['max_len'])
                test_ds  = CrossAttnDataset(X_test, y_test, vocab, cfg['model']['max_len'])
                if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
                    raise ValueError("Empty dataset")

                tl      = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                                     shuffle=True, collate_fn=crossattn_collate,
                                     num_workers=_NW, pin_memory=_PIN)
                vl      = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'],
                                     collate_fn=crossattn_collate,
                                     num_workers=_NW, pin_memory=_PIN)
                tl_test = DataLoader(test_ds,  batch_size=cfg['training']['batch_size'],
                                     collate_fn=crossattn_collate,
                                     num_workers=_NW, pin_memory=_PIN)

                set_seed(seed)
                m_ca = MolBCAT(vocab_size, cfg['model']['emb_dim'],
                                cfg['model']['hidden_dim'], cfg['model']['gin_hidden_dim'],
                                cfg['model']['node_feat_dim'], cfg['model']['nhead']).to(device)
                m_ca.pad_id = vocab.get('<PAD>', 0)
                load_pretrained_encoder(m_ca.gru_encoder,
                                         cfg['weights']['pretrained_encoder'], device)
                m_ca, _, tr_curve, vl_curve = train_molbcat(
                    m_ca, tl, vl, pos_weight,
                    ca_hp['lr_encoder'], ca_hp['lr_head'],
                    cfg['training'].get('molbcat_epochs', cfg['training']['epochs']), cfg['training']['early_stop'],
                    'cls', device)

                ca_preds, ca_labels = infer_molbcat(m_ca, tl_test, 'cls', device)
                seed_results['MolBCAT'] = calc_cls_metrics(ca_labels, ca_preds)
                torch.save(m_ca.state_dict(), os.path.join(ckpt_molbcat, 'MolBCAT.pt'))

                # Save learning curves
                curve_path = os.path.join(out_dir, f'{name}_curves.json')
                curves = load_json(curve_path)
                curves[str(seed)] = {'train': tr_curve, 'val': vl_curve}
                save_json(curve_path, curves)

            except Exception as e:
                print(f"    MolBCAT failed: {e}")
                seed_results['MolBCAT'] = {k: float('nan') for k in
                                            ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']}

            p2[str(seed)] = seed_results
            save_json(p2_path, p2)
            print(f"    AUC — Random:{seed_results['GRU_Random']['ROC_AUC']:.4f}  "
                  f"Frozen:{seed_results['GRU_Frozen']['ROC_AUC']:.4f}  "
                  f"FT:{seed_results['GRU_Finetune']['ROC_AUC']:.4f}  "
                  f"GIN:{seed_results['GIN']['ROC_AUC']:.4f}  "
                  f"MolBCAT:{seed_results['MolBCAT']['ROC_AUC']:.4f}")

    return load_json(p2_path)


def summarize(p2_data: dict, dataset_name: str, seeds: list) -> dict:
    """Compute mean ± std across seeds for all models and metrics."""
    result = {'Dataset': dataset_name}
    for model_key in ['GRU_Random', 'GRU_Frozen', 'GRU_Finetune', 'GIN', 'MolBCAT']:
        for metric in ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']:
            vals = [
                p2_data[str(s)][model_key][metric]
                for s in seeds
                if str(s) in p2_data
                and not np.isnan(p2_data[str(s)][model_key][metric])
            ]
            result[f'{model_key}_{metric}_mean'] = float(np.mean(vals)) if vals else float('nan')
            result[f'{model_key}_{metric}_std']  = float(np.std(vals))  if vals else float('nan')

    # Statistical comparison: GRU_Finetune vs MolBCAT
    rnd = [p2_data[str(s)]['GRU_Finetune']['ROC_AUC']
           for s in seeds if str(s) in p2_data]
    ca  = [p2_data[str(s)]['MolBCAT']['ROC_AUC']
           for s in seeds if str(s) in p2_data]
    if len(rnd) >= 2:
        from scipy.stats import ttest_rel
        result['p_FT_vs_MolBCAT'] = float(ttest_rel(rnd, ca).pvalue)
        result['d_FT_vs_MolBCAT'] = cohens_d_paired(rnd, ca)

    return result


def main(args=None):
    if args is None:
        args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    vocab  = load_vocab(cfg['weights']['vocab'])

    # Select datasets
    datasets = (
        cfg['data']['datasets'] if args.dataset == 'all'
        else [d for d in cfg['data']['datasets'] if d['name'] == args.dataset]
    )
    if not datasets:
        print(f"ERROR: Dataset '{args.dataset}' not found in config.")
        sys.exit(1)

    all_results = []
    for ds_cfg in datasets:
        print(f"\n{'#'*60}\n# Dataset: {ds_cfg['name']}\n{'#'*60}")
        try:
            best_hps = run_phase1(ds_cfg, cfg, vocab, device)
            p2_data  = run_phase2(ds_cfg, cfg, vocab, best_hps, device)
            result   = summarize(p2_data, ds_cfg['name'],
                                  cfg['split']['phase2_seeds'])
            all_results.append(result)
        except Exception as e:
            raise RuntimeError(f"Error on {ds_cfg['name']}: {e}")

    if not all_results:
        print("No results to summarize.")
        return

    # Print summary table
    models  = ['GRU_Random', 'GRU_Frozen', 'GRU_Finetune', 'GIN', 'MolBCAT']
    metrics = ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']
    summary = pd.DataFrame(all_results).set_index('Dataset')

    for metric in metrics:
        print(f"\n{'='*90}")
        print(f"CLASSIFICATION — {metric}  (mean ± std, {len(cfg['split']['phase2_seeds'])} seeds, scaffold split)")
        print('=' * 90)
        header = f"{'Dataset':<16}" + ''.join(f'{m:>20}' for m in models)
        print(header)
        print('-' * 90)
        for row_name in summary.index:
            row  = summary.loc[row_name]
            line = f"{row_name:<16}"
            for m in models:
                mean_v = row.get(f'{m}_{metric}_mean', float('nan'))
                std_v  = row.get(f'{m}_{metric}_std',  float('nan'))
                if np.isnan(mean_v):
                    line += f"{'N/A':>20}"
                else:
                    line += f"{mean_v:>8.4f}±{std_v:<8.4f}"
            print(line)

    # Save CSV
    out_csv = os.path.join(cfg['output']['save_dir'], 'classification_summary.csv')
    os.makedirs(cfg['output']['save_dir'], exist_ok=True)
    summary.to_csv(out_csv)
    print(f"\nFull results saved to {out_csv}")


if __name__ == '__main__':
    main()
