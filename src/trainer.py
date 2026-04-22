"""
Unified trainer for GRU, GIN, and MolBCAT models.
Supports both classification (BCE) and regression (MSE) tasks.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoLoader

from src.evaluation import safe_auc, calc_cls_metrics, calc_reg_metrics


def train_gru(model, train_loader, val_loader,
              pos_weight=None, lr: float = 1e-3,
              epochs: int = 20, early_stop: int = 8,
              task: str = 'cls', device: str = 'cpu') -> tuple:
    """
    Train a GRU or GIN model.

    Args:
        model:        GRUModel or GINModel instance.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        pos_weight:   Class weight tensor for imbalanced classification.
        lr:           Learning rate.
        epochs:       Maximum number of epochs.
        early_stop:   Patience for early stopping.
        task:         'cls' for classification, 'reg' for regression.
        device:       'cuda' or 'cpu'.

    Returns:
        (model, best_val_metric)
    """
    opt     = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if task == 'cls' else nn.MSELoss()
    )

    best_metric = 0.0 if task == 'cls' else float('inf')
    best_state  = copy.deepcopy(model.state_dict())
    patience    = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            if hasattr(batch, 'edge_index'):       # GIN
                batch = batch.to(device)
                out   = model(batch).view(-1)
                loss  = loss_fn(out, batch.y)
            else:                                   # GRU
                xb, yb = batch
                out  = model(xb.to(device)).view(-1)
                loss = loss_fn(out, yb.to(device))
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                if hasattr(batch, 'edge_index'):
                    batch = batch.to(device)
                    out   = model(batch).view(-1)
                    preds.extend(torch.sigmoid(out).cpu().numpy()
                                 if task == 'cls' else out.cpu().numpy())
                    trues.extend(batch.y.cpu().numpy())
                else:
                    xb, yb = batch
                    out = model(xb.to(device)).view(-1)
                    preds.extend(torch.sigmoid(out).cpu().numpy()
                                 if task == 'cls' else out.cpu().numpy())
                    trues.extend(yb.cpu().numpy())

        if task == 'cls':
            metric   = safe_auc(trues, preds)
            if np.isnan(metric):
                continue
            improved = metric > best_metric + 1e-4
        else:
            metric = calc_reg_metrics(trues, preds)['RMSE']
            improved = metric < best_metric - 1e-4

        if improved:
            best_metric = metric
            best_state  = copy.deepcopy(model.state_dict())
            patience    = 0
        else:
            patience += 1
            if patience >= early_stop:
                print(f"    Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, best_metric


def train_molbcat(model, train_loader, val_loader,
                  pos_weight=None,
                  lr_encoder: float = 1e-5,
                  lr_head: float = 1e-3,
                  epochs: int = 30,
                  early_stop: int = 8,
                  task: str = 'cls',
                  device: str = 'cpu') -> tuple:
    """
    Train MolBCAT with separate learning rates for encoder and head.
    Freezes the GRU encoder for the first 3 epochs, then unfreezes.

    Returns:
        (model, best_val_metric, train_curve, val_curve)
    """
    encoder_params = (
        list(model.gru_encoder.parameters()) +
        list(model.gin_encoder.parameters()) +
        list(model.gin_proj.parameters()) +
        list(model.attn_sg.parameters()) +
        list(model.attn_gs.parameters()) +
        list(model.norm.parameters()) +
        list(model.seq_gate.parameters())
    )
    head_params = (
        list(model.gate_fc.parameters()) +
        list(model.head.parameters())
    )

    opt = torch.optim.Adam([
        {'params': encoder_params, 'lr': lr_encoder},
        {'params': head_params,    'lr': lr_head},
    ])
    loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if task == 'cls' else nn.MSELoss()
    )

    best_metric  = 0.0 if task == 'cls' else float('inf')
    best_state   = copy.deepcopy(model.state_dict())
    patience     = 0
    train_curve, val_curve = [], []

    for epoch in range(epochs):
        # Freeze GRU encoder for the first 3 epochs (epoch 0, 1, 2), then unfreeze
        req_grad = epoch >= 3
        for p in model.gru_encoder.parameters():
            p.requires_grad = req_grad

        model.train()
        for smiles_x, graph_data, labs in train_loader:
            smiles_x   = smiles_x.to(device)
            graph_data = graph_data.to(device)
            labs       = labs.to(device)
            opt.zero_grad()
            loss_fn(model(smiles_x, graph_data).view(-1), labs).backward()
            opt.step()

        model.eval()

        def _infer(loader):
            ps, ts = [], []
            with torch.no_grad():
                for smiles_x, graph_data, labs in loader:
                    out = model(smiles_x.to(device),
                                graph_data.to(device)).view(-1)
                    ps.extend((torch.sigmoid(out) if task == 'cls'
                                else out).cpu().numpy())
                    ts.extend(labs.cpu().numpy())
            return ps, ts

        tr_p, tr_t = _infer(train_loader)
        vl_p, vl_t = _infer(val_loader)

        if task == 'cls':
            if len(set(vl_t)) < 2:
                continue
            tr_m = safe_auc(tr_t, tr_p)
            vl_m = safe_auc(vl_t, vl_p)
            if np.isnan(vl_m):
                continue
            improved = vl_m > best_metric + 1e-4
        else:
            tr_m = calc_reg_metrics(tr_t, tr_p)['RMSE'] 
            vl_m = calc_reg_metrics(vl_t, vl_p)['RMSE'] 
            improved = vl_m < best_metric - 1e-4

        train_curve.append(tr_m)
        val_curve.append(vl_m)
        print(f"  Epoch {epoch + 1} | train={tr_m:.4f} | val={vl_m:.4f}")

        if improved:
            best_metric = vl_m
            best_state  = copy.deepcopy(model.state_dict())
            patience    = 0
        else:
            patience += 1
            if patience >= early_stop:
                print(f"    Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, best_metric, train_curve, val_curve


def infer_gru(model, X_tensor: torch.Tensor,
              task: str = 'cls',
              batch_size: int = 256,
              device: str = 'cpu') -> np.ndarray:
    """Run batch inference for GRU model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(X_tensor), batch_size=batch_size):
            out = model(xb.to(device)).view(-1)
            preds.extend((torch.sigmoid(out) if task == 'cls'
                          else out).cpu().numpy())
    return np.array(preds)


def infer_gin(model, graph_list: list,
              task: str = 'cls',
              batch_size: int = 32,
              device: str = 'cpu') -> tuple:
    """Run batch inference for GIN model."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in GeoLoader(graph_list, batch_size=batch_size):
            batch = batch.to(device)
            out   = model(batch).view(-1)
            preds.extend((torch.sigmoid(out) if task == 'cls'
                          else out).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    return np.array(preds), np.array(labels)


def infer_molbcat(model, loader,
                  task: str = 'cls',
                  device: str = 'cpu') -> tuple:
    """Run batch inference for MolBCAT model."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for smiles_x, graph_data, labs in loader:
            out = model(smiles_x.to(device), graph_data.to(device)).view(-1)
            preds.extend((torch.sigmoid(out) if task == 'cls'
                          else out).cpu().numpy())
            labels.extend(labs.cpu().numpy())
    return np.array(preds), np.array(labels)
