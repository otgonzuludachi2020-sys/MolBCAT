"""
Utility functions — seed, checkpoint, JSON helpers.
"""
import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for full reproducibility across CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str) -> dict:
    """Load a JSON file. Returns empty dict if file does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    """Save data to a JSON file with indentation."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_vocab(vocab_path: str) -> dict:
    """Load vocabulary from JSON. Exits with error if not found."""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"vocab.json not found at: {vocab_path}\n"
            "Please run pretrain first or download weights from Google Drive."
        )
    with open(vocab_path) as f:
        return json.load(f)


def load_pretrained_encoder(model, pretrain_path: str, device: str) -> None:
    """Load pretrained GRU encoder weights into a model."""
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            f"Pretrained encoder not found at: {pretrain_path}\n"
            "Please run pretrain first or download weights from Google Drive."
        )
    state = torch.load(pretrain_path, map_location=device, weights_only=True)
    model.emb.weight.data = state['emb.weight']
    for name, p in model.gru.named_parameters():
        key = f'gru.{name}'
        if key in state:
            p.data = state[key]
    print(f"Pretrained weights loaded from: {pretrain_path}")


def save_checkpoint(model, optimizer, epoch: int, path: str) -> None:
    """Save training checkpoint for resume support."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save({
        'model':   model.state_dict(),
        'opt':     optimizer.state_dict(),
        'epoch':   epoch,
    }, path)


def load_checkpoint(model, optimizer, path: str, device: str) -> int:
    """
    Load checkpoint for resume. Returns the next epoch to start from.
    Returns 1 if no checkpoint found.
    """
    if not os.path.exists(path):
        return 1
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    optimizer.load_state_dict(ckpt['opt'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  Resuming from epoch {start_epoch}")
    return start_epoch


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    return device
