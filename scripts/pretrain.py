"""
Pretrain a masked SMILES language model on ZINC250k.

Usage:
    python scripts/pretrain.py --config configs/pretrain.yaml
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain SMILES encoder on ZINC250k')
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml')
    return parser.parse_args()


class SMILESEncoder(nn.Module):
    """Masked SMILES language model encoder."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru  = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.head(h)


class MaskedSMILESDataset(Dataset):
    """Dataset for masked SMILES language model pretraining."""

    def __init__(self, smiles: list, vocab: dict, max_len: int, mask_prob: float):
        self.data      = [self._encode(s, vocab, max_len) for s in smiles]
        self.vocab     = vocab
        self.mask_prob = mask_prob

    @staticmethod
    def _encode(smiles: str, vocab: dict, max_len: int) -> list:
        cls_id = vocab['<CLS>']
        unk_id = vocab['<UNK>']
        pad_id = vocab['<PAD>']
        seq    = [cls_id] + [vocab.get(c, unk_id) for c in smiles][:max_len - 1]
        seq   += [pad_id] * (max_len - len(seq))
        return seq

    def _mask(self, seq: list) -> tuple:
        """Apply BERT-style masking: 80% MASK, 10% random, 10% unchanged."""
        inp, tgt = seq.copy(), [-100] * len(seq)
        for i in range(1, len(seq)):
            if seq[i] == self.vocab['<PAD>']:
                continue
            if random.random() < self.mask_prob:
                tgt[i] = seq[i]
                r = random.random()
                if r < 0.8:
                    inp[i] = self.vocab['<MASK>']
                elif r < 0.9:
                    inp[i] = random.randint(4, len(self.vocab) - 1)
        return inp, tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self._mask(self.data[idx])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def main(args=None):
    if args is None:
        args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    set_seed(cfg['training']['seed'])
    os.makedirs(cfg['output']['save_dir'], exist_ok=True)

    # Load dataset
    print("Loading ZINC250k...")
    zinc        = load_dataset(cfg['data']['dataset'], split='train')
    smiles_list = zinc['smiles'][:cfg['data']['num_samples']]
    print(f"  SMILES count: {len(smiles_list)}")

    # Build vocabulary
    print("Building vocabulary...")
    vocab_path = os.path.join(cfg['output']['save_dir'], cfg['output']['vocab_file'])
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f"  Loaded existing vocab: {len(vocab)} tokens")
    else:
        chars = sorted(set(''.join(smiles_list)))
        vocab = {c: i + 4 for i, c in enumerate(chars)}
        vocab['<PAD>']  = 0
        vocab['<MASK>'] = 1
        vocab['<CLS>']  = 2
        vocab['<UNK>']  = 3
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        print(f"  Built vocab: {len(vocab)} tokens → saved to {vocab_path}")

    # Dataset and DataLoader
    dataset = MaskedSMILESDataset(
        smiles_list, vocab,
        cfg['training']['max_len'],
        cfg['training']['mask_prob']
    )
    pin_memory = (device == 'cuda')
    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True, drop_last=True,
        num_workers=0, pin_memory=pin_memory
    )

    # Model, optimizer, loss
    model   = SMILESEncoder(len(vocab),
                            cfg['model']['emb_dim'],
                            cfg['model']['hidden_dim']).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Resume from checkpoint if available
    ckpt_path = os.path.join(cfg['output']['save_dir'],
                              cfg['output']['checkpoint_file'])
    start_epoch = load_checkpoint(model, opt, ckpt_path, device)

    # Training loop
    print("Pretraining...")
    for epoch in range(start_epoch, cfg['training']['epochs'] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb  = xb.to(device), yb.to(device)
            loss    = loss_fn(model(xb).view(-1, len(vocab)), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        print(f"  Epoch {epoch:2d} | loss = {avg_loss:.4f}")

        # Save checkpoint and encoder weights
        save_checkpoint(model, opt, epoch, ckpt_path)
        enc_path = os.path.join(
            cfg['output']['save_dir'],
            cfg['output']['encoder_file'].format(epoch=epoch)
        )
        torch.save(model.state_dict(), enc_path)
        print(f"  Saved epoch {epoch} → {enc_path}")

    print("Pretraining complete.")


if __name__ == '__main__':
    main()
