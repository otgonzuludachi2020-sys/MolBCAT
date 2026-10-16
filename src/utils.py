"""
Utility functions for MolBCAT.

Provides:
    - Random seed control
    - JSON loading/saving
    - Vocabulary loading
    - Pretrained GRU encoder loading
    - Device selection

Fold-level / experiment-level checkpoint and resume logic
is implemented separately in src/checkpoint.py.
"""

import os
import json
import random

import numpy as np
import torch




def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA/CuDNN behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str) -> dict:
    """
    Load a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed dictionary.

        Returns an empty dictionary if the file does not exist.
    """

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        path: Output JSON path.
        data: Dictionary to save.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )



def load_vocab(vocab_path: str) -> dict:
    """
    Load vocabulary from JSON.

    Args:
        vocab_path:
            Path to vocab.json.

    Returns:
        Vocabulary dictionary.

    Raises:
        FileNotFoundError:
            If the vocabulary file does not exist.
    """

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"vocab.json not found at: {vocab_path}\n"
            "Please run pretrain first or download the pretrained "
            "weights and vocabulary."
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pretrained_encoder(
    model,
    pretrain_path: str,
    device: str,
) -> None:
    """
    Load pretrained GRU encoder weights into a model.

    The checkpoint is expected to contain:
        emb.weight
        gru.<parameter_name>

    Args:
        model:
            Model containing `emb` and `gru`.

        pretrain_path:
            Path to pretrained encoder checkpoint.

        device:
            Device used for loading the checkpoint.
    """

    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            f"Pretrained encoder not found at: {pretrain_path}\n"
            "Please run pretrain first or download the pretrained "
            "weights."
        )

    state = torch.load(
        pretrain_path,
        map_location=device,
        weights_only=True,
    )

    # Embedding weights.
    if "emb.weight" in state:
        model.emb.weight.data.copy_(
            state["emb.weight"].to(
                device=model.emb.weight.device,
                dtype=model.emb.weight.dtype,
            )
        )
    else:
        raise KeyError(
            "Pretrained checkpoint does not contain "
            "'emb.weight'."
        )

    # GRU weights.
    missing_gru_keys = []

    for name, param in model.gru.named_parameters():
        key = f"gru.{name}"

        if key in state:
            param.data.copy_(
                state[key].to(
                    device=param.device,
                    dtype=param.dtype,
                )
            )
        else:
            missing_gru_keys.append(key)

    if missing_gru_keys:
        print(
            "Warning: missing GRU parameters in pretrained "
            "checkpoint:"
        )

        for key in missing_gru_keys:
            print(f"  - {key}")

    print(
        f"Pretrained weights loaded from: {pretrain_path}"
    )


def get_device() -> str:
    """
    Return the available PyTorch device.

    Returns:
        'cuda' if CUDA is available, otherwise 'cpu'.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    return device