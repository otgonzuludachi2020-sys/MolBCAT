"""Small JSON checkpoint utilities for MolBCAT experiments.

The training scripts use these helpers to resume at the OUTER-FOLD level.
A completed fold is saved immediately after it finishes.  If a run stops,
completed folds are skipped when --resume is used.
"""

import json
import os
import tempfile
from typing import Any, Dict, Optional


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_checkpoint(path: str, data: Any) -> None:
    """Atomically write one JSON checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=".checkpoint_",
        suffix=".tmp",
        dir=directory,
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_checkpoint(path: str) -> Optional[Any]:
    """Load a JSON checkpoint. Return None if it does not exist."""
    if not os.path.isfile(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_exists(path: str) -> bool:
    return os.path.isfile(path)


def fold_checkpoint_path(output_dir: str, fold_id: int) -> str:
    """Return the standard per-fold checkpoint path."""
    return os.path.join(output_dir, f"fold_{int(fold_id)}.json")
