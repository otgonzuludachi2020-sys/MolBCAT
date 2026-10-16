"""
Scaffold-disjoint dataset splitting for MolBCAT.

Provides
--------
1. Scaffold-based K-fold cross-validation.
2. Scaffold-disjoint train/validation splitting.
3. Class-aware assignment for binary classification datasets.
4. Scaffold overlap checking.

"""
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


def _scaffold_from_smiles(smi, index):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES at index {index}: {smi}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def get_scaffold_groups(smiles):
  
    scaffolds = defaultdict(list)

    for i, smi in enumerate(smiles):
        scaffold = _scaffold_from_smiles(smi, i)
        scaffolds[scaffold].append(i)

    return list(scaffolds.values())


def get_scaffolds(smiles, indices):
   
    scaffolds = set()

    for i in indices:
        scaffolds.add(_scaffold_from_smiles(smiles[i], i))

    return scaffolds


def _group_class_counts(group, labels):
    pos = sum(int(labels[i]) == 1 for i in group)
    neg = len(group) - pos
    return pos, neg


def _validate_binary_labels(labels, indices=None):
    if indices is None:
        values = labels
    else:
        values = [labels[i] for i in indices]

    unique = set(int(x) for x in values)
    if not unique.issubset({0, 1}):
        raise ValueError("Class-aware splitting supports binary labels {0, 1}.")


def scaffold_kfold_split(
    smiles: Sequence[str],
    labels: Sequence[int],
    n_folds: int = 5,
    seed: int = 42,
) -> List[np.ndarray]:
    
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    smiles = list(smiles)
    labels = np.asarray(labels)

    if len(smiles) != len(labels):
        raise ValueError(
            f"smiles and labels length mismatch: "
            f"{len(smiles)} vs {len(labels)}"
        )

    if len(smiles) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} folds from {len(smiles)} samples."
        )

    _validate_binary_labels(labels)


    scaffold_groups = get_scaffold_groups(smiles)

    if len(scaffold_groups) < n_folds:
        raise ValueError(
            f"Only {len(scaffold_groups)} unique scaffolds are available "
            f"for {n_folds} folds."
        )

    group_ids = np.empty(len(smiles), dtype=int)

    for group_id, indices in enumerate(scaffold_groups):
        group_ids[indices] = group_id

   
    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    dummy_x = np.zeros(len(smiles))

    folds = []

    for fold_id, (_, test_idx) in enumerate(
        splitter.split(dummy_x, labels, groups=group_ids)
    ):
        test_idx = np.asarray(test_idx, dtype=int)
        folds.append(test_idx)

    all_indices = np.concatenate(folds)

    if len(all_indices) != len(smiles):
        raise ValueError(
            f"Fold assignment lost samples: "
            f"{len(all_indices)} vs {len(smiles)}."
        )

    if len(np.unique(all_indices)) != len(smiles):
        raise ValueError(
            "Duplicate sample indices detected across folds."
        )

    if set(all_indices.tolist()) != set(range(len(smiles))):
        raise ValueError(
            "Some sample indices are missing from the folds."
        )

 
    fold_scaffolds = []

    for fold_id, idx in enumerate(folds):
        scaffolds = set(group_ids[idx].tolist())
        fold_scaffolds.append(scaffolds)

        if len(idx) == 0:
            raise ValueError(f"Fold {fold_id} is empty.")

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            overlap = fold_scaffolds[i] & fold_scaffolds[j]

            if overlap:
                raise ValueError(
                    f"Scaffold leakage between folds "
                    f"{i} and {j}: {len(overlap)} shared groups."
                )


    for fold_id, idx in enumerate(folds):
        classes = set(labels[idx].tolist())

        if len(classes) < 2:
            raise ValueError(
                f"Fold {fold_id} contains only class "
                f"{next(iter(classes), 'none')}."
            )


    total = len(smiles)

    print("\nScaffold-Stratified K-Fold:")
    print(f"  Samples: {total}")
    print(f"  Folds:   {n_folds}")

    for fold_id, idx in enumerate(folds):
        y = labels[idx]
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        ratio = pos / len(idx)

        print(
            f"  Fold {fold_id}: "
            f"n={len(idx)}, "
            f"positive={pos}, "
            f"negative={neg}, "
            f"positive_ratio={ratio:.4f}"
        )

    print()

    return folds

def scaffold_train_val_split(
    smiles,
    indices,
    labels=None,
    val_ratio=0.10,
    seed=42,
):
    
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices = list(indices)

    if len(indices) < 2:
        raise ValueError("Development set is too small.")

    if labels is not None:
        _validate_binary_labels(labels, indices)

    # ---------------------------------------------------------
    # Build scaffold groups
    # ---------------------------------------------------------
    groups_dict = defaultdict(list)

    for i in indices:
        scaffold = _scaffold_from_smiles(smiles[i], i)
        groups_dict[scaffold].append(i)

    groups = list(groups_dict.values())

    if len(groups) < 2:
        raise ValueError(
            "At least two scaffold groups are required."
        )

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------
    total_size = len(indices)
    target_size = total_size * val_ratio

    if labels is not None:
        total_pos = sum(
            int(labels[i]) == 1
            for i in indices
        )
        total_neg = total_size - total_pos

        target_pos = total_pos * val_ratio
        target_neg = total_neg * val_ratio

 
    def candidate_score(val_groups):

        val_idx = [
            i
            for group in val_groups
            for i in group
        ]

        val_size = len(val_idx)

        size_error = (
            (val_size - target_size)
            / max(target_size, 1.0)
        ) ** 2

        if labels is None:
            return size_error

        val_pos = sum(
            int(labels[i]) == 1
            for i in val_idx
        )
        val_neg = val_size - val_pos

        pos_error = (
            (val_pos - target_pos)
            / max(target_pos, 1.0)
        ) ** 2

        neg_error = (
            (val_neg - target_neg)
            / max(target_neg, 1.0)
        ) ** 2

        # Strong emphasis on class balance.
        return (
            5.0 * size_error
            + 10.0 * pos_error
            + 10.0 * neg_error
        )


    rng = random.Random(seed)

    n_candidates = 300

    best_score = float("inf")
    best_val_groups = None

    for candidate in range(n_candidates):

        shuffled_groups = groups.copy()
        rng.shuffle(shuffled_groups)

        val_groups = []
        val_size = 0

        # Randomized greedy construction.
        remaining = shuffled_groups.copy()

        while remaining:

            # Stop once validation is sufficiently close to target.
            if val_size >= target_size:
                break

            # Evaluate adding each remaining scaffold.
            options = []

            for group in remaining:

                new_size = val_size + len(group)

                # Do not allow validation to become unnecessarily large.
                size_penalty = abs(
                    new_size - target_size
                )

                if labels is not None:

                    current_idx = [
                        i
                        for g in val_groups
                        for i in g
                    ]

                    current_pos = sum(
                        int(labels[i]) == 1
                        for i in current_idx
                    )

                    current_neg = (
                        len(current_idx) - current_pos
                    )

                    new_pos = (
                        current_pos
                        + sum(
                            int(labels[i]) == 1
                            for i in group
                        )
                    )

                    new_neg = (
                        current_neg
                        + sum(
                            int(labels[i]) == 0
                            for i in group
                        )
                    )

                    class_penalty = (
                        abs(new_pos - target_pos)
                        / max(target_pos, 1.0)
                        +
                        abs(new_neg - target_neg)
                        / max(target_neg, 1.0)
                    )

                    score = (
                        size_penalty
                        + 50.0 * class_penalty
                    )

                else:
                    score = size_penalty

                options.append(
                    (score, group)
                )

            options.sort(
                key=lambda x: x[0]
            )

            # Randomize among the best few options.
            top_k = min(5, len(options))
            _, chosen_group = rng.choice(
                options[:top_k]
            )

            val_groups.append(chosen_group)
            val_size += len(chosen_group)
            remaining.remove(chosen_group)

        val_idx = [
            i
            for group in val_groups
            for i in group
        ]

        train_idx_candidate = [
            i
            for i in indices
            if i not in set(val_idx)
        ]

        if not train_idx_candidate or not val_idx:
            continue

        if labels is not None:

            train_classes = {
                int(labels[i])
                for i in train_idx_candidate
            }

            val_classes = {
                int(labels[i])
                for i in val_idx
            }

            if train_classes != {0, 1}:
                continue

            if val_classes != {0, 1}:
                continue

        score = candidate_score(val_groups)

        if score < best_score:
            best_score = score
            best_val_groups = val_groups

   
    if best_val_groups is None:
        raise ValueError(
            "Could not create a valid scaffold-disjoint "
            "train/validation split containing both classes."
        )


    val_idx = [
        i
        for group in best_val_groups
        for i in group
    ]

    val_set = set(val_idx)

    train_idx = [
        i
        for i in indices
        if i not in val_set
    ]

 
    if set(train_idx) & set(val_idx):
        raise RuntimeError(
            "Train/validation molecule overlap detected."
        )

    
    train_scaffolds = get_scaffolds(
        smiles,
        train_idx,
    )

    val_scaffolds = get_scaffolds(
        smiles,
        val_idx,
    )

    overlap = train_scaffolds & val_scaffolds

    if overlap:
        raise RuntimeError(
            f"Train/validation scaffold overlap detected: "
            f"{len(overlap)} shared scaffolds."
        )

    if labels is not None:

        train_classes = {
            int(labels[i])
            for i in train_idx
        }

        val_classes = {
            int(labels[i])
            for i in val_idx
        }

        if train_classes != {0, 1}:
            raise ValueError(
                "Training split does not contain both classes."
            )

        if val_classes != {0, 1}:
            raise ValueError(
                "Validation split does not contain both classes."
            )

 
    if len(train_idx) + len(val_idx) != len(indices):
        raise RuntimeError(
            "Train/validation split does not contain all samples."
        )


    actual_ratio = len(val_idx) / len(indices)

    print(
        f"  Inner split: "
        f"train={len(train_idx)}, "
        f"validation={len(val_idx)}, "
        f"validation_ratio={actual_ratio:.4f}"
    )

    if labels is not None:

        train_pos = sum(
            int(labels[i]) == 1
            for i in train_idx
        )

        val_pos = sum(
            int(labels[i]) == 1
            for i in val_idx
        )

        print(
            f"  Train: "
            f"negative={len(train_idx) - train_pos}, "
            f"positive={train_pos}"
        )

        print(
            f"  Validation: "
            f"negative={len(val_idx) - val_pos}, "
            f"positive={val_pos}"
        )

    print(
        f"  Scaffold overlap: {len(overlap)}"
    )

    return train_idx, val_idx
 
def check_scaffold_disjoint(smiles, train_idx, val_idx, test_idx):
    """Return scaffold-overlap counts for train/validation/test."""
    train_scaffolds = get_scaffolds(smiles, train_idx)
    val_scaffolds = get_scaffolds(smiles, val_idx)
    test_scaffolds = get_scaffolds(smiles, test_idx)

    return {
        "train_val_overlap": len(train_scaffolds & val_scaffolds),
        "train_test_overlap": len(train_scaffolds & test_scaffolds),
        "val_test_overlap": len(val_scaffolds & test_scaffolds),
    }

def scaffold_kfold_split_regression(
    smiles,
    labels=None,
    n_folds=5,
    seed=42
):
    """
    Scaffold-disjoint K-fold split for regression.

    Keeps Bemis-Murcko scaffolds intact and uses a
    greedy local-search strategy to balance fold sizes.
    """

    import random

    smiles = list(smiles)
    n = len(smiles)

    # -----------------------------
    # Build scaffold groups
    # -----------------------------
    scaffold_groups = {}

    for i, smi in enumerate(smiles):
        scaffold = _scaffold_from_smiles(smi, i)

        if scaffold not in scaffold_groups:
            scaffold_groups[scaffold] = []

        scaffold_groups[scaffold].append(i)

    groups = list(scaffold_groups.values())

    # Largest groups first
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    # -----------------------------
    # Initial balanced assignment
    # -----------------------------
    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for group in groups:
        k = min(
            range(n_folds),
            key=lambda x: fold_sizes[x]
        )

        folds[k].extend(group)
        fold_sizes[k] += len(group)

    # -----------------------------
    # Local improvement
    # -----------------------------
    # Move a whole scaffold from the
    # largest fold to the smallest fold
    # when this reduces size imbalance.
    # -----------------------------

    def imbalance(sizes):
        return max(sizes) - min(sizes)

    improved = True

    while improved:

        improved = False

        largest = max(
            range(n_folds),
            key=lambda k: fold_sizes[k]
        )

        smallest = min(
            range(n_folds),
            key=lambda k: fold_sizes[k]
        )

        current_diff = imbalance(fold_sizes)

        best_group = None
        best_new_sizes = None
        best_diff = current_diff

        # Candidate scaffold groups in largest fold
        largest_groups = {}

        for idx in folds[largest]:
            scaffold = _scaffold_from_smiles(
                smiles[idx],
                idx
            )

            largest_groups.setdefault(
                scaffold,
                []
            ).append(idx)

        for scaffold, group in largest_groups.items():

            group_size = len(group)

            new_sizes = fold_sizes.copy()

            new_sizes[largest] -= group_size
            new_sizes[smallest] += group_size

            new_diff = imbalance(new_sizes)

            if new_diff < best_diff:

                best_diff = new_diff
                best_group = group
                best_new_sizes = new_sizes

        if best_group is not None:

            group_set = set(best_group)

            folds[largest] = [
                i
                for i in folds[largest]
                if i not in group_set
            ]

            folds[smallest].extend(best_group)

            fold_sizes = best_new_sizes
            improved = True

    # -----------------------------
    # Sort indices
    # -----------------------------
    folds = [
        sorted(fold)
        for fold in folds
    ]

    # -----------------------------
    # Integrity check
    # -----------------------------
    all_indices = sorted(
        i
        for fold in folds
        for i in fold
    )

    assert all_indices == list(range(n)), \
        "Some samples are missing or duplicated."

    # -----------------------------
    # Scaffold-disjoint check
    # -----------------------------
    fold_scaffolds = []

    for fold in folds:

        fold_scaffolds.append({
            _scaffold_from_smiles(smiles[i], i)
            for i in fold
        })

    for i in range(n_folds):
        for j in range(i + 1, n_folds):

            overlap = (
                fold_scaffolds[i]
                & fold_scaffolds[j]
            )

            assert len(overlap) == 0, (
                f"Scaffold overlap between "
                f"fold {i} and fold {j}: "
                f"{len(overlap)}"
            )

    # -----------------------------
    # Print results
    # -----------------------------
    print("\nScaffold K-Fold Regression:")
    print(f"  Samples: {n}")
    print(f"  Folds:   {n_folds}")

    for k, fold in enumerate(folds):
        print(
            f"  Fold {k}: n={len(fold)}"
        )

    print(
        f"  Fold-size range: "
        f"{min(fold_sizes)} - {max(fold_sizes)}"
    )

    return folds

def scaffold_train_val_split_regression(
    smiles,
    indices,
    val_ratio=0.10,
    seed=42
):
    """
    Scaffold-disjoint train/validation split for regression.

    Selects scaffold groups while targeting the requested
    validation sample ratio.
    """

    import random

    indices = list(indices)

    # -----------------------------
    # Build scaffold groups
    # -----------------------------
    scaffold_groups = {}

    for i in indices:

        scaffold = _scaffold_from_smiles(
            smiles[i],
            i
        )

        if scaffold not in scaffold_groups:
            scaffold_groups[scaffold] = []

        scaffold_groups[scaffold].append(i)

    groups = list(
        scaffold_groups.values()
    )

    rng = random.Random(seed)
    rng.shuffle(groups)

    # Large groups first
    groups.sort(
        key=len,
        reverse=True
    )

    target_val = len(indices) * val_ratio

    val_groups = []
    val_count = 0

    # -----------------------------
    # Select validation scaffolds
    # -----------------------------
    for group in groups:

        current_diff = abs(
            val_count - target_val
        )

        new_diff = abs(
            (val_count + len(group))
            - target_val
        )

        # Add group only if it moves
        # closer to the target
        if new_diff < current_diff:

            val_groups.append(group)
            val_count += len(group)

    val_idx = sorted(
        i
        for group in val_groups
        for i in group
    )

    val_set = set(val_idx)

    train_idx = sorted(
        i
        for i in indices
        if i not in val_set
    )

    # -----------------------------
    # Integrity checks
    # -----------------------------
    assert len(
        set(train_idx) & set(val_idx)
    ) == 0

    assert sorted(
        train_idx + val_idx
    ) == sorted(indices)

    train_scaffolds = {
        _scaffold_from_smiles(smiles[i], i)
        for i in train_idx
    }

    val_scaffolds = {
        _scaffold_from_smiles(smiles[i], i)
        for i in val_idx
    }

    overlap = (
        train_scaffolds
        & val_scaffolds
    )

    assert len(overlap) == 0, (
        f"Train/validation scaffold overlap: "
        f"{len(overlap)}"
    )

    actual_ratio = (
        len(val_idx) / len(indices)
    )

    print(
        f"  Inner split: "
        f"train={len(train_idx)}, "
        f"validation={len(val_idx)}, "
        f"validation_ratio={actual_ratio:.4f}"
    )

    print(
        f"  Scaffold overlap: {len(overlap)}"
    )

    return train_idx, val_idx