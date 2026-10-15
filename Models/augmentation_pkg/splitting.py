"""Deterministic group-aware holdout split and minority-focused sampling."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

from .io_utils import group_key


def grouped_stratified_split(rows: list[dict[str, str]], *, fraction: float = 0.2,
                             seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Best-effort class-stratified split without crossing exact normalized note groups.

    A class supported by only one distinct note-group remains in training. This avoids
    train/validation leakage but can make the validation subset lack rare classes.
    """
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_key(row)].append(index)
    groups = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    label_counts = Counter(row["medical_specialty"] for row in rows)
    group_counts = Counter()
    for group in groups:
        for label in set(rows[i]["medical_specialty"] for i in group):
            group_counts[label] += 1
    targets = {label: max(1, round(count * fraction)) if group_counts[label] > 1 else 0
               for label, count in label_counts.items()}
    global_target = max(1, round(len(rows) * fraction)) if len(groups) > 1 else 0
    validation_indices: set[int] = set()
    validation_counts: Counter[str] = Counter()
    selected: set[int] = set()
    tie_breakers = {j: rng.random() for j in range(len(groups))}

    while len(validation_indices) < global_target:
        choices = []
        for j, group in enumerate(groups):
            if j in selected:
                continue
            group_labels = Counter(rows[i]["medical_specialty"] for i in group)
            # Reserve at least one entire ORIGINAL group for each specialty in training.
            if any(group_counts[c] <= 1 for c in group_labels):
                continue
            remaining_need = sum(max(0, targets[c] - validation_counts[c]) / max(1, label_counts[c])
                                 for c in group_labels for _ in range(group_labels[c]))
            if remaining_need <= 0:
                continue
            overshoot = max(0, len(validation_indices) + len(group) - global_target)
            score = (remaining_need - (overshoot / max(1, len(rows))) * 0.2,
                     -abs(len(validation_indices) + len(group) - global_target), tie_breakers[j])
            choices.append((score, j, group_labels))
        if not choices:
            break
        _, chosen, group_labels = max(choices)
        selected.add(chosen)
        validation_indices.update(groups[chosen])
        validation_counts.update(group_labels)
        for label in group_labels:
            group_counts[label] -= 1

    train = [dict(row) for i, row in enumerate(rows) if i not in validation_indices]
    val = [dict(row) for i, row in enumerate(rows) if i in validation_indices]
    train_keys = {group_key(row) for row in train}
    val_keys = {group_key(row) for row in val}
    if not train_keys.isdisjoint(val_keys):
        raise AssertionError("Data leakage detected in group-aware split")
    warnings = []
    if not val:
        warnings.append("No validation group could be assigned without losing all training examples of a specialty")
    for label in sorted(label_counts):
        if label not in validation_counts:
            warnings.append(f"Specialty {label!r} has no validation examples; consider cross-validation")
        if group_counts[label] < 1:
            raise AssertionError("Specialty missing entirely from training")
    return train, val, warnings


def _quantile(numbers: list[int], q: float) -> float:
    values = sorted(numbers)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if lo != hi else values[lo]


def choose_source_indices(rows: list[dict[str, str]], selection_cfg: dict[str, Any],
                          *, seed: int = 42, override_limit: int | None = None, text_field: str = "description") -> list[int]:
    """Select the same source examples for RWD and LLM at the same settings."""
    if not rows:
        return []
    strategy = selection_cfg.get("strategy", "minority")
    if strategy not in ("minority", "all"):
        raise ValueError("selection.strategy must be 'minority' or 'all'")
    limit = selection_cfg.get("max_source_rows", 200) if override_limit is None else override_limit
    limit = int(limit)
    if limit < 0:
        raise ValueError("max_source_rows/--limit must be >= 0 (0 means unlimited)")
    pools: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if row.get(text_field, "").strip():
            pools[row["medical_specialty"]].append(idx)
    all_counts = Counter(row["medical_specialty"] for row in rows)
    target = int(math.ceil(_quantile(list(all_counts.values()), float(selection_cfg.get("target_quantile", .75)))))
    rng = random.Random(seed)
    quotas: dict[str, int] = {}
    for label, indices in pools.items():
        rng.shuffle(indices)
        quotas[label] = (len(indices) if strategy == "all"
                         else min(len(indices), max(0, target - all_counts[label])))
    selected: list[int] = []
    picked = Counter()
    # Round-robin among small classes. No source row selected more than once.
    while any(picked[cls] < quota for cls, quota in quotas.items()):
        made_progress = False
        for label in sorted(quotas, key=lambda c: (all_counts[c], c)):
            if picked[label] >= quotas[label]:
                continue
            selected.append(pools[label][picked[label]])
            picked[label] += 1
            made_progress = True
            if limit and len(selected) >= limit:
                return selected
        if not made_progress:
            break
    return selected
