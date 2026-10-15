"""Fold-safe dataset construction for the Help the Doctor augmentation experiments.

Uses the audit/source IDs from the October 2026 offline augmentation package.
Synthetic examples NEVER participate in cross-validation fold assignment or validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from augmentation_pkg.io_utils import FIELDS, group_key, read_records, sha256_text, stable_row_ids, write_records, write_table

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:
    raise RuntimeError("Install scikit-learn in your active .venv: pip install scikit-learn") from exc

METHODS = ("baseline", "rwd", "llm", "combined")


@dataclass
class Synthetic:
    source_id: str
    source_index: int
    method: str
    row: dict[str, str]


class ProvenanceError(ValueError):
    pass


def rows_fingerprint(rows: list[dict[str, str]]) -> str:
    data = [[r.get(f, "") for f in FIELDS] for r in rows]
    return sha256_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def records_fingerprint(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps([[r.get(f, "") for f in FIELDS] for r in rows],
                                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_audit(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Audit file missing: {path}; keep reports/augmentation/*_final_audit.csv")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"source_row_id", "status", "method", "target_column", "augmented"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ProvenanceError(f"Audit file has missing columns: {path}")
        return list(reader)


def load_originals(path: Path) -> list[dict[str, str]]:
    rows = read_records(path)
    if not rows:
        raise ValueError(f"Empty original dataset: {path}")
    return rows


def load_synthetic(*, base: list[dict[str, str]], path: Path, audit_path: Path,
                   method: str) -> list[Synthetic]:
    """Verify original prefix, accepted audit order, unchanged fields and provenance.

    The previous augmentation exporter writes originals before accepted synthetic rows.
    An accepted audit row is emitted at the time each synthetic row is appended.
    """
    if method not in ("rwd", "llm"):
        raise ValueError(f"Unsupported augmentation method: {method}")
    if not path.is_file():
        raise FileNotFoundError(f"Augmented file missing: {path}")
    all_rows = read_records(path)
    n = len(base)
    if len(all_rows) < n or all_rows[:n] != base:
        raise ProvenanceError(f"{path} does not contain the exact original rows in order. "
                              "Your source CSV may have changed. Do not use this augmented file.")
    extra = all_rows[n:]
    audit = [r for r in read_audit(audit_path) if r["status"] == "accepted"]
    if len(extra) != len(audit):
        raise ProvenanceError(f"{method}: {len(extra)} synthetic rows but {len(audit)} "
                              "accepted audit rows. Regenerate or restore the correct audit.")
    ids = stable_row_ids(base)
    source_by_id = {ident: (idx, row) for idx, (ident, row) in enumerate(zip(ids, base))}
    output: list[Synthetic] = []
    for i, (synthetic, record) in enumerate(zip(extra, audit)):
        sid = record["source_row_id"]
        if sid not in source_by_id:
            raise ProvenanceError(f"{method}: synthetic {i} has an unknown source ID: {sid}")
        src_index, src = source_by_id[sid]
        changed_field = record["target_column"]
        if changed_field not in ("description", "transcription"):
            raise ProvenanceError(f"{method}: unexpected modified field: {changed_field}")
        if record["method"] != method:
            raise ProvenanceError(f"{method}: incorrect audit method for item {i}")
        for field in FIELDS:
            expected = record["augmented"] if field == changed_field else src[field]
            if synthetic[field] != expected:
                raise ProvenanceError(f"{method}: source mapping mismatch at synthetic {i}, "
                                      f"field {field}. Refusing to risk leakage.")
        output.append(Synthetic(sid, src_index, method, synthetic))
    return output


def verify_combined(base: list[dict[str, str]], rwd: list[Synthetic], llm: list[Synthetic],
                    combined_path: Path | None) -> list[Synthetic]:
    seen: set[tuple[str, ...]] = set()
    merged: list[Synthetic] = []
    for syn in [*rwd, *llm]:
        key = tuple(syn.row[x] for x in FIELDS)
        if key not in seen:
            seen.add(key)
            merged.append(syn)
    if combined_path is not None and combined_path.is_file():
        actual = read_records(combined_path)
        expected = [*base, *(syn.row for syn in merged)]
        if actual != expected:
            raise ProvenanceError("Combined CSV differs from reconstructed RWD + LLM rows. "
                                  "Rerun augment_data.py combined with the original train.csv.")
    return merged


def quality_report(rows: list[dict[str, str]], *, expected_classes: int = 12,
                   max_description_chars: int = 2000):
    counts = Counter(row["medical_specialty"] for row in rows)
    # A label that occurs once in an unexpectedly >12-label dataset is suspect,
    # not automatically a new specialty. No medical label is silently corrected.
    if len(counts) > expected_classes:
        labels = {label for label, _ in counts.most_common(expected_classes)}
    else:
        labels = set(counts)
    bad = []
    ids = stable_row_ids(rows)
    for idx, row in enumerate(rows):
        reasons = []
        if row["medical_specialty"] not in labels:
            reasons.append("unexpected_specialty")
        if len(row["medical_specialty"]) > 100:
            reasons.append("label_looks_like_clinical_text")
        if len(row["description"]) > max_description_chars:
            reasons.append("description_suspiciously_long")
        if reasons:
            bad.append({"index_0_based": idx, "row_number_1_based_including_header": idx + 2,
                        "source_row_id": ids[idx], "specialty": row["medical_specialty"],
                        "description_chars": len(row["description"]),
                        "reasons": ";".join(reasons),
                        "description_preview": row["description"][:220]})
    return labels, bad


def make_folds(rows: list[dict[str, str]], eligible: list[int], n_splits: int, seed: int):
    if n_splits < 2:
        raise ValueError("--folds must be at least 2")
    if len(eligible) < n_splits:
        raise ValueError("Not enough original records for requested folds")
    labels = [rows[i]["medical_specialty"] for i in eligible]
    groups = [group_key(rows[i]) for i in eligible]
    if len(set(groups)) < n_splits:
        raise ValueError("Not enough distinct clinical-note groups for requested folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    observed_validation = []
    for train_positions, valid_positions in splitter.split(eligible, labels, groups):
        train_ids = [eligible[int(i)] for i in train_positions]
        valid_ids = [eligible[int(i)] for i in valid_positions]
        train_groups = {group_key(rows[i]) for i in train_ids}
        valid_groups = {group_key(rows[i]) for i in valid_ids}
        if train_groups & valid_groups:
            raise AssertionError("Original note group appears in both train and validation")
        if set(train_ids) & set(valid_ids):
            raise AssertionError("Original row index in both train and validation")
        observed_validation.extend(valid_ids)
        folds.append((train_ids, valid_ids))
    if sorted(observed_validation) != sorted(eligible):
        raise AssertionError("Every eligible original must be validated exactly once")
    return folds


def materialize_folds(*, original: list[dict[str, str]], synthetic: dict[str, list[Synthetic]],
                      folds: list[tuple[list[int], list[int]]], methods: list[str],
                      dest: Path, source: Path, quarantine: list[dict]):
    """Create fold-specific training CSVs and untouched validation CSVs."""
    dest.mkdir(parents=True, exist_ok=True)
    write_table(dest / "data_quality_quarantine.csv", [
        "index_0_based", "row_number_1_based_including_header", "source_row_id", "specialty",
        "description_chars", "reasons", "description_preview"], quarantine)
    stable_ids = stable_row_ids(original)
    manifests = []
    for fold_idx, (train_indices, val_indices) in enumerate(folds, 1):
        sub = dest / f"fold_{fold_idx:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        train_set = set(train_indices)
        valid_set = set(val_indices)
        val_rows = [original[i] for i in val_indices]
        orig_train_rows = [original[i] for i in train_indices]
        write_records(sub / "validation.csv", val_rows)
        write_table(sub / "validation_provenance.csv", ["source_row_id", "original_row_index", "specialty"],
                    [{"source_row_id": stable_ids[i], "original_row_index": i,
                      "specialty": original[i]["medical_specialty"]} for i in val_indices])
        stats = {"fold": fold_idx, "train_originals": len(train_indices),
                 "validation_originals": len(val_indices),
                 "validation_labels": dict(Counter(original[i]["medical_specialty"] for i in val_indices))}
        for method in methods:
            if method == "baseline":
                additives: list[Synthetic] = []
            else:
                additives = [s for s in synthetic[method] if s.source_index in train_set]
            if any(s.source_index in valid_set for s in additives):
                raise AssertionError("Synthetic example from held-out original in train")
            rows = orig_train_rows + [s.row for s in additives]
            write_records(sub / f"train_{method}.csv", rows)
            stats[method] = {"train_total": len(rows), "train_synthetic": len(additives)}
        manifests.append(stats)
        (sub / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {"source": str(source.resolve()), "source_fingerprint": records_fingerprint(original),
                "folds": len(folds), "quarantine_rows": len(quarantine),
                "methods": methods, "fold_stats": manifests}
    (dest / "cv_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifests


def clean_text(row: dict[str, str], fields: list[str], strip_keyword_prefix: bool):
    import re
    parts = []
    for name in fields:
        value = row.get(name, "")
        if name == "keywords" and strip_keyword_prefix:
            value = re.sub(r"^\s*(?:keywords?|key\s*words?)\s*[:\-]\s*", "", value, flags=re.IGNORECASE)
        parts.append(f"[{name}] {value}" if value.strip() else "")
    return " ".join(part for part in parts if part).strip() or "[empty]"
