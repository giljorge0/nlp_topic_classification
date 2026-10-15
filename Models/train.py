"""Five-fold grouped CV, final fitting, and inference for four PyTorch RNN classifiers.

Use `python train.py --help` and README.md for commands.
The official unlabelled test set is only read by the `predict` subcommand.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from birnn import BiRNN
from birnn_attention import BiRNNAttention
from bilstm import BiLSTM
from bilstm_attention import BiLSTMAttention

LABELS = (
    "Cardiovascular-Pulmonary", "Dermatology", "Gastroenterology",
    "General Medicine", "Neurology", "Neurosurgery", "Obstetrics-Gynecology",
    "Ophthalmology", "Orthopedic", "Psychiatry-Psychology", "Radiology", "Surgery",
)
FIELDS = ("description", "sample_name", "transcription", "keywords")
MODELS = {
    "birnn": BiRNN,
    "birnn_attention": BiRNNAttention,
    "bilstm": BiLSTM,
    "bilstm_attention": BiLSTMAttention,
}
TOKEN_REGEX = re.compile(r"\b\w+(?:[-']\w+)*\b", re.UNICODE)
LABEL_PREFIXES = {re.sub(r"[^a-z0-9]", "", label.casefold()) for label in LABELS}
FOLDS = 5


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            name = "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise ValueError("Intel XPU is not available in this PyTorch installation")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is not available")
    return device


def read_labelled_csv(path: str) -> tuple[list[dict], int]:
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames or not {"medical_specialty", *FIELDS}.issubset(reader.fieldnames):
            raise ValueError("Training CSV must have a semicolon-separated five-column header")
        rows, skipped = [], 0
        for line, raw in enumerate(reader, start=2):
            label = (raw.get("medical_specialty") or "").strip()
            if label not in LABELS:
                skipped += 1
                continue
            row = {field: (raw.get(field) or "").strip() for field in FIELDS}
            row.update(medical_specialty=label, row_number=line)
            rows.append(row)
    if not rows:
        raise ValueError("No valid labelled rows")
    return rows, skipped


def read_unlabelled_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row_number, cells in enumerate(csv.reader(stream, delimiter=";"), start=1):
            if len(cells) != 4:
                raise ValueError(f"Test row {row_number}: expected four fields without a header")
            item = dict(zip(FIELDS, (x.strip() for x in cells)))
            item["row_number"] = row_number
            rows.append(item)
    if not rows:
        raise ValueError("Empty test CSV")
    return rows


def configuration_name(fields: list[str], strip: bool, keep: bool) -> str:
    if fields == list(FIELDS):
        base = "all_four_fields"
    elif len(fields) == 1:
        base = fields[0] + "_only"
    else:
        base = "__".join(fields)
    suffix = "__prefix_removed" if strip else ("__prefix_only" if keep else "")
    return base + suffix


def make_text(row: dict, fields: list[str], strip: bool = False, keep: bool = False) -> str:
    if strip and keep:
        raise ValueError("Keyword modes are mutually exclusive")
    parts = []
    for field in fields:
        content = (row.get(field) or "").strip()
        if field == "keywords" and content:
            first, separator, rest = content.partition(",")
            # Never consult the true row label, including for prefix recognition.
            recognized = bool(separator) and re.sub(r"[^a-z0-9]", "", first.casefold()) in LABEL_PREFIXES
            if recognized and strip:
                content = rest.strip()
            elif recognized and keep:
                content = first.strip()
        if content:
            parts.append(f"{field.replace('_', ' ')}: {content}")
    return " ".join(parts)


def tokenize(text: str) -> list[str]:
    return TOKEN_REGEX.findall(text.casefold())


def build_vocab(texts: list[str], max_vocab: int, min_freq: int, max_length: int) -> dict[str, int]:
    counter = Counter(token for text in texts for token in tokenize(text)[:max_length])
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, frequency in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if frequency < min_freq or len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


class ClinicalDataset(Dataset):
    def __init__(self, rows: list[dict], vocab: dict[str, int], args, labelled: bool = True):
        self.rows = rows
        self.tokens = []
        for row in rows:
            text = make_text(row, args.fields, args.strip_keyword_prefix, args.keep_keyword_prefix_only)
            values = [vocab.get(token, 1) for token in tokenize(text)[:args.max_length]]
            self.tokens.append(torch.tensor(values or [1], dtype=torch.long))
        self.labels = ([LABELS.index(row["medical_specialty"]) for row in rows]
                       if labelled else [-1] * len(rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.tokens[index], self.labels[index], index


def collate(batch):
    values, targets, positions = zip(*batch)
    padded = pad_sequence(values, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(values_) for values_ in values], dtype=torch.long)
    return padded, lengths, torch.tensor(targets, dtype=torch.long), positions


def loader(rows: list[dict], vocab: dict[str, int], args, shuffle: bool, labelled: bool = True):
    return DataLoader(ClinicalDataset(rows, vocab, args, labelled), batch_size=args.batch_size,
                      shuffle=shuffle, collate_fn=collate, num_workers=0)


def group_key(row: dict) -> str:
    # Exact transcript grouping, case-insensitive. Missing notes grouped by the remaining fields.
    basis = row["transcription"].strip().casefold()
    if not basis:
        basis = "||".join(row[field] for field in FIELDS).casefold()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def data_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row["row_number"], row["medical_specialty"],
                                  *[row[f] for f in FIELDS]], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_partition(train_indices: list[int], val_indices: list[int], keys: list[str]) -> None:
    assert not set(train_indices).intersection(val_indices), "Row leakage between partitions"
    assert not {keys[i] for i in train_indices}.intersection(keys[i] for i in val_indices), "Group leakage between partitions"


def create_or_load_folds(rows: list[dict], runs_dir: Path, seed: int) -> dict:
    """Freeze outer/inner folds once and share them across ALL architectures/field configurations."""
    path = runs_dir / "cv_splits.json"
    digest = data_digest(rows)
    if path.exists():
        manifest = read_json(path)
        if (manifest.get("fingerprint") != digest or manifest.get("seed") != seed
                or manifest.get("folds") != FOLDS or manifest.get("version") != 1):
            raise ValueError(f"Existing {path} belongs to different data/seed. Use a different --runs-dir.")
        return manifest
    y = np.array([LABELS.index(r["medical_specialty"]) for r in rows])
    keys = [group_key(r) for r in rows]
    indices = np.arange(len(rows))
    outer = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    records = []
    for fold, (outer_train, outer_test) in enumerate(outer.split(indices, y, keys), start=1):
        tr, te = outer_train.tolist(), outer_test.tolist()
        check_partition(tr, te, keys)
        inner = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed + fold)
        # The inner training pool is the 80% outer training pool, not all examples.
        candidates = []
        desired = np.bincount(y[tr], minlength=len(LABELS)) / len(tr)
        for inner_train_local, inner_val_local in inner.split(outer_train, y[tr], np.array(keys)[tr]):
            fit = outer_train[inner_train_local].tolist()
            val = outer_train[inner_val_local].tolist()
            val_counts = np.bincount(y[val], minlength=len(LABELS))
            fit_counts = np.bincount(y[fit], minlength=len(LABELS))
            missing = int((val_counts == 0).sum() + (fit_counts == 0).sum())
            distribution_gap = float(np.abs(val_counts / len(val) - desired).sum())
            size_gap = abs(len(val) / len(tr) - 1 / FOLDS)
            candidates.append((missing, distribution_gap + size_gap, fit, val))
        _, _, fit, val = min(candidates, key=lambda x: (x[0], x[1]))
        check_partition(fit, val, keys)
        check_partition(fit, te, keys)
        check_partition(val, te, keys)
        records.append({
            "fold": fold,
            "fit": fit,
            "inner_validation": val,
            "outer_test": te,
            "sizes": {"fit": len(fit), "inner_validation": len(val), "outer_test": len(te)},
            "outer_class_counts": dict(zip(LABELS, map(int, np.bincount(y[te], minlength=len(LABELS))))),
        })
    outer_tests = [i for record in records for i in record["outer_test"]]
    if len(outer_tests) != len(rows) or set(outer_tests) != set(indices.tolist()):
        raise RuntimeError("Outer folds must cover each example exactly once")
    manifest = {
        "version": 1, "seed": seed, "folds": FOLDS, "fingerprint": digest,
        "valid_rows": len(rows), "grouping": "exact casefold transcript; fallback to all four fields",
        "method": "StratifiedGroupKFold outer 5; separate inner grouped 5-way selection",
        "records": records,
    }
    write_json(path, manifest)
    print(f"Created frozen folds: {path}")
    return manifest


def model_config(args, vocab_size: int):
    return dict(vocab_size=vocab_size, embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim, num_classes=len(LABELS), num_layers=args.num_layers,
                dropout=args.dropout, padding_idx=0)


def loss_function(args, rows: list[dict], device: torch.device):
    if args.class_weighting == "none":
        return nn.CrossEntropyLoss()
    counts = np.bincount([LABELS.index(r["medical_specialty"]) for r in rows], minlength=len(LABELS))
    weights = np.sqrt(counts.max() / np.maximum(counts, 1))
    weights /= weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device, dtype=torch.float32))


def train_epoch(model, batches, optimizer, loss_fn, device, clip: float):
    model.train()
    total, items = 0.0, 0
    for padded, lengths, target, _ in batches:
        padded, target = padded.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(padded, lengths)
        loss = loss_fn(logits, target)
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item() * len(target)
        items += len(target)
    return total / items


@torch.no_grad()
def infer(model, batches, device, loss_fn=None):
    model.eval()
    predictions, targets, indices = [], [], []
    total_loss, item_count = 0., 0
    for padded, lengths, expected, positions in batches:
        logits = model(padded.to(device), lengths)
        predictions.extend(logits.argmax(1).cpu().tolist())
        targets.extend(expected.tolist())
        indices.extend(positions)
        if loss_fn is not None:
            total_loss += loss_fn(logits, expected.to(device)).item() * len(expected)
            item_count += len(expected)
    return predictions, targets, indices, total_loss / item_count if item_count else None


def score(y_true: list[int], y_pred: list[int]) -> dict:
    ids = list(range(len(LABELS)))
    report = classification_report(y_true, y_pred, labels=ids, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    f1 = {name: float(report[name]["f1-score"]) for name in LABELS}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "min_class_f1": float(min(f1.values())),
        "per_class_f1": f1, "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "passes_accuracy": float(accuracy_score(y_true, y_pred)) > 0.49,
        "passes_min_class_f1": float(min(f1.values())) >= 0.25,
    }


def write_csv(path: Path, columns: list[str], records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    write_csv(path, ["actual\\predicted", *LABELS], [
        {"actual\\predicted": label, **dict(zip(LABELS, row))}
        for label, row in zip(LABELS, matrix)
    ])


def save_checkpoint(path: Path, model, args, vocab, model_kwargs, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": args.model, "model_state": model.state_dict(), "model_kwargs": model_kwargs,
        "vocab": vocab, "labels": list(LABELS), "fields": args.fields,
        "strip_keyword_prefix": args.strip_keyword_prefix,
        "keep_keyword_prefix_only": args.keep_keyword_prefix_only,
        "max_length": args.max_length, "seed": args.seed, "extra": extra,
    }, path)


def open_checkpoint(path: Path, device: torch.device):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["labels"] != list(LABELS):
        raise ValueError("Checkpoint label order is not compatible")
    model = MODELS[data["model_name"]](**data["model_kwargs"])
    model.load_state_dict(data["model_state"])
    return model.to(device).eval(), data


def config_dict(args, fingerprint: str) -> dict:
    keys = ("model", "fields", "strip_keyword_prefix", "keep_keyword_prefix_only",
            "embedding_dim", "hidden_dim", "num_layers", "dropout", "max_length",
            "max_vocab", "min_freq", "batch_size", "epochs", "lr", "weight_decay",
            "grad_clip", "class_weighting", "select_metric", "patience", "seed")
    return {**{k: getattr(args, k) for k in keys}, "data_fingerprint": fingerprint,
            "config_name": configuration_name(args.fields, args.strip_keyword_prefix,
                                               args.keep_keyword_prefix_only)}


def row_prediction_records(rows, predictions, targets, positions, fold):
    return [{
        "csv_row_number": rows[i]["row_number"], "fold": fold,
        "actual": LABELS[true], "predicted": LABELS[pred],
        "description": rows[i]["description"], "sample_name": rows[i]["sample_name"],
        "text_excerpt": rows[i]["transcription"][:350].replace("\n", " "),
        "correct": int(pred == true),
    } for pred, true, i in zip(predictions, targets, positions)]


PRED_COLUMNS = ["csv_row_number", "fold", "actual", "predicted", "description", "sample_name", "text_excerpt", "correct"]


def run_fold(args, rows: list[dict], manifest: dict, record: dict, run_root: Path, device):
    fold = record["fold"]
    directory = run_root / f"fold_{fold:02d}"
    settings = config_dict(args, manifest["fingerprint"])
    cfg_path = directory / "run_config.json"
    complete = directory / "outer_metrics.json"
    if directory.exists() and args.overwrite:
        shutil.rmtree(directory)
    if directory.exists():
        if not cfg_path.exists() or read_json(cfg_path) != settings:
            raise ValueError(f"Existing folder {directory} has a different configuration. Choose another --runs-dir or use --overwrite.")
        if complete.exists() and (directory / "outer_predictions.csv").exists():
            print(f"Fold {fold}: complete, skipping (use --overwrite to rerun)")
            return
        # If interrupted, start fold clean rather than mixing artifacts.
        shutil.rmtree(directory)
    # Invalidate aggregate artifacts BEFORE changing any fold; never show stale 5-fold reports.
    for filename in ("cv_summary.json", "out_of_fold_predictions.csv", "misclassified.csv", "confusion_matrix.csv"):
        stale_aggregate = run_root / filename
        if stale_aggregate.exists():
            stale_aggregate.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(cfg_path, settings)
    fold_seed = args.seed + fold * 1000
    set_seed(fold_seed)
    fit = [rows[i] for i in record["fit"]]
    inner_val = [rows[i] for i in record["inner_validation"]]
    outer_test = [rows[i] for i in record["outer_test"]]
    texts = [make_text(r, args.fields, args.strip_keyword_prefix, args.keep_keyword_prefix_only) for r in fit]
    vocab = build_vocab(texts, args.max_vocab, args.min_freq, args.max_length)
    kwargs = model_config(args, len(vocab))
    model = MODELS[args.model](**kwargs).to(device)
    print(f"Fold {fold}/5 | device={device} | fit={len(fit)} | inner_val={len(inner_val)} "
          f"| outer_test={len(outer_test)} | vocab={len(vocab)} "
          f"| parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    fit_loader = loader(fit, vocab, args, shuffle=True)
    val_loader = loader(inner_val, vocab, args, shuffle=False)
    loss_fn = loss_function(args, fit, device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score, best_epoch, stale = -1.0, None, 0
    history = []
    best_path = directory / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, fit_loader, optim, loss_fn, device, args.grad_clip)
        guesses, actual, _, val_loss = infer(model, val_loader, device, loss_fn)
        stats = score(actual, guesses)
        selection = stats[args.select_metric]
        history.append({"epoch": epoch, "train_loss": train_loss, "inner_val_loss": val_loss,
                        "inner_val_accuracy": stats["accuracy"], "inner_val_macro_f1": stats["macro_f1"],
                        "inner_val_min_class_f1": stats["min_class_f1"]})
        print(f"  Epoch {epoch:02d}: train_loss={train_loss:.4f} inner_f1={stats['macro_f1']:.3f} "
              f"inner_acc={stats['accuracy']:.3f}", flush=True)
        if selection > best_score + 1e-8:
            best_score, best_epoch, stale = selection, epoch, 0
            save_checkpoint(best_path, model, args, vocab, kwargs,
                            {"fold": fold, "best_epoch": best_epoch, "select_metric": args.select_metric,
                             "best_metric": best_score, "evaluation_method": "inner_validation"})
            write_json(directory / "inner_validation_metrics.json", stats)
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break
    write_csv(directory / "history.csv", list(history[0]), history)
    write_json(directory / "split_rows.json", {
        "fit_csv_rows": [r["row_number"] for r in fit],
        "inner_validation_csv_rows": [r["row_number"] for r in inner_val],
        "outer_test_csv_rows": [r["row_number"] for r in outer_test],
    })
    # IMPORTANT: outer evaluation occurs ONCE after the best checkpoint was frozen.
    del model
    model, _ = open_checkpoint(best_path, device)
    guesses, actual, positions, _ = infer(model, loader(outer_test, vocab, args, False), device)
    outer_scores = score(actual, guesses)
    outer_scores.update(fold=fold, best_epoch=best_epoch, epochs_trained=len(history),
                        outer_samples=len(outer_test), inner_samples=len(inner_val), fit_samples=len(fit),
                        vocabulary_size=len(vocab), parameters=sum(p.numel() for p in model.parameters()))
    rows_out = row_prediction_records(outer_test, guesses, actual, positions, fold)
    write_csv(directory / "outer_predictions.csv", PRED_COLUMNS, rows_out)
    write_csv(directory / "misclassified.csv", PRED_COLUMNS,
              [record for record in rows_out if not record["correct"]])
    write_matrix(directory / "confusion_matrix.csv", outer_scores["confusion_matrix"])
    write_json(directory / "outer_metrics.json", outer_scores)  # last: marks fold complete
    print(f"  OUTER fold {fold}: accuracy={outer_scores['accuracy']:.3f} "
          f"macro_f1={outer_scores['macro_f1']:.3f} min_class_f1={outer_scores['min_class_f1']:.3f}", flush=True)


def summarize_cv(run_root: Path, manifest: dict, rows: list[dict], settings: dict):
    fold_metrics, oof = [], []
    for fold in range(1, FOLDS + 1):
        directory = run_root / f"fold_{fold:02d}"
        if not (directory / "outer_metrics.json").exists() or not (directory / "outer_predictions.csv").exists():
            print(f"Incomplete cross-validation ({len(fold_metrics)}/{FOLDS} complete); summary not generated yet")
            return
        if read_json(directory / "run_config.json") != settings:
            raise ValueError(f"Configuration mismatch at fold {fold}")
        fold_metrics.append(read_json(directory / "outer_metrics.json"))
        with (directory / "outer_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
            oof.extend(list(csv.DictReader(file)))
    # Compare original CSV row IDs; no row is ever evaluated twice or omitted.
    correct_rows = {str(r["row_number"]) for r in rows}
    actual_rows = [r["csv_row_number"] for r in oof]
    if len(actual_rows) != len(correct_rows) or set(actual_rows) != correct_rows:
        raise RuntimeError("OOF evaluation must cover every original labelled row exactly once")
    oof.sort(key=lambda r: int(r["csv_row_number"]))
    target = [LABELS.index(r["actual"]) for r in oof]
    guesses = [LABELS.index(r["predicted"]) for r in oof]
    pooled = score(target, guesses)

    def stats(values):
        a = np.array(values, dtype=float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "folds": [float(v) for v in a]}

    aggregate = {
        "cv_version": 1, "n_folds": FOLDS, "configuration": settings,
        "data_fingerprint": manifest["fingerprint"], "n_examples": len(rows),
        "metrics": {m: stats([fold[m] for fold in fold_metrics])
                    for m in ("accuracy", "macro_f1", "weighted_f1", "min_class_f1")},
        "per_class_f1": {label: stats([fold["per_class_f1"][label] for fold in fold_metrics])
                         for label in LABELS},
        "folds_passing_both_thresholds": sum(bool(f["passes_accuracy"] and f["passes_min_class_f1"])
                                             for f in fold_metrics),
        "best_epoch": stats([f["best_epoch"] for f in fold_metrics]),
        "vocabulary_size": stats([f["vocabulary_size"] for f in fold_metrics]),
        "parameter_count": stats([f["parameters"] for f in fold_metrics]),
        "pooled_oof": pooled,
        "fold_sizes": [f["outer_samples"] for f in fold_metrics],
    }
    write_json(run_root / "cv_summary.json", aggregate)
    write_csv(run_root / "out_of_fold_predictions.csv", PRED_COLUMNS, oof)
    write_csv(run_root / "misclassified.csv", PRED_COLUMNS,
              [r for r in oof if r["correct"] == "0"])
    write_matrix(run_root / "confusion_matrix.csv", pooled["confusion_matrix"])
    print(f"Complete CV: pooled_acc={pooled['accuracy']:.3f}, pooled_macro_f1={pooled['macro_f1']:.3f}, "
          f"pooled_min_class_f1={pooled['min_class_f1']:.3f}. Saved {run_root / 'cv_summary.json'}")


def cv_command(args):
    set_seed(args.seed)
    rows, skipped = read_labelled_csv(args.train_csv)
    runs = Path(args.runs_dir)
    manifest = create_or_load_folds(rows, runs, args.seed)
    print(f"Valid={len(rows)} | Invalid labels skipped={skipped} | Outer folds={FOLDS}")
    device = get_device(args.device)
    name = configuration_name(args.fields, args.strip_keyword_prefix, args.keep_keyword_prefix_only)
    folder = runs / "cv" / name / args.model
    settings = config_dict(args, manifest["fingerprint"])
    records = manifest["records"]
    if args.fold is not None:
        records = [r for r in records if r["fold"] == args.fold]
    for record in records:
        run_fold(args, rows, manifest, record, folder, device)
    summarize_cv(folder, manifest, rows, settings)


def make_folds_command(args):
    rows, skipped = read_labelled_csv(args.train_csv)
    data = create_or_load_folds(rows, Path(args.runs_dir), args.seed)
    print(f"Valid={len(rows)}, skipped={skipped}; folds frozen at {Path(args.runs_dir) / 'cv_splits.json'}")
    for record in data["records"]:
        print(f"  Fold {record['fold']}: fit={record['sizes']['fit']}, "
              f"inner_val={record['sizes']['inner_validation']}, outer_test={record['sizes']['outer_test']}")


def final_command(args):
    rows, skipped = read_labelled_csv(args.train_csv)
    device = get_device(args.device)
    set_seed(args.seed)
    vocab = build_vocab([make_text(r, args.fields, args.strip_keyword_prefix,
                                   args.keep_keyword_prefix_only) for r in rows],
                        args.max_vocab, args.min_freq, args.max_length)
    kwargs = model_config(args, len(vocab))
    model = MODELS[args.model](**kwargs).to(device)
    loss_fn = loss_function(args, rows, device)
    batches = loader(rows, vocab, args, True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Final fit on {len(rows)} labelled rows; skipped={skipped}; device={device}; epochs={args.epochs}")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, batches, optim, loss_fn, device, args.grad_clip)
        print(f"Epoch {epoch:02d}: train_loss={loss:.4f}", flush=True)
    path = Path(args.output_dir) / "final.pt"
    save_checkpoint(path, model, args, vocab, kwargs, {"final_fit": True, "epochs": args.epochs})
    print(f"Saved {path}")


def predict_command(args):
    device = get_device(args.device)
    model, checkpoint = open_checkpoint(Path(args.checkpoint), device)
    rows = read_unlabelled_csv(args.test_csv)
    settings = argparse.Namespace(
        fields=checkpoint["fields"], strip_keyword_prefix=checkpoint["strip_keyword_prefix"],
        keep_keyword_prefix_only=checkpoint.get("keep_keyword_prefix_only", False),
        max_length=checkpoint["max_length"], batch_size=args.batch_size,
    )
    predictions, _, positions, _ = infer(model, loader(rows, checkpoint["vocab"], settings, False, False), device)
    if list(positions) != list(range(len(rows))):
        raise RuntimeError("Prediction row order changed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for value in predictions:
            file.write(LABELS[value] + "\n")
    print(f"Wrote {len(predictions)} lines to {output}; original test order preserved")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--train-csv", required=True)
        p.add_argument("--model", choices=sorted(MODELS), required=True)
        p.add_argument("--fields", nargs="+", choices=FIELDS, default=list(FIELDS))
        group = p.add_mutually_exclusive_group()
        group.add_argument("--strip-keyword-prefix", action="store_true")
        group.add_argument("--keep-keyword-prefix-only", action="store_true")
        p.add_argument("--embedding-dim", type=int, default=128)
        p.add_argument("--hidden-dim", type=int, default=128)
        p.add_argument("--num-layers", type=int, default=1)
        p.add_argument("--dropout", type=float, default=0.3)
        p.add_argument("--max-length", type=int, default=512)
        p.add_argument("--max-vocab", type=int, default=20000)
        p.add_argument("--min-freq", type=int, default=2)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--lr", type=float, default=.001)
        p.add_argument("--weight-decay", type=float, default=0.0)
        p.add_argument("--grad-clip", type=float, default=1.0)
        p.add_argument("--class-weighting", choices=["none", "sqrt_inv"], default="none")
        p.add_argument("--device", default="auto", help="auto/cpu/cuda/xpu/mps")
        p.add_argument("--seed", type=int, default=42)

    folds = commands.add_parser("make-folds", help="Create/inspect shared five-fold manifest")
    folds.add_argument("--train-csv", required=True)
    folds.add_argument("--runs-dir", default="runs")
    folds.add_argument("--seed", type=int, default=42)
    folds.set_defaults(run=make_folds_command)

    cv = commands.add_parser("cv", help="Five-fold StratifiedGroupKFold with inner early stopping")
    common(cv)
    cv.add_argument("--runs-dir", default="runs")
    cv.add_argument("--fold", type=int, choices=range(1, FOLDS + 1),
                    help="Optional: run just one outer fold; rerun for all 5 to aggregate")
    cv.add_argument("--patience", type=int, default=6)
    cv.add_argument("--select-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    cv.add_argument("--overwrite", action="store_true", help="Retrain selected folds even if complete")
    cv.set_defaults(run=cv_command)

    final = commands.add_parser("train-final", help="Fit on all labelled rows after selecting settings by CV")
    common(final)
    final.add_argument("--output-dir", default="runs/final")
    final.set_defaults(run=final_command)

    pred = commands.add_parser("predict", help="Predict official unlabelled CSV in original order")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--test-csv", required=True)
    pred.add_argument("--output", default="results.txt")
    pred.add_argument("--batch-size", type=int, default=32)
    pred.add_argument("--device", default="auto")
    pred.set_defaults(run=predict_command)
    args = parser.parse_args()
    if args.command in ("cv", "train-final"):
        if len(args.fields) != len(set(args.fields)):
            parser.error("--fields cannot include duplicates")
        if (args.strip_keyword_prefix or args.keep_keyword_prefix_only) and "keywords" not in args.fields:
            parser.error("Keyword-prefix options require --fields to include keywords")
        for name in ("max_length", "max_vocab", "min_freq", "batch_size", "epochs"):
            if getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if args.max_vocab < 2 or not 0 <= args.dropout < 1 or args.num_layers < 1:
            parser.error("Invalid vocabulary size, dropout or layer count")
        if args.command == "cv" and args.patience < 0:
            parser.error("--patience cannot be negative")
    return args


if __name__ == "__main__":
    parsed = get_args()
    parsed.run(parsed)
