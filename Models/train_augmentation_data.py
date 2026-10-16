#!/usr/bin/env python3
"""Leakage-safe augmentation cross-validation with the ORIGINAL train.py models.

Neural models: 30 maximum epochs and patience 6, selecting the best model
using a separate, original-only inner validation split. The outer fold is
used once for evaluation. No sequence truncation is applied.

Preserves the augmentation provenance/audit checks in augmentation_cv_core.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from augmentation_cv_core import (
    METHODS, ProvenanceError, Synthetic, load_originals, load_synthetic,
    make_folds, materialize_folds, quality_report, verify_combined,
)
from augmentation_pkg.io_utils import FIELDS, stable_row_ids, write_table

# Import the user's original implementations, not the separate five-epoch
# augmentation_cv_models.birnn_predict model.
from train import (
    FIELDS as TRAIN_FIELDS,
    LABELS, MODELS, build_vocab, get_device, infer, loader, loss_function,
    make_text, model_config, score, set_seed, train_epoch,
)

NEURAL_MODELS = tuple(MODELS)
BASELINES = ("naive_bayes", "linear_svm")
ALL_MODELS = (*NEURAL_MODELS, *BASELINES)
CSV_COLUMNS = tuple(FIELDS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prepare", "cv"):
        p = sub.add_parser(command)
        p.add_argument("--train-csv", default="../train.csv")
        p.add_argument("--rwd-csv", default="data/augmented/train_rwd_final.csv")
        p.add_argument("--llm-csv", default="data/augmented/train_llm_final.csv")
        p.add_argument("--combined-csv", default="data/augmented/train_combined_final.csv")
        p.add_argument("--rwd-audit", default="reports/augmentation/rwd_final_audit.csv")
        p.add_argument("--llm-audit", default="reports/augmentation/llm_final_audit.csv")
        p.add_argument("--output-dir", default="reports/augmentation_cv")
        p.add_argument("--quarantine-suspicious", action="store_true")
        p.add_argument("--max-description-chars", type=int, default=2000)
        p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--fields", nargs="+", choices=FIELDS[1:],
                       default=["description", "keywords", "transcription"])
        p.add_argument("--strip-keyword-prefix", action="store_true")
        if command == "cv":
            p.add_argument("--model", choices=ALL_MODELS, default="bilstm_attention")
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--patience", type=int, default=6)
            p.add_argument("--batch-size", type=int, default=16)
            p.add_argument("--max-vocab", type=int, default=20000)
            p.add_argument("--min-freq", type=int, default=2)
            p.add_argument("--embedding-dim", type=int, default=128)
            p.add_argument("--hidden-dim", type=int, default=128)
            p.add_argument("--num-layers", type=int, default=1)
            p.add_argument("--dropout", type=float, default=0.3)
            p.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=0.001)
            p.add_argument("--weight-decay", type=float, default=0.0)
            p.add_argument("--grad-clip", type=float, default=1.0)
            p.add_argument("--class-weighting", choices=["none", "sqrt_inv"], default="none")
            p.add_argument("--device", default="auto", help="auto/cpu/cuda/xpu/mps")
            p.add_argument("--overwrite", action="store_true", help="Redo the chosen methods' model results")
    args = parser.parse_args(argv)
    if len(args.fields) != len(set(args.fields)):
        parser.error("--fields must not contain duplicates")
    if args.command == "cv":
        if args.epochs < 1 or args.patience < 0 or args.batch_size < 1:
            parser.error("--epochs and --batch-size must be positive, --patience >= 0")
        if args.model in BASELINES and args.class_weighting != "none":
            parser.error("--class-weighting applies only to neural models")
        if args.max_vocab < 2 or args.min_freq < 1:
            parser.error("Invalid vocabulary settings")
        if not 0 <= args.dropout < 1 or args.num_layers < 1:
            parser.error("Invalid dropout or number of recurrent layers")
    return args


def row_signature(row):
    return tuple((row.get(column) or "").strip() for column in CSV_COLUMNS)


def identity_without_description(row):
    """Augmentations change description only; this identifies their original note.

    Using non-description fields means even a deliberately modified description
    remains attached to its unaugmented source's train/validation partition.
    """
    return tuple((row.get(column) or "").strip().casefold() for column in
                 ("medical_specialty", "sample_name", "transcription", "keywords"))


def inner_group(row):
    """Group identical transcriptions, falling back to unchanged note fields."""
    transcript = (row.get("transcription") or "").strip().casefold()
    if transcript:
        return "transcript:" + transcript
    return "missing_transcript:" + "|".join((row.get(f) or "").strip().casefold()
                                                for f in ("sample_name", "keywords"))


def choose_inner_split(original_rows, seed):
    """Select ~20% of outer-training ORIGINALS for early stopping, group-safe."""
    y = np.array([LABELS.index(row["medical_specialty"]) for row in original_rows])
    groups = np.array([inner_group(row) for row in original_rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []
    target_dist = np.bincount(y, minlength=len(LABELS)) / len(y)
    for fit_indices, val_indices in splitter.split(np.arange(len(y)), y, groups):
        fit_counts = np.bincount(y[fit_indices], minlength=len(LABELS))
        val_counts = np.bincount(y[val_indices], minlength=len(LABELS))
        missing = int((fit_counts == 0).sum() + (val_counts == 0).sum())
        deviation = float(np.abs(val_counts / len(val_indices) - target_dist).sum())
        size_deviation = abs(len(val_indices) / len(y) - 0.2)
        candidates.append((missing, deviation + size_deviation,
                           fit_indices.tolist(), val_indices.tolist()))
    if not candidates:
        raise ValueError("Unable to create separate inner validation split")
    _, _, fit, val = min(candidates, key=lambda item: (item[0], item[1]))
    if {groups[i] for i in fit} & {groups[i] for i in val}:
        raise AssertionError("Inner validation contains shared source groups")
    return fit, val


def isolate_inner_fit(original_rows, augmented_rows, fit_indices, val_indices):
    """Use original-only inner validation; all synthetic children stay with fit."""
    original_counts = Counter(map(row_signature, original_rows))
    augmented_counts = Counter(map(row_signature, augmented_rows))
    for signature, n in original_counts.items():
        if augmented_counts[signature] < n:
            raise ValueError("Augmented fold is missing an original training record")
    known_parents = {identity_without_description(row) for row in original_rows}
    if any(identity_without_description(row) not in known_parents for row in augmented_rows):
        raise ValueError("Augmented row changed non-description fields; cannot prove its parent group")
    fit_groups = {inner_group(original_rows[i]) for i in fit_indices}
    val_groups = {inner_group(original_rows[i]) for i in val_indices}
    assert not fit_groups & val_groups
    fit_rows = [row for row in augmented_rows if inner_group(row) in fit_groups]
    val_rows = [original_rows[i] for i in val_indices]
    if not fit_rows or not val_rows:
        raise ValueError("Empty inner training/validation split")
    return fit_rows, val_rows


def neural_predict(args, original_rows, augmented_rows, outer_rows, fold, method, run_path):
    """Fit original model with max 30 / patience 6; never tune on outer fold."""
    import torch

    fit_idx, val_idx = choose_inner_split(original_rows, args.seed + fold)
    fit_rows, inner_val_rows = isolate_inner_fit(original_rows, augmented_rows, fit_idx, val_idx)
    device = get_device(args.device)
    set_seed(args.seed + 1000 * fold)
    # The ORIGINAL tokenizer, full-text vocabulary and dynamic-padding loader.
    args.keep_keyword_prefix_only = False
    fit_text = [make_text(row, args.fields, args.strip_keyword_prefix, False) for row in fit_rows]
    vocab = build_vocab(fit_text, args.max_vocab, args.min_freq)
    kwargs = model_config(args, len(vocab))
    model = MODELS[args.model](**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = loss_function(args, fit_rows, device)
    fit_loader = loader(fit_rows, vocab, args, shuffle=True)
    val_loader = loader(inner_val_rows, vocab, args, shuffle=False)
    run_path.mkdir(parents=True, exist_ok=True)
    best_checkpoint = run_path / "best.pt"
    history = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    print(f"    fit={len(fit_rows)} (includes augmentation) | "
          f"inner_val={len(inner_val_rows)} (originals only) | "
          f"outer_test={len(outer_rows)} | device={device} | vocab={len(vocab)}",
          flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, fit_loader, optimizer, loss_fn, device, args.grad_clip)
        val_prediction, val_target, _, val_loss = infer(model, val_loader, device, loss_fn)
        val_macro = f1_score(val_target, val_prediction, labels=list(range(len(LABELS))),
                             average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "inner_val_loss": val_loss, "inner_val_macro_f1": val_macro})
        print(f"    Epoch {epoch:02d}/{args.epochs}: loss={train_loss:.4f} "
              f"inner_val_macro_f1={val_macro:.4f}", flush=True)
        if val_macro > best_f1 + 1e-8:
            best_f1, best_epoch, stale_epochs = val_macro, epoch, 0
            torch.save({"model_state": model.state_dict(), "model_kwargs": kwargs,
                        "vocab": vocab, "labels": list(LABELS), "model_name": args.model,
                        "fields": args.fields, "strip_keyword_prefix": args.strip_keyword_prefix,
                        "max_length": None, "sequence_mode": "full_text",
                        "best_epoch": best_epoch, "best_inner_macro_f1": best_f1,
                        "class_weighting": args.class_weighting}, best_checkpoint)
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(f"    Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    with (run_path / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    # Reload the model at the best *inner* validation epoch, then evaluate outer.
    payload = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    best_model = MODELS[args.model](**payload["model_kwargs"])
    best_model.load_state_dict(payload["model_state"])
    best_model.to(device).eval()
    outer_prediction, outer_targets, positions, _ = infer(
        best_model, loader(outer_rows, payload["vocab"], args, shuffle=False), device)
    if positions != list(range(len(outer_rows))):
        raise RuntimeError("Evaluation changed held-out record order")
    outer_scores = score(outer_targets, outer_prediction)
    with (run_path / "training_summary.json").open("w", encoding="utf-8") as out:
        json.dump({"best_epoch": best_epoch, "trained_epochs": len(history),
                   "best_inner_macro_f1": best_f1, "outer_accuracy": outer_scores["accuracy"],
                   "outer_macro_f1": outer_scores["macro_f1"],
                   "fit_rows": len(fit_rows), "inner_val_rows": len(inner_val_rows),
                   "outer_rows": len(outer_rows), "class_weighting": args.class_weighting,
                   "max_length": None}, out, indent=2)
    return [LABELS[i] for i in outer_prediction]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_fingerprint(args, method):
    files = {"original_csv": args.train_csv}
    if method in ("rwd", "combined"):
        files.update(rwd_csv=args.rwd_csv, rwd_audit=args.rwd_audit)
    if method in ("llm", "combined"):
        files.update(llm_csv=args.llm_csv, llm_audit=args.llm_audit)
    if method == "combined":
        files["combined_csv"] = args.combined_csv
    return {name: sha256_file(path) for name, path in files.items()}


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def training_settings(args, selected_methods):
    config = {"model": args.model, "fields": args.fields,
              "strip_keyword_prefix": args.strip_keyword_prefix, "seed": args.seed,
              "folds": args.folds, "class_weighting": args.class_weighting,
              "max_length": None, "sequence_mode": "full_text",
              "quarantine_suspicious": args.quarantine_suspicious,
              "max_description_chars": args.max_description_chars,
              "original_data_hash": sha256_file(args.train_csv)}
    if args.model in NEURAL_MODELS:
        config.update(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                      max_vocab=args.max_vocab, min_freq=args.min_freq,
                      embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers, dropout=args.dropout,
                      lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                      validation_metric="inner_macro_f1")
    elif args.model == "naive_bayes":
        config.update(algorithm="Original baseline.py make_baseline: Porter-stemmed TF-IDF + MultinomialNB(alpha=1)")
    else:
        config.update(algorithm="Linear SVM (original augmentation reference)")
    return config


def model_output_path(args, destination):
    mode = "weighted_sqrt_inv" if args.class_weighting == "sqrt_inv" else "normal"
    field_slug = "__".join(args.fields) + ("__prefix_removed" if args.strip_keyword_prefix else "")
    return destination / "original_models" / args.model / mode / (field_slug + "__full_text")


def run(args):
    source = Path(args.train_csv).resolve()
    destination = Path(args.output_dir).resolve()
    if source.name.startswith("train_") and any(k in source.name for k in ("rwd", "llm", "combined")):
        raise ValueError("--train-csv MUST be original train.csv, not an augmented file")
    base = load_originals(source)
    selected_methods = list(dict.fromkeys(args.methods))
    if "description" not in args.fields and any(m != "baseline" for m in selected_methods):
        raise ValueError("Augmentations change 'description'; include --fields description")
    classes, bad = quality_report(base, max_description_chars=args.max_description_chars)
    if len(classes) != len(LABELS) or classes != set(LABELS):
        raise ValueError(f"Expected precisely the 12 original classes, found {len(classes)}")
    quarantined = {int(row["index_0_based"]) for row in bad}
    destination.mkdir(parents=True, exist_ok=True)
    write_table(destination / "data_quality_quarantine.csv", [
        "index_0_based", "row_number_1_based_including_header", "source_row_id",
        "specialty", "description_chars", "reasons", "description_preview"], bad)
    if bad and not args.quarantine_suspicious:
        raise ValueError(f"Found {len(bad)} suspicious original rows. Review "
                         f"{destination / 'data_quality_quarantine.csv'} and use "
                         "--quarantine-suspicious or repair/rebuild augmentation data")
    eligible = [i for i in range(len(base)) if i not in quarantined]
    label_counts = Counter(base[i]["medical_specialty"] for i in eligible)
    if len(label_counts) != len(LABELS) or min(label_counts.values()) < args.folds:
        raise ValueError("Too few eligible examples in one or more classes")

    synthetic: dict[str, list[Synthetic]] = {}
    if "rwd" in selected_methods or "combined" in selected_methods:
        synthetic["rwd"] = load_synthetic(base=base, path=Path(args.rwd_csv).resolve(),
                                          audit_path=Path(args.rwd_audit).resolve(), method="rwd")
    if "llm" in selected_methods or "combined" in selected_methods:
        synthetic["llm"] = load_synthetic(base=base, path=Path(args.llm_csv).resolve(),
                                          audit_path=Path(args.llm_audit).resolve(), method="llm")
    if "combined" in selected_methods:
        synthetic["combined"] = verify_combined(
            base, synthetic["rwd"], synthetic["llm"], Path(args.combined_csv).resolve())

    folds = make_folds(base, eligible, args.folds, args.seed)
    fold_dir = destination / "folds"
    # Materialize an unaugmented training file for the *inner* validation split,
    # even when user requests only rwd/llm/combined.
    materialize_methods = list(dict.fromkeys(["baseline", *selected_methods]))
    stats = materialize_folds(original=base, synthetic=synthetic, folds=folds,
                              methods=materialize_methods, dest=fold_dir, source=source,
                              quarantine=bad if args.quarantine_suspicious else [])
    print(f"Verified {len(base)} originals; {len(eligible)} eligible; "
          f"{len(quarantined)} excluded; {len(folds)} outer folds")
    for s in stats:
        print(f"Fold {s['fold']}: " + ", ".join(
            f"{m}={s[m]['train_total']} training ({s[m]['train_synthetic']} synthetic)"
            for m in selected_methods))
    if args.command == "audit":
        print("Audit completed (no models trained)")
        return
    if args.command == "prepare":
        print(f"Fold CSV files prepared at {fold_dir} (no models trained)")
        return

    if args.model == "naive_bayes":
        from baseline import make_baseline
    if args.model == "linear_svm":
        from augmentation_cv_models import svm_predict

    labels = sorted(classes)
    original_ids = stable_row_ids(base)
    model_dir = model_output_path(args, destination)
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = training_settings(args, selected_methods)
    cfg_path = model_dir / "run_config.json"
    if cfg_path.exists():
        previous = json.loads(cfg_path.read_text(encoding="utf-8"))
        if previous != cfg:
            raise ValueError(f"Different settings or input dataset in {model_dir}; "
                             "use another --output-dir. Do not mix comparisons.")
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    previous_results = {r["method"]: r for r in read_table(model_dir / "comparison.csv")} if (
        model_dir / "comparison.csv").exists() else {}
    existing_fold_metrics = read_table(model_dir / "fold_metrics.csv") if (
        model_dir / "fold_metrics.csv").exists() else []
    existing_fold_metrics = [r for r in existing_fold_metrics if r["method"] not in selected_methods]

    for method in selected_methods:
        predictions = []
        fold_metrics = []
        method_dir = model_dir / method
        source_hashes = method_fingerprint(args, method)
        hash_file = method_dir / "source_hashes.json"
        if hash_file.exists() and json.loads(hash_file.read_text(encoding="utf-8")) != source_hashes and not args.overwrite:
            raise ValueError(f"Augmentation files changed for {method}; use --overwrite "
                             "to recompute, or a separate --output-dir")
        if method_dir.exists() and args.overwrite:
            shutil.rmtree(method_dir)
        method_dir.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
        for fold_number, (train_indices, val_indices) in enumerate(folds, 1):
            fold_source = fold_dir / f"fold_{fold_number:02d}"
            original_rows = load_originals(fold_source / "train_baseline.csv")
            train_rows = load_originals(fold_source / f"train_{method}.csv")
            outer_rows = load_originals(fold_source / "validation.csv")
            if [row_signature(r) for r in outer_rows] != [row_signature(base[i]) for i in val_indices]:
                raise AssertionError("The outer evaluation fold has changed")
            if Counter(map(row_signature, original_rows)) != Counter(
                row_signature(base[i]) for i in train_indices):
                raise AssertionError("Original outer-training partition has changed")
            fold_path = method_dir / f"fold_{fold_number:02d}"
            pred_path = fold_path / "outer_predictions.csv"
            if pred_path.exists() and not args.overwrite:
                cached = read_table(pred_path)
                if len(cached) != len(val_indices) or [p["source_row_id"] for p in cached] != [
                    original_ids[i] for i in val_indices]:
                    raise ValueError("Saved predictions do not match current folds; use --overwrite")
                fold_predictions = [p["predicted_label"] for p in cached]
                print(f"method={method} fold={fold_number}/{len(folds)} already done; reusing", flush=True)
            else:
                print(f"Training model={args.model}, augmentation={method}, "
                      f"fold={fold_number}/{len(folds)}, max_epochs={args.epochs}, "
                      f"patience={args.patience}", flush=True)
                if args.model in NEURAL_MODELS:
                    fold_predictions = neural_predict(args, original_rows, train_rows, outer_rows,
                                                      fold_number, method, fold_path)
                else:
                    if args.model == "naive_bayes":
                        def as_text(row):
                            return row["description"] if args.fields == ["description"] else make_text(
                                row, args.fields, args.strip_keyword_prefix, False)
                        estimator = make_baseline()
                        estimator.fit([as_text(r) for r in train_rows],
                                      [r["medical_specialty"] for r in train_rows])
                        fold_predictions = estimator.predict([as_text(r) for r in outer_rows]).tolist()
                    else:
                        texts = [make_text(r, args.fields, args.strip_keyword_prefix, False) for r in train_rows]
                        vals = [make_text(r, args.fields, args.strip_keyword_prefix, False) for r in outer_rows]
                        fold_predictions = svm_predict(texts,
                            [r["medical_specialty"] for r in train_rows], vals,
                            seed=args.seed + fold_number)
                    fold_path.mkdir(parents=True, exist_ok=True)
                write_table(pred_path, ["source_row_id", "original_row_index", "fold",
                                        "true_label", "predicted_label", "method"], [{
                    "source_row_id": original_ids[idx], "original_row_index": idx,
                    "fold": fold_number, "true_label": base[idx]["medical_specialty"],
                    "predicted_label": prediction, "method": method,
                } for idx, prediction in zip(val_indices, fold_predictions)])
            for idx, prediction in zip(val_indices, fold_predictions):
                predictions.append({"source_row_id": original_ids[idx], "original_row_index": idx,
                                    "fold": fold_number, "true_label": base[idx]["medical_specialty"],
                                    "predicted_label": prediction, "method": method})
            actual_labels = [r["medical_specialty"] for r in outer_rows]
            fold_accuracy = accuracy_score(actual_labels, fold_predictions)
            fold_macro = f1_score(actual_labels, fold_predictions, labels=labels,
                                  average="macro", zero_division=0)
            fold_metrics.append({"method": method, "fold": fold_number,
                                 "validation_originals": len(outer_rows),
                                 "accuracy": float(fold_accuracy), "macro_f1": float(fold_macro)})
            print(f"  Outer fold {fold_number}: accuracy={fold_accuracy:.4f}, "
                  f"macro_F1={fold_macro:.4f}", flush=True)
        if len(predictions) != len(eligible) or len({p["source_row_id"] for p in predictions}) != len(eligible):
            raise AssertionError("OOF must cover each eligible original exactly once")
        actual = [p["true_label"] for p in predictions]
        predicted = [p["predicted_label"] for p in predictions]
        prec, rec, per_f1, support = precision_recall_fscore_support(
            actual, predicted, labels=labels, zero_division=0)
        accuracy = accuracy_score(actual, predicted)
        macro_f1 = f1_score(actual, predicted, labels=labels,
                            average="macro", zero_division=0)
        weighted_f1 = f1_score(actual, predicted, labels=labels,
                               average="weighted", zero_division=0)
        minimum_f1 = min(per_f1)
        previous_results[method] = {
            "method": method, "accuracy": float(accuracy), "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1), "minimum_class_f1": float(minimum_f1),
            "above_49pct_accuracy": bool(accuracy > 0.49),
            "all_class_f1_at_least_25pct": bool(minimum_f1 >= 0.25),
            "validation_originals": len(predictions),
        }
        write_table(model_dir / f"oof_{method}.csv", ["source_row_id", "original_row_index",
                    "fold", "true_label", "predicted_label", "method"], predictions)
        write_table(model_dir / f"per_class_{method}.csv", ["specialty", "precision", "recall", "f1", "support"], [
            {"specialty": l, "precision": float(p), "recall": float(r),
             "f1": float(f), "support": int(s)}
            for l,p,r,f,s in zip(labels, prec, rec, per_f1, support)])
        cm = confusion_matrix(actual, predicted, labels=labels)
        write_table(model_dir / f"confusion_{method}.csv", ["true_label", *labels], [
            dict(zip(["true_label", *labels], [lab, *cnt.tolist()]))
            for lab, cnt in zip(labels, cm)])
        existing_fold_metrics.extend(fold_metrics)
        write_table(model_dir / "fold_metrics.csv", ["method", "fold", "validation_originals",
                    "accuracy", "macro_f1"], existing_fold_metrics)
        write_table(model_dir / "comparison.csv", list(next(iter(previous_results.values())).keys()),
                    list(previous_results.values()))
        print(f"RESULT {method}: acc={accuracy:.4f}, macro_F1={macro_f1:.4f}, "
              f"min_class_F1={minimum_f1:.4f}", flush=True)
    print(f"Saved results to {model_dir}")
    # Rebuild the unified dashboard after successful augmentation evaluation.
    # Reuse the main report generator so Results/index.html links to
    # Results/Augmentation/index.html. Never delete or overwrite training runs.
    try:
        from compare_results import generate
        runs_dir = Path("runs").resolve()
        generate(runs_dir, Path("Results").resolve())
    except Exception as exc:
        print(f"WARNING: Saved training results, but HTML dashboard refresh failed: {exc}")
        print("Refresh manually with: python compare_results.py")


def main(argv=None):
    try:
        args = parse_args(argv)
        run(args)
        return 0
    except (ValueError, OSError, ProvenanceError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
