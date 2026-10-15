#!/usr/bin/env python3
"""Leakage-safe cross-validation for Help the Doctor data augmentation.

Runs from Models/ alongside augment_data.py. Does not modify existing train.py.
Uses original clinical records for grouped folds and quarantines corrupted cases only
when the user explicitly opts in. All CV validation examples are untouched originals.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from augmentation_cv_core import (
    METHODS, ProvenanceError, Synthetic, clean_text, load_originals, load_synthetic,
    make_folds, materialize_folds, quality_report, verify_combined,
)
from augmentation_pkg.io_utils import FIELDS, stable_row_ids, write_table


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Leakage-safe augmentation comparison; run from Models directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prepare", "cv"):
        p = sub.add_parser(command)
        p.add_argument("--train-csv", default="../train.csv",
                       help="Original labeled CSV, never one of the augmented CSVs")
        p.add_argument("--rwd-csv", default="data/augmented/train_rwd_final.csv")
        p.add_argument("--llm-csv", default="data/augmented/train_llm_final.csv")
        p.add_argument("--combined-csv", default="data/augmented/train_combined_final.csv")
        p.add_argument("--rwd-audit", default="reports/augmentation/rwd_final_audit.csv")
        p.add_argument("--llm-audit", default="reports/augmentation/llm_final_audit.csv")
        p.add_argument("--output-dir", default="reports/augmentation_cv")
        p.add_argument("--quarantine-suspicious", action="store_true",
                       help="Explicitly exclude suspect original records and their augmentations")
        p.add_argument("--max-description-chars", type=int, default=2000)
        p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--fields", nargs="+", choices=FIELDS[1:],
                       default=["description", "keywords", "transcription"])
        p.add_argument("--strip-keyword-prefix", action="store_true")
        if command == "cv":
            p.add_argument("--model", choices=["birnn_attention", "linear_svm"],
                           default="birnn_attention")
            p.add_argument("--epochs", type=int, default=5)
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--max-tokens", type=int, default=256)
            p.add_argument("--max-vocab", type=int, default=30000)
            p.add_argument("--embedding-dim", type=int, default=96)
            p.add_argument("--hidden-dim", type=int, default=96)
            p.add_argument("--learning-rate", type=float, default=0.002)
            p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args(argv)


def run(args):
    source = Path(args.train_csv).resolve()
    destination = Path(args.output_dir).resolve()
    if source.name.startswith("train_") and ("rwd" in source.name or "llm" in source.name or "combined" in source.name):
        raise ValueError("--train-csv MUST point to the ORIGINAL train.csv, not an augmented file")
    base = load_originals(source)
    selected_methods = list(dict.fromkeys(args.methods))
    if "description" not in args.fields and any(m != "baseline" for m in selected_methods):
        raise ValueError("Your augmentations modify 'description', but --fields omits description. "
                         "Use --fields description keywords transcription to measure augmentation.")
    if len(set(args.fields)) < len(args.fields):
        raise ValueError("--fields includes duplicate column names")

    classes, bad = quality_report(base, max_description_chars=args.max_description_chars)
    if len(classes) != 12:
        raise ValueError(f"Expected 12 medical specialties, found {len(classes)} after data-quality review; "
                         "inspect and correct the source CSV.")
    quarantined = {int(r["index_0_based"]) for r in bad}
    destination.mkdir(parents=True, exist_ok=True)
    write_table(destination / "data_quality_quarantine.csv", [
        "index_0_based", "row_number_1_based_including_header", "source_row_id", "specialty",
        "description_chars", "reasons", "description_preview"], bad)
    if bad and not args.quarantine_suspicious:
        raise ValueError(f"Found {len(bad)} suspicious original rows. Review "
                         f"{destination / 'data_quality_quarantine.csv'} and run again "
                         "with --quarantine-suspicious to exclude them, or repair the source "
                         "and REGENERATE all augmentations and audits.")
    eligible = [i for i in range(len(base)) if i not in quarantined]
    label_counts = Counter(base[i]["medical_specialty"] for i in eligible)
    if set(label_counts) != classes or len(label_counts) != 12:
        raise ValueError("Quarantine would remove an entire medical specialty; refusing CV")
    if min(label_counts.values()) < args.folds:
        raise ValueError("Some classes have fewer original records than the number of CV folds")

    synthetic: dict[str, list[Synthetic]] = {}
    needs_rwd = "rwd" in selected_methods or "combined" in selected_methods
    needs_llm = "llm" in selected_methods or "combined" in selected_methods
    if needs_rwd:
        synthetic["rwd"] = load_synthetic(
            base=base, path=Path(args.rwd_csv).resolve(),
            audit_path=Path(args.rwd_audit).resolve(), method="rwd")
    if needs_llm:
        synthetic["llm"] = load_synthetic(
            base=base, path=Path(args.llm_csv).resolve(),
            audit_path=Path(args.llm_audit).resolve(), method="llm")
    if "combined" in selected_methods:
        synthetic["combined"] = verify_combined(
            base, synthetic["rwd"], synthetic["llm"],
            Path(args.combined_csv).resolve())

    folds = make_folds(base, eligible, args.folds, args.seed)
    fold_dir = destination / "folds"
    stats = materialize_folds(original=base, synthetic=synthetic, folds=folds,
                              methods=selected_methods, dest=fold_dir, source=source,
                              quarantine=bad if args.quarantine_suspicious else [])
    print(f"Verified {len(base)} originals, {len(eligible)} eligible originals, "
          f"{len(quarantined)} quarantined records, 12 classes")
    print("Verified synthetic counts:", ", ".join(f"{k}={len(v)}" for k,v in synthetic.items()) or "none")
    for s in stats:
        aug_summary = ", ".join(f"{k}: {s[k]['train_total']} train "
                                f"({s[k]['train_synthetic']} augmented)" for k in selected_methods)
        print(f"Fold {s['fold']}: {s['validation_originals']} original validation rows; {aug_summary}")
    print(f"Fold CSVs and manifest: {fold_dir}")
    if args.command == "audit":
        print("AUDIT COMPLETE: no models were trained")
        return
    if args.command == "prepare":
        print("PREPARE COMPLETE: use fold-specific train_*.csv and validation.csv for your existing model")
        return

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    from augmentation_cv_models import birnn_predict, svm_predict
    labels = sorted(classes)
    original_ids = stable_row_ids(base)
    model_dir = destination / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    fold_scores = []
    for method in selected_methods:
        predictions = []
        actuals = []
        for fold_number, (train_indices, val_indices) in enumerate(folds, 1):
            fold_source = fold_dir / f"fold_{fold_number:02d}"
            train_rows = load_originals(fold_source / f"train_{method}.csv")
            val_rows = load_originals(fold_source / "validation.csv")
            assert val_rows == [base[i] for i in val_indices], "Validation data changed"
            train_texts = [clean_text(row, args.fields, args.strip_keyword_prefix) for row in train_rows]
            val_texts = [clean_text(row, args.fields, args.strip_keyword_prefix) for row in val_rows]
            train_labels = [row["medical_specialty"] for row in train_rows]
            fold_seed = args.seed + fold_number
            print(f"Training model={args.model}, method={method}, fold={fold_number}/{len(folds)}, "
                  f"train={len(train_rows)}, validation originals={len(val_rows)}", flush=True)
            if args.model == "linear_svm":
                predicted = svm_predict(train_texts, train_labels, val_texts, seed=fold_seed)
            else:
                predicted = birnn_predict(
                    train_texts, train_labels, val_texts, labels,
                    seed=fold_seed, epochs=args.epochs, batch_size=args.batch_size,
                    max_tokens=args.max_tokens, max_vocab=args.max_vocab,
                    embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                    learning_rate=args.learning_rate, device_name=args.device)
            for idx, prediction in zip(val_indices, predicted):
                predictions.append({"source_row_id": original_ids[idx],
                                    "original_row_index": idx,
                                    "fold": fold_number,
                                    "true_label": base[idx]["medical_specialty"],
                                    "predicted_label": prediction,
                                    "method": method})
            actuals.extend(row["medical_specialty"] for row in val_rows)
            fold_actual = [row["medical_specialty"] for row in val_rows]
            fold_accuracy = accuracy_score(fold_actual, predicted)
            fold_macro = f1_score(fold_actual, predicted, labels=labels, average="macro", zero_division=0)
            fold_scores.append({"method": method, "fold": fold_number,
                                "validation_originals": len(fold_actual),
                                "accuracy": float(fold_accuracy), "macro_f1": float(fold_macro)})
            write_table(model_dir / "fold_metrics.csv",
                        ["method", "fold", "validation_originals", "accuracy", "macro_f1"], fold_scores)
            print(f"  Fold {fold_number} accuracy: {fold_accuracy:.4f}, macro-F1: {fold_macro:.4f}", flush=True)
        if len(predictions) != len(eligible) or len({p["source_row_id"] for p in predictions}) != len(eligible):
            raise AssertionError("OOF output must contain each eligible original exactly once")
        true = [p["true_label"] for p in predictions]
        pred = [p["predicted_label"] for p in predictions]
        precision, recall, f1s, supports = precision_recall_fscore_support(
            true, pred, labels=labels, zero_division=0)
        accuracy = accuracy_score(true, pred)
        macro_f1 = f1_score(true, pred, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(true, pred, labels=labels, average="weighted", zero_division=0)
        min_f1 = min(f1s)
        results[method] = {"method": method, "accuracy": float(accuracy),
                           "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
                           "minimum_class_f1": float(min_f1),
                           "above_49pct_accuracy": bool(accuracy > 0.49),
                           "all_class_f1_at_least_25pct": bool(min_f1 >= .25),
                           "validation_originals": len(predictions)}
        write_table(model_dir / f"oof_{method}.csv",
                    ["source_row_id", "original_row_index", "fold", "true_label", "predicted_label", "method"],
                    predictions)
        write_table(model_dir / f"per_class_{method}.csv", ["specialty", "precision", "recall", "f1", "support"],
                    [{"specialty": name, "precision": float(p), "recall": float(r),
                      "f1": float(f), "support": int(s)} for name, p, r, f, s in zip(labels, precision, recall, f1s, supports)])
        matrix = confusion_matrix(true, pred, labels=labels)
        write_table(model_dir / f"confusion_{method}.csv",
                    ["true_label", *labels],
                    [dict(zip(["true_label", *labels], [name, *counts.tolist()]))
                     for name, counts in zip(labels, matrix)])
        write_table(model_dir / "comparison.csv", list(next(iter(results.values())).keys()), list(results.values()))
        print(f"RESULT {method}: accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}, "
              f"minimum per-class F1={min_f1:.4f}", flush=True)
    run_config = {"model": args.model, "fields": args.fields, "strip_keyword_prefix": args.strip_keyword_prefix,
                  "folds": args.folds, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                  "max_tokens": args.max_tokens, "max_vocab": args.max_vocab,
                  "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
                  "learning_rate": args.learning_rate, "device": args.device,
                  "quarantined_originals": len(bad), "methods": selected_methods,
                  "note": "BiLSTM attention is a stand-alone reference implementation, NOT the existing train.py model."}
    (model_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"\nAll evaluation files saved to {model_dir}", flush=True)


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
