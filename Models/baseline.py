"""TF-IDF + MultinomialNB baseline, including the professor-specified description variant.

Uses the SAME frozen 5-fold outer splits as train.py. Naive Bayes has no early
stopping, so each fold fits on all available outer-training rows (~80%) and is
scored only on that fold's held-out rows (~20%).

Usage: python baseline.py --train-csv "train.csv" [--fields FIELD [FIELD ...]]
       python baseline.py --train-csv "train.csv" --fields keywords --strip-keyword-prefix
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from functools import lru_cache

import joblib
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from train import (
    FIELDS, FOLDS, LABELS, PRED_COLUMNS,
    configuration_name, create_or_load_folds, make_text, read_labelled_csv, read_json, row_prediction_records,
    score, summarize_cv, write_csv, write_json, write_matrix,
)

# Top-level function: allows the fitted sklearn pipeline to be reloaded from joblib.
_WORDS = re.compile(r"(?u)\b\w+\b")
_STEMMER = PorterStemmer()


@lru_cache(maxsize=100_000)
def stem_cached(word: str) -> str:
    return _STEMMER.stem(word)


def stemmed_words(text: str) -> list[str]:
    """Lowercase, split into word unigrams, apply Porter stemming; no stopword removal."""
    return [stem_cached(word) for word in _WORDS.findall(text.lower())]


def make_baseline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=stemmed_words,
            token_pattern=None,
            lowercase=False,  # stemmed_words already lowercases
            ngram_range=(1, 1),
            min_df=1,
            smooth_idf=True,
            norm="l2",
        )),
        ("naive_bayes", MultinomialNB(alpha=1.0)),
    ])


def settings_for(manifest: dict, fields: list[str], strip: bool, keep: bool) -> dict:
    settings = {
        "model": "naive_bayes",
        "config_name": configuration_name(fields, strip, keep),
        "fields": fields,
        "data_fingerprint": manifest["fingerprint"],
        "seed": manifest["seed"],
        "algorithm": "MultinomialNB",
        "alpha": 1.0,
        "features": "TF-IDF word unigrams",
        "lowercase": True,
        "stemming": "NLTK PorterStemmer",
        "tfidf_min_df": 1,
        "tfidf_norm": "l2",
        "fit_partition": "full outer training split (no inner validation)",
    }
    # Preserve the existing description-only settings byte-for-byte, so prior
    # five-fold results can be reused without overwriting or training again.
    if fields != ["description"] or strip or keep:
        settings["strip_keyword_prefix"] = strip
        settings["keep_keyword_prefix_only"] = keep
    return settings


def input_text(row: dict, settings: dict) -> str:
    """Match the neural pipelines' field composition, except keep the
    original professor baseline using plain description text."""
    if settings["fields"] == ["description"]:
        return row["description"]
    return make_text(row, settings["fields"],
                     settings.get("strip_keyword_prefix", False),
                     settings.get("keep_keyword_prefix_only", False))


def run_fold(rows: list[dict], record: dict, folder: Path, settings: dict, overwrite: bool) -> None:
    fold = record["fold"]
    directory = folder / f"fold_{fold:02d}"
    if directory.exists() and overwrite:
        shutil.rmtree(directory)
    if directory.exists():
        config = directory / "run_config.json"
        if not config.exists() or read_json(config) != settings:
            raise ValueError(
                f"Different settings already saved in {directory}. Use --overwrite or another --runs-dir."
            )
        if (directory / "outer_metrics.json").exists() and (directory / "outer_predictions.csv").exists():
            print(f"Fold {fold}/{FOLDS}: already complete, skipping", flush=True)
            return
        shutil.rmtree(directory)  # Incomplete fold; restart cleanly.

    # Changing any one fold invalidates previously saved five-fold aggregates.
    for name in ("cv_summary.json", "out_of_fold_predictions.csv", "misclassified.csv", "confusion_matrix.csv"):
        aggregate = folder / name
        if aggregate.exists():
            aggregate.unlink()

    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "run_config.json", settings)

    # Unlike the neural models, this non-iterative baseline does not select an
    # epoch on the inner split, so it can fit on the full outer-training pool.
    held_out = set(record["outer_test"])
    training_indices = [i for i in range(len(rows)) if i not in held_out]
    outer_indices = record["outer_test"]
    train_rows = [rows[i] for i in training_indices]
    eval_rows = [rows[i] for i in outer_indices]

    classifier = make_baseline()
    classifier.fit(
        [input_text(r, settings) for r in train_rows],
        [r["medical_specialty"] for r in train_rows],
    )
    guessed_labels = classifier.predict([input_text(r, settings) for r in eval_rows]).tolist()
    guesses = [LABELS.index(value) for value in guessed_labels]
    truths = [LABELS.index(r["medical_specialty"]) for r in eval_rows]
    fold_metrics = score(truths, guesses)
    features = len(classifier.named_steps["tfidf"].vocabulary_)
    fold_metrics.update(
        fold=fold,
        best_epoch=0,  # Not applicable to a non-iterative classifier.
        epochs_trained=0,
        outer_samples=len(eval_rows),
        inner_samples=0,
        fit_samples=len(train_rows),
        vocabulary_size=features,
        parameters=0,  # No trainable neural-network parameters.
        estimator="MultinomialNB(alpha=1.0)",
        preprocessing=f"{settings['config_name']}: lowercase + Porter stemming + TF-IDF unigrams",
    )

    # Store the fitted pipeline as well as per-fold predictions and statistics.
    joblib.dump(classifier, directory / "model.joblib")
    write_json(directory / "split_rows.json", {
        "fit_csv_rows": [r["row_number"] for r in train_rows],
        "inner_validation_csv_rows": [],
        "outer_test_csv_rows": [r["row_number"] for r in eval_rows],
    })
    predictions = row_prediction_records(eval_rows, guesses, truths, list(range(len(eval_rows))), fold)
    write_csv(directory / "outer_predictions.csv", PRED_COLUMNS, predictions)
    write_csv(directory / "misclassified.csv", PRED_COLUMNS, [r for r in predictions if not r["correct"]])
    write_matrix(directory / "confusion_matrix.csv", fold_metrics["confusion_matrix"])
    write_json(directory / "outer_metrics.json", fold_metrics)  # Completion marker, written last.
    print(f"Fold {fold}/{FOLDS}: train={len(train_rows)} outer={len(eval_rows)} "
          f"features={features} accuracy={fold_metrics['accuracy']:.3f} "
          f"macro_f1={fold_metrics['macro_f1']:.3f} "
          f"min_class_f1={fold_metrics['min_class_f1']:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--fields", nargs="+", choices=FIELDS, default=["description"])
    keyword_modes = parser.add_mutually_exclusive_group()
    keyword_modes.add_argument("--strip-keyword-prefix", action="store_true")
    keyword_modes.add_argument("--keep-keyword-prefix-only", action="store_true")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=range(1, FOLDS + 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(args.fields) != len(set(args.fields)):
        parser.error("Each field may be specified only once")
    if (args.strip_keyword_prefix or args.keep_keyword_prefix_only) and "keywords" not in args.fields:
        parser.error("Keyword-prefix options require including --fields keywords")
    rows, skipped = read_labelled_csv(args.train_csv)
    manifest = create_or_load_folds(rows, Path(args.runs_dir), args.seed)
    settings = settings_for(manifest, args.fields, args.strip_keyword_prefix, args.keep_keyword_prefix_only)
    folder = Path(args.runs_dir) / "cv" / settings["config_name"] / "naive_bayes"
    print(f"Naive Bayes baseline | valid={len(rows)} skipped={skipped} "
          f"folds={FOLDS} | input={settings['config_name']}", flush=True)

    records = manifest["records"]
    if args.fold is not None:
        records = [r for r in records if r["fold"] == args.fold]
    for record in records:
        run_fold(rows, record, folder, settings, args.overwrite)
    summarize_cv(folder, manifest, rows, settings)


if __name__ == "__main__":
    main()
