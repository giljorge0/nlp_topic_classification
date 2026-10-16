#!/usr/bin/env python3
"""Export misclassified clinical cases for each medical-specialty classifier.

This script ONLY evaluates already-generated predictions on a LABELED, held-out
validation/test set. It does not train models or predict missing labels.

CLI example (prediction files contain ONE predicted specialty per row):
    python export_misclassifications.py \
        --data validation.csv \
        --pred svm=svm_validation_predictions.txt \
        --pred rnn=rnn_validation_predictions.txt

Notebook / training-script example:
    from export_misclassifications import save_model_errors
    save_model_errors(
        model_name="svm",
        data=validation_df,              # original, unprocessed clinical fields
        y_true=validation_df["medical_specialty"],
        y_pred=svm_predictions,           # exactly one label per validation row
        output_dir="predictions",
    )

Outputs for EACH model:
    predictions/MODEL_misclassified.csv    every incorrect prediction, full fields
    predictions/MODEL_misclassified.md     easy-to-read error cases and notes
    predictions/MODEL_confusion_matrix.csv correct-vs-predicted counts
    predictions/MODEL_all_predictions.csv  full evaluation with error indicator

Only select examples for the report after manually checking them. The
'interpretation' column is intentionally left blank for human analysis.

IMPORTANT: test_no_labels.csv cannot be used to detect errors because it does
not contain ground-truth medical_specialty labels. Also, don't use your model's
training rows as the evaluation set; use a held-out split.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_FIELDS = ("description", "sample_name", "transcription", "keywords")
KNOWN_LABEL_COLUMNS = (
    "predicted_medical_specialty",
    "predicted_label",
    "prediction",
    "medical_specialty",
    "label",
)


def _as_aligned_list(values: Iterable[Any], expected_index: pd.Index, name: str) -> list[str]:
    """Coerce a 1D list/array/Series of labels, checking any Series index."""
    if isinstance(values, pd.Series):
        # If not aligned by index, a default RangeIndex means that the caller
        # explicitly supplies predictions in evaluation-row order.
        n = len(expected_index)
        if not (values.index.equals(expected_index) or
                values.index.equals(pd.RangeIndex(n))):
            raise ValueError(
                f"{name} is a pandas Series with a different, non-sequential "
                "index from data. Align it to data first or use a positional "
                "list / numpy array in EXACT validation-row order."
            )

    items = list(values)
    if len(items) != len(expected_index):
        raise ValueError(
            f"{name} contains {len(items)} labels, but data contains "
            f"{len(expected_index)} rows. They must match exactly."
        )

    result = []
    for pos, item in enumerate(items, start=1):
        if isinstance(item, (list, tuple, dict)) or getattr(item, "ndim", 0) > 0:
            raise ValueError(
                f"{name} item {pos} isn't a single class label. "
                "Pass predicted specialty labels, not probability vectors."
            )
        if pd.isna(item) or not str(item).strip():
            raise ValueError(f"{name} contains an empty/missing label at row {pos}.")
        result.append(str(item).strip())
    return result


def _safe_model_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._-")
    if not cleaned:
        raise ValueError("Model name must include at least one letter or number.")
    return cleaned


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _make_excerpt(row: pd.Series, max_chars: int) -> str:
    # Prefer the short clinical description, with a transcription snippet for
    # additional context. Don't silently claim the excerpt is the full note.
    description = _text(row.get("description", ""))
    transcription = _text(row.get("transcription", ""))
    sample_name = _text(row.get("sample_name", ""))
    keywords = _text(row.get("keywords", ""))
    parts: list[str] = []
    if description:
        parts.append("Description: " + description)
    if transcription:
        parts.append("Transcription: " + transcription)
    if not parts and sample_name:
        parts.append("Sample: " + sample_name)
    if not parts and keywords:
        parts.append("Keywords: " + keywords)
    excerpt = " | ".join(parts)
    return excerpt[: max_chars - 3].rstrip() + "..." if len(excerpt) > max_chars else excerpt


def _format_markdown(errors: pd.DataFrame, model_name: str, accuracy: float,
                     total_count: int, markdown_limit: int | None) -> str:
    n_errors = len(errors)
    shown = errors if markdown_limit is None else errors.head(markdown_limit)
    lines = [
        f"# Misclassified cases: {model_name}",
        "",
        f"- Evaluated labeled records: **{total_count}**",
        f"- Misclassified records: **{n_errors}**",
        f"- Accuracy on this evaluation set: **{accuracy:.2%}**",
        "- `evaluation_row` is 1-based within the supplied evaluation CSV or DataFrame.",
        "- `dataframe_index` preserves the original pandas row index, if available.",
        "- Review the original fields in the accompanying CSV before selecting paper examples.",
        "",
    ]
    if not n_errors:
        lines.extend(["No misclassified rows were found.", ""])
        return "\n".join(lines)

    for n, (_, row) in enumerate(shown.iterrows(), start=1):
        lines.extend([
            f"## Mistake {n} (evaluation row {row['evaluation_row']})",
            "",
            f"**Correct label:** {row['correct_label']}",
            "",
            f"**Predicted label:** {row['predicted_label']}",
            "",
            "**Clinical text excerpt:**",
            "",
        ])
        excerpt = str(row["text_excerpt"]).replace("\n", " ")
        # Prefix each line to retain a readable Markdown quotation.
        lines.append("> " + (excerpt or "[No clinical text available]") )
        lines.extend([
            "",
            "**Interpretation:** [Write your analysis here: why might the model "
            "have confused these specialties? Is the note ambiguous, does the "
            "correct label appear questionable, or is important context missing?]",
            "",
            "---",
            "",
        ])
    if markdown_limit is not None and n_errors > markdown_limit:
        lines.append(
            f"Only the first {markdown_limit} mistakes are shown here. "
            "ALL mistakes are included in the accompanying CSV."
        )
        lines.append("")
    return "\n".join(lines)


def save_model_errors(
    model_name: str,
    data: pd.DataFrame,
    y_pred: Iterable[Any],
    y_true: Iterable[Any] | None = None,
    output_dir: str | Path = "predictions",
    excerpt_chars: int = 900,
    markdown_limit: int | None = None,
) -> dict[str, Any]:
    """Save all model errors and human-readable examples for one labeled split.

    Args:
        model_name: Unique name (e.g. 'svm_tfidf', 'lstm', 'bert').
        data: Original validation/test DataFrame with the clinical text fields.
              It may contain 'medical_specialty', in which case y_true is optional.
        y_pred: Sequence of predicted class names, in EXACTLY data row order.
        y_true: Correct class names, in EXACTLY data row order. If omitted,
                use data['medical_specialty'].
        output_dir: Where to save the files. Default: predictions/.
        excerpt_chars: Maximum clinical excerpt length (>= 100).
        markdown_limit: Maximum mistakes included in Markdown; None means all.

    Returns:
        Dict with output paths and accuracy / number of errors.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if excerpt_chars < 100:
        raise ValueError("excerpt_chars must be at least 100.")
    if markdown_limit is not None and markdown_limit < 1:
        raise ValueError("markdown_limit must be positive or None.")
    if not len(data):
        raise ValueError("Cannot evaluate an empty DataFrame.")
    if y_true is None:
        if "medical_specialty" not in data.columns:
            raise ValueError(
                "Ground-truth labels missing! Supply y_true or provide the "
                "medical_specialty column in data. You cannot evaluate "
                "test_no_labels.csv for misclassifications."
            )
        y_true = data["medical_specialty"]

    gold = _as_aligned_list(y_true, data.index, "y_true")
    predicted = _as_aligned_list(y_pred, data.index, "y_pred")

    name = _safe_model_name(model_name)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    evaluated = data.copy()
    evaluated.insert(0, "evaluation_row", range(1, len(data) + 1))
    evaluated.insert(1, "dataframe_index", list(data.index))
    # Explicitly avoid accidental pandas index-based alignment.
    evaluated["correct_label"] = gold
    evaluated["predicted_label"] = predicted
    evaluated["is_misclassified"] = [a != b for a, b in zip(gold, predicted)]
    evaluated["text_excerpt"] = [
        _make_excerpt(data.iloc[i], excerpt_chars) for i in range(len(data))
    ]
    evaluated["interpretation"] = ""  # Must be written by the researchers.

    errors = evaluated.loc[evaluated["is_misclassified"]].copy()
    total = len(evaluated)
    n_errors = len(errors)
    accuracy = (total - n_errors) / total

    all_file = dest / f"{name}_all_predictions.csv"
    errors_file = dest / f"{name}_misclassified.csv"
    markdown_file = dest / f"{name}_misclassified.md"
    confusion_file = dest / f"{name}_confusion_matrix.csv"

    evaluated.to_csv(all_file, index=False, encoding="utf-8-sig")
    errors.to_csv(errors_file, index=False, encoding="utf-8-sig")
    markdown_file.write_text(
        _format_markdown(errors, name, accuracy, total, markdown_limit),
        encoding="utf-8",
    )
    confusion = pd.crosstab(
        pd.Series(gold, name="correct_label"),
        pd.Series(predicted, name="predicted_label"),
        dropna=False,
    )
    confusion.to_csv(confusion_file, encoding="utf-8-sig")

    result = {
        "model": name,
        "total": total,
        "errors": n_errors,
        "accuracy": accuracy,
        "all_predictions": all_file,
        "misclassified_csv": errors_file,
        "misclassified_md": markdown_file,
        "confusion_matrix": confusion_file,
    }
    print(
        f"[{name}] {n_errors}/{total} mistakes, accuracy={accuracy:.2%}\n"
        f"  CSV: {errors_file}\n"
        f"  Readable report: {markdown_file}\n"
        f"  Confusion matrix: {confusion_file}\n"
    )
    return result


def _load_label_file(filename: str | Path) -> list[str]:
    """Read a one-label-per-line .txt or labeled .csv prediction file."""
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Label file not found: {filename}")
    if filename.suffix.lower() in (".txt", ".labels"):
        # splitlines does NOT discard blank lines between entries.
        return filename.read_text(encoding="utf-8-sig").splitlines()
    if filename.suffix.lower() == ".csv":
        frame = pd.read_csv(filename, keep_default_na=False, dtype=str)
        for column in KNOWN_LABEL_COLUMNS:
            if column in frame.columns:
                return frame[column].tolist()
        raise ValueError(
            f"{filename}: no recognized prediction column. Use one of "
            f"{', '.join(KNOWN_LABEL_COLUMNS)}, or save labels as a "
            "one-label-per-line .txt (no header)."
        )
    raise ValueError(f"Unsupported file type: {filename} (use .txt or .csv)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and export all wrong model predictions on a LABELED validation set."
    )
    parser.add_argument(
        "--data", required=True,
        help="Original labeled validation/test CSV, in the same order as the predictions.",
    )
    parser.add_argument(
        "--pred", action="append", required=True, metavar="MODEL=FILE",
        help="Model name and prediction file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Optional file of correct labels if --data lacks medical_specialty.",
    )
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--excerpt-chars", type=int, default=900)
    parser.add_argument(
        "--markdown-limit", type=int, default=None,
        help="Only include first N errors in Markdown (CSV still includes all).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    if args.labels is None and "medical_specialty" not in df.columns:
        parser.error(
            "--data must contain the true 'medical_specialty' column, "
            "or use --labels. test_no_labels.csv cannot be used to find errors."
        )
    gold = _load_label_file(args.labels) if args.labels else df["medical_specialty"]
    names_seen: set[str] = set()
    for spec in args.pred:
        if "=" not in spec:
            parser.error(f"Expected MODEL=FILE after --pred, got {spec!r}.")
        model, path = spec.split("=", 1)
        normalized = _safe_model_name(model)
        if normalized in names_seen:
            parser.error(f"Duplicate model name: {normalized}")
        names_seen.add(normalized)
        predicted = _load_label_file(path)
        save_model_errors(
            model_name=normalized,
            data=df,
            y_true=gold,
            y_pred=predicted,
            output_dir=args.output_dir,
            excerpt_chars=args.excerpt_chars,
            markdown_limit=args.markdown_limit,
        )


if __name__ == "__main__":
    main()
