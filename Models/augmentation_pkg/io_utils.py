"""Dataset, configuration and file IO without third-party dependencies."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

FIELDS = ("medical_specialty", "description", "sample_name", "transcription", "keywords")
PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
DEFAULT_CONFIG = MODELS_DIR / "augmentation_config.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not 0 < float(config["validation_fraction"]) < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 <= float(config["random_dropout"]["probability"]) <= 1:
        raise ValueError("random_dropout.probability must be between 0 and 1")
    if int(config["selection"]["max_source_rows"]) < 0:
        raise ValueError("selection.max_source_rows must be >= 0 (0 means unlimited)")
    if config["source_column"] not in ("description", "transcription"):
        raise ValueError("source_column must be description or transcription")
    return config


def discover_train_file(explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training CSV not found: {path}")
        return path
    candidates = [
        MODELS_DIR / "train.csv",
        MODELS_DIR.parent / "train.csv",
        MODELS_DIR / "data" / "train.csv",
        MODELS_DIR.parent / "data" / "train.csv",
        MODELS_DIR.parent / "StudentsPack" / "train.csv",
    ]
    present = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(present) == 1:
        return present[0]
    if not present:
        raise FileNotFoundError(
            "Could not find train.csv. Run from Models with "
            "'python augment_data.py split --input ../train.csv' "
            "or provide its actual path."
        )
    raise RuntimeError(
        "Found multiple train.csv files; specify --input explicitly:\n" +
        "\n".join(str(p) for p in present)
    )


def read_records(path: str | Path, *, require_label: bool = True) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = set(FIELDS if require_label else FIELDS[1:])
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
        rows: list[dict[str, str]] = []
        for i, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"Malformed CSV row {i}: more fields than header")
            row = {field: (raw.get(field) or "") for field in FIELDS if field in raw}
            if require_label and not row["medical_specialty"].strip():
                raise ValueError(f"Missing medical_specialty on CSV row {i}")
            rows.append(row)
    return rows


def write_records(path: str | Path, rows: list[dict[str, str]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=target.parent,
                                      prefix=f".{target.name}.", suffix=".tmp", delete=False)
    try:
        with tmp as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in FIELDS})
        os.replace(tmp.name, target)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def write_table(path: str | Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_row_ids(rows: list[dict[str, str]]) -> list[str]:
    """Stable IDs including a repeat index for identical records."""
    seen: Counter[str] = Counter()
    ids = []
    for row in rows:
        fingerprint = sha256_text(json.dumps([row.get(f, "") for f in FIELDS], ensure_ascii=False))[:20]
        seen[fingerprint] += 1
        ids.append(f"{fingerprint}-{seen[fingerprint]}")
    return ids


def group_key(row: dict[str, str]) -> str:
    """Group exact normalized clinical notes, regardless of class label."""
    def normalized(text: str) -> str:
        return " ".join(text.casefold().split())
    transcription = normalized(row.get("transcription", ""))
    if transcription:
        return "t:" + sha256_text(transcription)
    return "d:" + sha256_text("|".join((normalized(row.get("description", "")),
                                        normalized(row.get("sample_name", "")))))


def report_class_distribution(directory: Path, datasets: dict[str, list[dict[str, str]]]) -> None:
    classes = sorted(set().union(*(set(x["medical_specialty"] for x in rows) for rows in datasets.values())))
    counts = {name: Counter(x["medical_specialty"] for x in rows) for name, rows in datasets.items()}
    write_table(directory / "class_distribution.csv", ["specialty", *datasets.keys()], [
        {"specialty": cls, **{name: counts[name][cls] for name in datasets}} for cls in classes
    ])
