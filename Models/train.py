# Same preprocessing, training, evaluation and inference for 3 RNN models
# Input with semicolon-separated train.csv with header, test_no_labels.csv without header
#
# HANDOFF MODE (opt-in): `--handoff-dir model_handoff --dataset <name>` trains/predicts from the data
# pipeline's ready-made, pre-split CSVs (see MODEL_TRAINING_HANDOFF.md). Without --handoff-dir this file
# behaves exactly as before (raw train.csv, own split, max_length 512, `--class-weighting none|sqrt_inv`).

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

# ---- hand-off mode constants -------------------------------------------------------------
HANDOFF_VARIANTS = ("baseline_all_fields", "clinical_text", "compressed_unique_text", "diagnostic_original_keywords")
HANDOFF_AUGMENTATIONS = {"_word_dropout": "word_dropout", "_sentence_dropout": "sentence_dropout"}
HANDOFF_SPLITS = ("stratified", "group_aware")
HANDOFF_DEFAULT_MAX_LENGTH, RAW_DEFAULT_MAX_LENGTH = 1024, 512
SPLIT_MANIFESTS = {"stratified": "splits/split_manifest_primary_stratified.csv",
                   "group_aware": "splits/split_manifest_optional_group_aware.csv"}
ROOT_SPLIT_NAME = {"stratified": "primary", "group_aware": "group_aware"}   # value recorded in handoff_manifest.json
CLASS_WEIGHT_CHOICES = ("none", "inverse_sqrt", "inverse_frequency", "interpolated")


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


# =============================== HANDOFF MODE: data loading ===============================
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dataset_name(name: str) -> tuple[str, str | None]:
    """'baseline_all_fields_word_dropout' -> ('baseline_all_fields', 'word_dropout')."""
    for suffix, augmentation in HANDOFF_AUGMENTATIONS.items():
        if name.endswith(suffix):
            name, found = name[: -len(suffix)], augmentation
            break
    else:
        found = None
    if name not in HANDOFF_VARIANTS:
        raise ValueError(f"unknown --dataset variant {name!r}; choose from {list(HANDOFF_VARIANTS)} "
                         f"(optionally + {list(HANDOFF_AUGMENTATIONS)})")
    return name, found


def resolve_root(handoff_dir: str, split: str, group_aware_dir: str | None) -> Path:
    """--handoff-dir is the PRIMARY (stratified) hand-off. The group-aware hand-off is a different folder
    (default: sibling '<name>_group_aware', or --group-aware-dir); the two are never mixed."""
    primary = Path(handoff_dir)
    if split == "stratified":
        root = primary
    else:
        root = Path(group_aware_dir) if group_aware_dir else primary.with_name(primary.name.rstrip("/\\") + "_group_aware")
    if not root.is_dir():
        raise FileNotFoundError(f"hand-off folder for split '{split}' not found: {root}"
                                + ("  (unzip model_handoff_extras.zip next to model_handoff, or pass --group-aware-dir)" if split == "group_aware" else ""))
    return root


def load_root_manifest(root: Path, split: str) -> dict:
    path = root / "handoff_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: not a model_handoff folder")
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    built_with = manifest.get("configuration", {}).get("args", {}).get("split", "primary")
    if built_with != ROOT_SPLIT_NAME[split]:
        raise ValueError(f"{root} was built from the '{built_with}' split but --split {split} was requested. "
                         "Stratified and group-aware files are never mixed: point at the matching folder.")
    return manifest


def verify_against_manifest(root: Path, manifest: dict, paths: list[Path]) -> dict[str, dict]:
    """sha256 of every file used. Files listed in handoff_manifest.json must match; others are recorded as unlisted."""
    out = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required hand-off file missing: {path}")
        digest = sha256_file(path)
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = None
        listed = manifest["files"].get(rel) if rel else None
        if listed is not None and listed["sha256"] != digest:
            raise ValueError(f"checksum mismatch for {path}: the file differs from handoff_manifest.json (modified or corrupted)")
        out[str(path)] = {"sha256": digest, "in_manifest": listed is not None}
    return out


def read_handoff_csv(path: str | Path, labelled: bool = True) -> list[dict]:
    """Read a hand-off CSV (COMMA separated, QUOTE_ALL, `text` column = ready-made model input). File order is kept."""
    path = Path(path)
    need = {"example_id", "text"} | ({"source_record_id", "medical_specialty", "partition", "augmentation_type"} if labelled
                                     else {"test_record_id", "original_order"})
    rows = []
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = need - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for record in reader:
            item = {"_text": record["text"], "example_id": record["example_id"], "row_number": record["example_id"],
                    "description": "", "sample_name": "", "transcription": record["text"][:350]}
            if labelled:
                if record["medical_specialty"] not in LABELS:
                    raise ValueError(f"{path}: unexpected label {record['medical_specialty']!r}")
                item.update(medical_specialty=record["medical_specialty"], source_record_id=record["source_record_id"],
                            partition=record["partition"], augmentation_type=record["augmentation_type"])
            else:
                item.update(test_record_id=record["test_record_id"], source_record_id=record["test_record_id"],
                            original_order=int(record["original_order"]))
            rows.append(item)
    if not rows:
        raise ValueError(f"{path} is empty")
    ids = [r["example_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicated example_id values")
    return rows


def _order_problem(got: list, expected: list, what: str) -> str | None:
    if got == expected:
        return None
    if sorted(got) == sorted(expected):
        first = next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)
        return f"{what}: same records but NOT in the manifest order (first difference at row {first}: {got[first]} vs expected {expected[first]}). Shuffled or re-sorted files are rejected"
    return f"{what}: records differ from the split manifest ({len(got)} rows vs {len(expected)} expected)"


def validate_handoff_split(root: Path, split: str, train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict] | None) -> None:
    """Rows must be EXACTLY the split manifest's records in the manifest's order; test must be the verified original order."""
    with open(root / SPLIT_MANIFESTS[split], encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    expected = {part: [(m["source_record_id"], m["medical_specialty"]) for m in manifest if m["partition"] == part] for part in ("train", "validation")}
    problems = []
    if any(r["partition"] != "validation" or r["augmentation_type"] != "none" for r in validation_rows):
        problems.append("validation file contains non-validation or augmented rows (validation is never augmented)")
    problems.append(_order_problem([(r["source_record_id"], r["medical_specialty"]) for r in validation_rows], expected["validation"], "validation.csv"))
    originals = [r for r in train_rows if r["augmentation_type"] == "none"]
    copies = [r for r in train_rows if r["augmentation_type"] != "none"]
    problems.append(_order_problem([(r["source_record_id"], r["medical_specialty"]) for r in originals], expected["train"], "train rows (originals)"))
    train_ids = {sid for sid, _ in expected["train"]}
    if any(r["partition"] != "train" for r in train_rows):
        problems.append("training file contains rows whose partition is not 'train'")
    if any(r["source_record_id"] not in train_ids for r in copies):
        problems.append("an augmented row maps to a source record outside the TRAIN partition")
    if test_rows is not None:
        with open(root / "audit" / "cleaned_test_source_fields.csv", encoding="utf-8", newline="") as stream:
            expected_test = [r["test_record_id"] for r in csv.DictReader(stream)]
        problems.append(_order_problem([r["test_record_id"] for r in test_rows], expected_test, "test.csv"))
        if [r["original_order"] for r in test_rows] != list(range(len(test_rows))):
            problems.append("test.csv: original_order is not 0..N-1")
    problems = [p for p in problems if p]
    if problems:
        raise ValueError("hand-off data validation failed:\n  - " + "\n  - ".join(problems))


def find_train_file(root: Path, variant: str, augmentation: str | None, extras_dir: str | None) -> Path:
    folder = root / "datasets" / variant
    if augmentation is None:
        return folder / "train.csv"
    name = f"train_augmented_{augmentation}.csv"
    candidates = [folder / name]
    if extras_dir:
        extras = Path(extras_dir)
        candidates += [extras / root.name / "datasets" / variant / name, extras / "datasets" / variant / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"augmented training file not found; looked in {[str(c) for c in candidates]}. "
                            "Unzip model_handoff_extras.zip next to model_handoff (or pass --extras-dir).")


def load_handoff_training_data(args: argparse.Namespace) -> dict:
    variant, augmentation = parse_dataset_name(args.dataset)
    root = resolve_root(args.handoff_dir, args.split, args.group_aware_dir)
    manifest = load_root_manifest(root, args.split)
    folder = root / "datasets" / variant
    train_file = find_train_file(root, variant, augmentation, args.extras_dir)
    files = verify_against_manifest(root, manifest, [train_file, folder / "validation.csv", folder / "test.csv", folder / "train.csv"])
    train_rows = read_handoff_csv(train_file)
    validation_rows = read_handoff_csv(folder / "validation.csv")
    test_rows = read_handoff_csv(folder / "test.csv", labelled=False)
    validate_handoff_split(root, args.split, train_rows, validation_rows, test_rows)
    if augmentation:      # originals inside the augmented file must be the plain train.csv rows, unchanged
        base = {r["example_id"]: r["_text"] for r in read_handoff_csv(folder / "train.csv")}
        if {r["example_id"]: r["_text"] for r in train_rows if r["augmentation_type"] == "none"} != base:
            raise ValueError(f"{train_file}: its original rows differ from {folder / 'train.csv'}")
        if not any(r["augmentation_type"] != "none" for r in train_rows):
            raise ValueError(f"{train_file} contains no augmented rows")
    return {"root": root, "variant": variant, "augmentation": augmentation, "train_file": train_file, "train": train_rows,
            "validation": validation_rows, "test": test_rows, "files": files}


def load_handoff_test(root: Path, variant: str, split: str) -> tuple[list[dict], dict]:
    manifest = load_root_manifest(root, split)
    test_file = root / "datasets" / variant / "test.csv"
    files = verify_against_manifest(root, manifest, [test_file])
    rows = read_handoff_csv(test_file, labelled=False)
    with open(root / "audit" / "cleaned_test_source_fields.csv", encoding="utf-8", newline="") as stream:
        expected = [r["test_record_id"] for r in csv.DictReader(stream)]
    problem = _order_problem([r["test_record_id"] for r in rows], expected, "test.csv")
    if problem or [r["original_order"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"hand-off test validation failed: {problem or 'original_order is not 0..N-1'}")
    return rows, files


# =============================== HANDOFF MODE: class weights ===============================
def weight_key(choice: str, alpha: float | None) -> str:
    if choice == "interpolated":
        if alpha is None or not 0.0 <= alpha <= 1.0:
            raise ValueError("--class-weight interpolated needs --class-weight-alpha in [0, 1]")
        return f"interpolated_alpha_{alpha:g}"
    return {"none": "none", "inverse_sqrt": "inverse_sqrt_frequency", "inverse_frequency": "inverse_frequency"}[choice]


def check_weights(vector: np.ndarray) -> np.ndarray:
    if vector.shape != (len(LABELS),) or not np.all(np.isfinite(vector)) or np.any(vector <= 0) or abs(vector.mean() - 1.0) > 1e-6:
        raise ValueError("class weights must be finite, positive, one per class and have mean 1")
    return vector


def load_handoff_class_weights(root: Path, key: str) -> np.ndarray:
    """Weights in THIS script's label order, aligned by class NAME through weights/label_to_index.json (never by position).
    Cross-checked against class_weights_primary_split_wide.csv. (File name says 'primary' but it holds the weights of the
    split the folder was built from; the group-aware folder has its own.)"""
    wdir = root / "weights"
    with open(wdir / "label_to_index.json", encoding="utf-8") as stream:
        label_to_index = json.load(stream)
    if set(label_to_index) != set(LABELS) or sorted(label_to_index.values()) != list(range(len(LABELS))):
        raise ValueError("weights/label_to_index.json does not describe exactly the 12 expected classes")
    with open(wdir / "class_weights_primary_split.json", encoding="utf-8") as stream:
        table = json.load(stream)
    if key not in table:
        raise ValueError(f"weight strategy {key!r} not available in {wdir}; available: {sorted(table)}")
    handoff_order = np.zeros(len(LABELS))
    for name, index in label_to_index.items():
        handoff_order[index] = table[key][name]
    vector = np.zeros(len(LABELS))
    for name, index in label_to_index.items():
        vector[LABELS.index(name)] = handoff_order[index]
    with open(wdir / "class_weights_primary_split_wide.csv", encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            name = record["medical_specialty"]
            if int(record["class_index"]) != label_to_index[name] or abs(float(record[key]) - vector[LABELS.index(name)]) > 1e-9:
                raise ValueError(f"weights/*.json and *_wide.csv disagree for class {name!r}")
    return check_weights(vector)


def compute_class_weights(counts: np.ndarray, choice: str, alpha: float | None) -> np.ndarray:
    """Same formulas as the data pipeline: freq=count/N (a proportion); every vector is normalised to mean 1."""
    if np.any(counts <= 0):
        raise ValueError(f"cannot weight classes with no training rows: {[LABELS[i] for i in np.where(counts <= 0)[0]]}")
    freq = counts / counts.sum()
    if choice == "none":
        raw = np.ones(len(LABELS))
    elif choice == "inverse_frequency":
        raw = 1.0 / freq
    elif choice == "inverse_sqrt":
        raw = 1.0 / np.sqrt(freq)
    else:
        weight_key(choice, alpha)
        raw = (1.0 - alpha) + alpha / freq
    return check_weights(raw / raw.mean())


def resolve_handoff_class_weights(args: argparse.Namespace, root: Path, training_rows: list[dict]) -> np.ndarray | None:
    if args.class_weight == "none":
        return None
    key = weight_key(args.class_weight, args.class_weight_alpha)
    counts = np.bincount([LABELS.index(r["medical_specialty"]) for r in training_rows], minlength=len(LABELS))
    if args.class_weight_source == "train_rows":
        return compute_class_weights(counts.astype(float), args.class_weight, args.class_weight_alpha)
    if any(r["augmentation_type"] != "none" for r in training_rows):
        print("WARNING: the provided class weights were computed from the UNAUGMENTED train partition, but this training file "
              "contains augmented copies. Add --class-weight-source train_rows to recompute them from the rows actually used.")
    return load_handoff_class_weights(root, key)


def warn_if_keywords_truncated(rows: list[dict], max_length: int) -> None:
    """Keywords come LAST in the text: report how many training notes lose them at this max_length."""
    marked = [r["_text"] for r in rows if "[KEYWORDS]" in r["_text"]]
    if not marked:
        return
    cut = sum(1 for text in marked if len(tokenize(text[: text.index("[KEYWORDS]")])) >= max_length)
    share = 100.0 * cut / len(marked)
    if share >= 5.0:
        print(f"WARNING: with --max-length {max_length} the [KEYWORDS] section is cut off for {share:.0f}% of training notes "
              "(it is the last field). Use a larger --max-length if keywords matter, and keep it fixed across compared runs.")


def write_prediction_table(path: Path, rows: list[dict], predictions: list[int], confidence: list[float], positions: list[int],
                           actual: list[int] | None = None) -> None:
    """example_id, source_record_id, true_label (when available), predicted_label, prediction_probability - in file order."""
    order = sorted(range(len(positions)), key=lambda k: positions[k])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["example_id", "source_record_id", "true_label", "predicted_label", "prediction_probability"])
        for k in order:
            row = rows[positions[k]]
            writer.writerow([row["example_id"], row["source_record_id"], LABELS[actual[k]] if actual is not None else "",
                             LABELS[predictions[k]], f"{confidence[k]:.6f}"])


def write_run_config(outdir: Path, args: argparse.Namespace, data: dict, weights: np.ndarray | None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    config = {"mode": "handoff", "command": args.command, "dataset": args.dataset, "variant": data["variant"], "augmentation": data["augmentation"],
              "split": args.split, "handoff_root": str(data["root"]), "train_file": str(data["train_file"]),
              "max_length": args.max_length, "max_length_source": args.max_length_source, "model": args.model, "seed": args.seed,
              "class_weight": args.class_weight, "class_weight_alpha": args.class_weight_alpha, "class_weight_source": args.class_weight_source,
              "class_weights_by_name": ({LABELS[i]: float(w) for i, w in enumerate(weights)} if weights is not None else None),
              "rows": {"train_total": len(data["train"]), "train_original": sum(r["augmentation_type"] == "none" for r in data["train"]),
                       "train_augmented": sum(r["augmentation_type"] != "none" for r in data["train"]),
                       "validation": len(data["validation"]), "test": len(data["test"])},
              "files": data["files"],
              "hyperparameters": {k: getattr(args, k) for k in ("embedding_dim", "hidden_dim", "num_layers", "dropout", "max_vocab", "min_freq",
                                                                "batch_size", "epochs", "lr", "weight_decay", "grad_clip")}}
    with (outdir / "run_config.json").open("w", encoding="utf-8") as out:
        json.dump(config, out, indent=2)


def announce_handoff(args: argparse.Namespace, data: dict, weights: np.ndarray | None) -> None:
    print(f"HANDOFF | root={data['root']} | split={args.split} | dataset={args.dataset} | train file={data['train_file'].name} "
          f"| max_length={args.max_length} ({args.max_length_source}) | class_weight={args.class_weight}"
          + (f" (source={args.class_weight_source})" if args.class_weight != "none" else ""))
    if weights is not None:
        print("Class weights by name:", {label: round(float(w), 3) for label, w in zip(LABELS, weights)})
    if data["variant"] == "diagnostic_original_keywords":
        print("NOTE: diagnostic_original_keywords keeps the specialty name in keywords: a LEAKAGE investigation, not for final reporting.")
    warn_if_keywords_truncated(data["train"], args.max_length)


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


LABEL_PREFIXES = {normalize_label(label) for label in LABELS}


def make_text(row: dict, fields: list[str] | tuple[str, ...],
              strip_keyword_prefix: bool = False) -> str:
    if row.get("_text") is not None:      # hand-off mode: the data pipeline already built the model input
        return row["_text"]
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


def training_loss(args: argparse.Namespace, rows: list[dict], device: torch.device):
    resolved = getattr(args, "resolved_class_weights", None)      # hand-off mode: vector aligned to LABELS by class name
    if resolved is not None:
        return nn.CrossEntropyLoss(weight=torch.tensor(resolved, dtype=torch.float32, device=device))
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
def run_inference(model, loader, device, criterion=None, return_probabilities: bool = False):
    model.eval()
    predictions, actual, positions, confidence = [], [], [], []
    loss_sum, examples = 0.0, 0
    for padded, lengths, targets, row_positions in loader:
        logits = model(padded.to(device), lengths)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        if return_probabilities:
            confidence.extend(torch.softmax(logits, dim=1).max(dim=1).values.cpu().tolist())
        actual.extend(targets.tolist())
        positions.extend(row_positions)
        if criterion is not None:
            loss_sum += criterion(logits, targets.to(device)).item() * len(targets)
            examples += len(targets)
    result = predictions, actual, positions, (loss_sum / examples if examples else None)
    return (*result, confidence) if return_probabilities else result
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


def train_command(args):
    if args.mode == "raw" and not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    set_seed(args.seed)
    device = device_from_arg(args.device)
    data = None
    if args.mode == "handoff":
        data = load_handoff_training_data(args)
        training_rows, validation_rows = data["train"], data["validation"]
        rows = [r for r in training_rows if r["augmentation_type"] == "none"] + validation_rows
        skipped = 0
        args.resolved_class_weights = resolve_handoff_class_weights(args, data["root"], training_rows)
        announce_handoff(args, data, args.resolved_class_weights)
        write_run_config(Path(args.output_dir), args, data, args.resolved_class_weights)
    else:
        rows, skipped = read_labelled_csv(args.train_csv)
        training_rows, validation_rows = split_rows(rows, args.split, args.val_fraction, args.seed)
    print(f"Device={device} | Valid={len(rows)} | Invalid labels skipped={skipped} "
          f"| Train={len(training_rows)} | Validation={len(validation_rows)} | Split={args.split}")
    vocab, kwargs, model = prepare_training(args, training_rows)
    model = model.to(device)
    train_data = MedicalDataset(training_rows, vocab, args.max_length,
                                args.fields, args.strip_keyword_prefix, labelled=True)
    valid_data = MedicalDataset(validation_rows, vocab, args.max_length,
                                args.fields, args.strip_keyword_prefix, labelled=True)
    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_data, args.batch_size, shuffle=False)
    criterion = training_loss(args, training_rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale = -1.0, 0, 0
    history = []
    checkpoint = outdir / "best.pt"
    extra_info = ({"input_mode": "handoff_text", "handoff_dataset": args.dataset, "handoff_split": args.split,
                   "handoff_variant": data["variant"], "class_weight": args.class_weight} if data else {})
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        guesses, truths, positions, valid_loss, confidence = run_inference(model, valid_loader, device, criterion, return_probabilities=True)
        metrics = score_predictions(truths, guesses)
        score = metrics[args.select_metric]
        history.append({"epoch": epoch, "train_loss": loss, "validation_loss": valid_loss,
                        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                        "min_class_f1": metrics["min_class_f1"]})
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | val_loss={valid_loss:.4f} | "
              f"accuracy={metrics['accuracy']:.3f} | macro_f1={metrics['macro_f1']:.3f} "
              f"| min_class_f1={metrics['min_class_f1']:.3f}")
        if score > best_score + 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            save_model(checkpoint, model, args, vocab, kwargs,
                       {"best_epoch": best_epoch, "best_metric": score,
                        "select_metric": args.select_metric, **extra_info})
            write_analysis(outdir, metrics, validation_rows, guesses, truths, positions)
            if data:
                write_prediction_table(outdir / "validation_predictions.csv", validation_rows, guesses, confidence, positions, truths)
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
    device = device_from_arg(args.device)
    data = None
    if args.mode == "handoff":
        # hand-off rows are already cleaned/traceable. Final fit = training file + the (unaugmented) validation file.
        data = load_handoff_training_data(args)
        rows, skipped = data["train"] + data["validation"], 0
        args.resolved_class_weights = resolve_handoff_class_weights(args, data["root"], rows)
        announce_handoff(args, data, args.resolved_class_weights)
        print("NOTE: train-final uses the VALIDATION rows for training; there is no held-out score for this model.")
        write_run_config(Path(args.output_dir), args, data, args.resolved_class_weights)
    else:
        rows, skipped = read_labelled_csv(args.train_csv)
    print(f"Final training: valid={len(rows)}, skipped={skipped}; no validation or official test used")
    vocab, kwargs, model = prepare_training(args, rows)
    model = model.to(device)
    dataset = MedicalDataset(rows, vocab, args.max_length,
                             args.fields, args.strip_keyword_prefix, labelled=True)
    loader = make_loader(dataset, args.batch_size, shuffle=True)
    criterion = training_loss(args, rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    save_model(path, model, args, vocab, kwargs,
               {"final_training": True, "epochs": args.epochs,
                **({"input_mode": "handoff_text", "handoff_dataset": args.dataset, "handoff_split": args.split,
                    "handoff_variant": data["variant"], "class_weight": args.class_weight} if data else {})})
    save_checkpoint(path, model, args, vocab, kwargs, {"final_fit": True, "epochs": args.epochs})
    print(f"Saved {path}")


def predict_command(args):
    device = device_from_arg(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    extra = checkpoint.get("extra", {})
    handoff = extra.get("input_mode") == "handoff_text"
    table_path = None
    if handoff:
        if args.test_csv:
            raise ValueError("this checkpoint was trained in hand-off mode: use --handoff-dir (not --test-csv)")
        if not args.handoff_dir:
            raise ValueError("this checkpoint was trained in hand-off mode: pass --handoff-dir (the same folder family used for training)")
        split = args.split or extra["handoff_split"]
        dataset = args.dataset or extra["handoff_dataset"]
        if (split, dataset) != (extra["handoff_split"], extra["handoff_dataset"]):
            raise ValueError(f"checkpoint was trained on split={extra['handoff_split']} dataset={extra['handoff_dataset']}; "
                             f"got split={split} dataset={dataset}. Text formats differ between datasets: use the matching ones")
        root = resolve_root(args.handoff_dir, split, args.group_aware_dir)
        rows, _ = load_handoff_test(root, parse_dataset_name(dataset)[0], split)
    else:
        if args.handoff_dir or args.dataset or args.split:
            raise ValueError("this checkpoint was trained on raw fields: use --test-csv (hand-off flags are not valid)")
        if not args.test_csv:
            raise ValueError("--test-csv is required for checkpoints trained on raw fields")
        rows = read_unlabelled_csv(args.test_csv)
    dataset = MedicalDataset(rows, checkpoint["vocab"], checkpoint["max_length"],
                             checkpoint["fields"], checkpoint["strip_keyword_prefix"],
                             labelled=False)
    loader = make_loader(dataset, args.batch_size, shuffle=False)
    predictions, _, positions, _, confidence = run_inference(model, loader, device, return_probabilities=True)
    if positions != list(range(len(rows))):
        raise RuntimeError("Unexpected row reordering during prediction")
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8", newline="\n") as out:
        for prediction in predictions:
            out.write(LABELS[prediction] + "\n")
    print(f"Saved {len(predictions)} predictions in original row order: {outpath}")
    if handoff:
        table_path = outpath.with_name("test_predictions.csv")
        write_prediction_table(table_path, rows, predictions, confidence, positions)
        print(f"Saved per-row predictions with IDs and probabilities: {table_path}")
    print("Predicted class counts:", dict(Counter(LABELS[p] for p in predictions)))


def evaluate_command(args):
    """Evaluate a saved model on a SEPARATE labelled holdout CSV"""
    device = device_from_arg(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    rows, skipped = read_labelled_csv(args.data_csv)
    dataset = MedicalDataset(rows, checkpoint["vocab"], checkpoint["max_length"],
                             checkpoint["fields"], checkpoint["strip_keyword_prefix"],
                             labelled=True)
    guesses, truths, positions, _ = run_inference(
        model, make_loader(dataset, args.batch_size, False), device
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


def resolve_args(a: argparse.Namespace) -> argparse.Namespace:
    """Decide raw vs hand-off mode and apply mode-specific defaults. Raw mode keeps every original default."""
    if a.command == "predict":
        return a
    a.mode = "handoff" if a.handoff_dir else "raw"
    handoff_only = {"--dataset": a.dataset, "--group-aware-dir": a.group_aware_dir, "--extras-dir": a.extras_dir,
                    "--class-weight": a.class_weight, "--class-weight-alpha": a.class_weight_alpha}
    if a.mode == "raw":
        used = [flag for flag, value in handoff_only.items() if value is not None]
        if used:
            raise ValueError(f"{', '.join(used)} only work with --handoff-dir")
        if a.class_weight_source != "handoff_file":
            raise ValueError("--class-weight-source only works with --handoff-dir")
        if not a.train_csv:
            raise ValueError("--train-csv is required unless --handoff-dir is given")
        if a.split == "group_aware":
            raise ValueError("--split group_aware needs --handoff-dir (raw mode has 'grouped' and 'stratified')")
        if a.command == "train":
            a.split = a.split or "grouped"
            a.val_fraction = 0.2 if a.val_fraction is None else a.val_fraction
        elif a.split is not None:
            raise ValueError("--split is not used by train-final in raw mode")
        a.fields = list(FIELDS) if a.fields is None else a.fields
        a.max_length_source = "explicit" if a.max_length is not None else "raw default"
        a.max_length = RAW_DEFAULT_MAX_LENGTH if a.max_length is None else a.max_length
        return a
    # ---- hand-off mode
    if not a.dataset:
        raise ValueError("--dataset is required with --handoff-dir (e.g. --dataset baseline_all_fields)")
    ignored = {"--train-csv": a.train_csv, "--fields": a.fields, "--strip-keyword-prefix": a.strip_keyword_prefix or None,
               "--val-fraction": getattr(a, "val_fraction", None)}
    used = [flag for flag, value in ignored.items() if value]
    if used:
        raise ValueError(f"{', '.join(used)} cannot be combined with --handoff-dir: the dataset, fields and split come from the hand-off files")
    if a.class_weighting != "none":
        raise ValueError("in hand-off mode use --class-weight (none | inverse_sqrt | inverse_frequency | interpolated), not --class-weighting")
    if a.split == "grouped":
        raise ValueError("--split grouped is the raw-mode splitter; in hand-off mode use --split stratified or --split group_aware")
    a.split = a.split or "stratified"
    a.class_weight = a.class_weight or "none"
    if a.class_weight == "interpolated":
        weight_key(a.class_weight, a.class_weight_alpha)
    elif a.class_weight_alpha is not None:
        raise ValueError("--class-weight-alpha is only used with --class-weight interpolated")
    parse_dataset_name(a.dataset)
    a.fields, a.strip_keyword_prefix = list(FIELDS), False
    a.max_length_source = "explicit" if a.max_length is not None else "handoff default"
    a.max_length = HANDOFF_DEFAULT_MAX_LENGTH if a.max_length is None else a.max_length
    return a


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_handoff_flags(p, with_training_options: bool):
        p.add_argument("--handoff-dir", default=None, help="model_handoff folder (the PRIMARY/stratified one). Enables hand-off mode")
        p.add_argument("--dataset", default=None,
                       help="baseline_all_fields | clinical_text | compressed_unique_text | diagnostic_original_keywords "
                            "(+ _word_dropout / _sentence_dropout for the augmented training file)")
        p.add_argument("--group-aware-dir", default=None, help="folder of the group-aware hand-off (default: sibling <handoff-dir>_group_aware)")
        if with_training_options:
            p.add_argument("--extras-dir", default=None, help="where model_handoff_extras.zip was unzipped, if its augmented files were not merged into model_handoff")
            p.add_argument("--class-weight", choices=CLASS_WEIGHT_CHOICES, default=None,
                           help="hand-off mode only. Weights come from <handoff>/weights/ aligned by class name")
            p.add_argument("--class-weight-alpha", type=float, default=None, help="alpha in [0,1] for --class-weight interpolated")
            p.add_argument("--class-weight-source", choices=["handoff_file", "train_rows"], default="handoff_file",
                           help="handoff_file = provided weights (computed from the UNAUGMENTED train partition); "
                                "train_rows = recompute from the rows actually used (recommended with augmented datasets and train-final)")

    def add_shared_training_flags(p):
        p.add_argument("--train-csv", default=None, help="raw mode: train.csv with 5-column header")
        add_handoff_flags(p, True)
        p.add_argument("--split", choices=["grouped", "stratified", "group_aware"], default=None,
                       help="raw mode: grouped (default) | stratified. hand-off mode: stratified (default) | group_aware")
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--fields", nargs="+", choices=FIELDS, default=None,
                       help="Concatenate selected fields in the given order (raw mode)")
        p.add_argument("--strip-keyword-prefix", action="store_true",
                       help="Remove the first keyword ONLY when it names a specialty (raw mode)")
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
        p.add_argument("--max-length", type=int, default=None,
                       help=f"tokens kept per note. Raw mode default {RAW_DEFAULT_MAX_LENGTH}; hand-off mode default {HANDOFF_DEFAULT_MAX_LENGTH} "
                            "(use 2048 for diagnostic runs). Keep it FIXED when comparing experiments")
        p.add_argument("--max-vocab", type=int, default=20000)
        p.add_argument("--min-freq", type=int, default=2)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--lr", type=float, default=.001)
        p.add_argument("--weight-decay", type=float, default=0.0)
        p.add_argument("--grad-clip", type=float, default=1.0)
        p.add_argument("--class-weighting", choices=["none", "sqrt_inv"], default="none", help="raw mode only")
        p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps...")
        p.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train with a labelled validation split")
    add_shared_training_flags(train)
    train.add_argument("--val-fraction", type=float, default=None, help="raw mode only (default 0.2); hand-off splits are fixed")
    train.add_argument("--select-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    train.add_argument("--patience", type=int, default=4)
    train.set_defaults(run=train_command)
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

    predict = commands.add_parser("predict", help="Predict the official HEADERLESS test file")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--test-csv", default=None, help="raw-mode checkpoints")
    add_handoff_flags(predict, False)
    predict.add_argument("--split", choices=["stratified", "group_aware"], default=None, help="default: the split stored in the checkpoint")
    predict.add_argument("--output", default="results.txt")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(run=predict_command)
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
    parsed = resolve_args(get_args())
    if parsed.command in {"train", "train-final"}:
        for name in ("max_length", "max_vocab", "min_freq", "batch_size", "epochs"):
            if getattr(parsed, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if parsed.max_vocab < 2:
            raise ValueError("--max-vocab must be at least 2")
    parsed = get_args()
    parsed.run(parsed)
