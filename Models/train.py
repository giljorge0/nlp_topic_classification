# Same preprocessing, training, evaluation and inference for 3 RNN models
# Input with semicolon-separated train.csv with header, test_no_labels.csv without header
#
# HANDOFF MODE (opt-in): `--handoff-dir model_handoff --dataset <name>` trains/predicts from the data
# pipeline's ready-made, pre-split CSVs (see MODEL_TRAINING_HANDOFF.md). Without --handoff-dir this file
# behaves exactly as before (raw train.csv, own split, max_length 512, `--class-weighting none|sqrt_inv`).


from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from birnn import BiRNN
from bilstm import BiLSTM
from bilstm_attention import BiLSTMAttention


LABELS = (
    "Cardiovascular-Pulmonary",
    "Dermatology",
    "Gastroenterology",
    "General Medicine",
    "Neurology",
    "Neurosurgery",
    "Obstetrics-Gynecology",
    "Ophthalmology",
    "Orthopedic",
    "Psychiatry-Psychology",
    "Radiology",
    "Surgery",
)
FIELDS = ("description", "sample_name", "transcription", "keywords")
MODELS = {"birnn": BiRNN, "bilstm": BiLSTM, "bilstm_attention": BiLSTMAttention}
TOKEN_REGEX = re.compile(r"\b\w+(?:[-']\w+)*\b", re.UNICODE)
PAD, UNK = "<pad>", "<unk>"

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


def device_from_arg(value: str) -> torch.device:
    if value != "auto":
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but MPS is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_labelled_csv(path: str) -> tuple[list[dict], int]:
    """Skips invalid target labels, including the malformed lines in supplied train.csv"""
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        expected = {"medical_specialty", *FIELDS}
        if not reader.fieldnames or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(f"Expected semicolon CSV header with columns: {sorted(expected)}")
        rows, skipped = [], 0
        for line_number, row in enumerate(reader, start=2):
            if row.get("medical_specialty", "").strip() not in LABELS:
                skipped += 1
                continue
            cleaned = {field: (row.get(field) or "").strip() for field in FIELDS}
            cleaned["medical_specialty"] = row["medical_specialty"].strip()
            cleaned["row_number"] = line_number
            rows.append(cleaned)
    if not rows:
        raise ValueError("No labelled rows found")
    return rows, skipped


def read_unlabelled_csv(path: str) -> list[dict]:
    """Official test CSV has FOUR columns and NO header; preserve exact row order"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row_number, cells in enumerate(csv.reader(stream, delimiter=";"), start=1):
            if len(cells) != 4:
                raise ValueError(f"Test row {row_number}: expected 4 columns, got {len(cells)}")
            item = dict(zip(FIELDS, (x.strip() for x in cells)))
            item["row_number"] = row_number
            rows.append(item)
    if not rows:
        raise ValueError("Test CSV is empty")
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
    parts = []
    for field in fields:
        text = (row.get(field) or "").strip()
        if field == "keywords" and text and strip_keyword_prefix:
            first, separator, remaining = text.partition(",")
            if separator and normalize_label(first) in LABEL_PREFIXES:
                text = remaining.strip()
        if text:
            parts.append(f"{field.replace('_', ' ')}: {text}")
    return " ".join(parts)


def tokenize(text: str) -> list[str]:
    """Deliberately simple baseline tokenizer: lowercase; keep words and numbers"""
    return TOKEN_REGEX.findall(text.casefold())


def build_vocabulary(texts: list[str], max_vocab: int,
                     min_freq: int, max_length: int) -> dict[str, int]:
    counts = Counter(
        token for text in texts for token in tokenize(text)[:max_length]
    )
    vocab = {PAD: 0, UNK: 1}
    for token, frequency in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if frequency < min_freq or len(vocab) >= max_vocab:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    sequence = [vocab.get(token, 1) for token in tokenize(text)[:max_length]]
    return sequence if sequence else [1]  # RNN packing requires length >= 1


class MedicalDataset(Dataset):
    def __init__(self, rows: list[dict], vocab: dict[str, int], max_length: int,
                 fields: list[str], strip_keyword_prefix: bool, labelled: bool):
        self.rows = rows
        self.tokens = [
            torch.tensor(encode_text(make_text(row, fields, strip_keyword_prefix),
                                     vocab, max_length), dtype=torch.long)
            for row in rows
        ]
        self.targets = (
            [LABELS.index(row["medical_specialty"]) for row in rows]
            if labelled else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.tokens[index], (self.targets[index] if self.targets is not None else -1), index


def collate_examples(examples):
    tokens, targets, positions = zip(*examples)
    lengths = torch.tensor([len(seq) for seq in tokens], dtype=torch.long)
    padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(targets, dtype=torch.long), list(positions)


def make_loader(dataset: MedicalDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_examples, num_workers=0)


def split_rows(rows: list[dict], split: str, val_fraction: float,
               seed: int) -> tuple[list[dict], list[dict]]:
    labels = np.array([LABELS.index(row["medical_specialty"]) for row in rows])
    indices = np.arange(len(rows))
    if split == "stratified":
        training, validation = train_test_split(
            indices, test_size=val_fraction, stratify=labels, random_state=seed
        )
    else:
        # An exact duplicate clinical transcription never occurs on both sides
        # For empty transcriptions, group by the other input fields
        def group_id(row):
            basis = row["transcription"].strip().casefold()
            if not basis:
                basis = "||".join(row[f] for f in FIELDS).casefold()
            return hashlib.sha1(basis.encode("utf-8")).hexdigest()

        groups = [group_id(row) for row in rows]
        splitter = GroupShuffleSplit(n_splits=64, test_size=val_fraction,
                                    random_state=seed)
        total_counts = np.bincount(labels, minlength=len(LABELS))
        total_rates = total_counts / len(labels)
        candidates = []
        for t, v in splitter.split(indices, labels, groups):
            tcounts = np.bincount(labels[t], minlength=len(LABELS))
            vcounts = np.bincount(labels[v], minlength=len(LABELS))
            covers_all = np.all(tcounts > 0) and np.all(vcounts > 0)
            v_rates = vcounts / len(v)
            size_error = abs(len(v) / len(rows) - val_fraction)
            distribution_error = float(np.abs(v_rates - total_rates).sum())
            score = distribution_error + size_error
            candidates.append((not covers_all, score, t, v))
        candidates.sort(key=lambda result: (result[0], result[1]))
        _, _, training, validation = candidates[0]
        missing = [LABELS[k] for k in range(len(LABELS))
                   if k not in labels[validation]]
        if missing:
            print(f"WARNING: validation split lacks classes: {missing}")
    return [rows[int(i)] for i in training], [rows[int(i)] for i in validation]


def model_hyperparameters(args: argparse.Namespace, vocab_size: int) -> dict:
    return dict(vocab_size=vocab_size, embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim, num_classes=len(LABELS),
                num_layers=args.num_layers, dropout=args.dropout, padding_idx=0)


def training_loss(args: argparse.Namespace, rows: list[dict], device: torch.device):
    resolved = getattr(args, "resolved_class_weights", None)      # hand-off mode: vector aligned to LABELS by class name
    if resolved is not None:
        return nn.CrossEntropyLoss(weight=torch.tensor(resolved, dtype=torch.float32, device=device))
    if args.class_weighting == "none":
        return nn.CrossEntropyLoss()
    counts = np.bincount([LABELS.index(r["medical_specialty"]) for r in rows],
                         minlength=len(LABELS))
    # Tempered inverse-frequency weights to reduce excessive minority oversampling
    weights = np.sqrt(np.max(counts) / np.maximum(counts, 1))
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def train_epoch(model, loader, optimizer, criterion, device, grad_clip: float) -> float:
    model.train()
    total_loss, total_items = 0.0, 0
    for padded, lengths, targets, _ in loader:
        padded, targets = padded.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(padded, lengths)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(targets)
        total_items += len(targets)
    return total_loss / max(1, total_items)


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


def score_predictions(actual, predictions) -> dict:
    ids = list(range(len(LABELS)))
    report = classification_report(actual, predictions, labels=ids,
                                   target_names=LABELS, zero_division=0, output_dict=True)
    per_class_f1 = {label: float(report[label]["f1-score"]) for label in LABELS}
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(f1_score(actual, predictions, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predictions, labels=ids, average="weighted", zero_division=0)),
        "min_class_f1": min(per_class_f1.values()),
        "per_class_f1": per_class_f1,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predictions, labels=ids).tolist(),
    }


def write_analysis(output_dir: Path, scores: dict, rows: list[dict],
                   predictions: list[int], actual: list[int], positions: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as out:
        json.dump(scores, out, indent=2, ensure_ascii=False)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, values in zip(LABELS, scores["confusion_matrix"]):
            writer.writerow([label, *values])
    with (output_dir / "misclassified.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["csv_row_number", "actual", "predicted", "description", "sample_name", "text_excerpt"])
        for guessed, truth, position in zip(predictions, actual, positions):
            if guessed != truth:
                row = rows[position]
                excerpt = row["transcription"][:350].replace("\n", " ")
                writer.writerow([row["row_number"], LABELS[truth], LABELS[guessed],
                                 row["description"], row["sample_name"], excerpt])


def save_model(path: Path, model: nn.Module, args, vocab: dict[str, int],
               kwargs: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": args.model,
        "model_state": model.state_dict(),
        "model_kwargs": kwargs,
        "vocab": vocab,
        "labels": list(LABELS),
        "fields": args.fields,
        "strip_keyword_prefix": args.strip_keyword_prefix,
        "max_length": args.max_length,
        "seed": args.seed,
        "extra": extra or {},
    }, path)


def prepare_training(args, training_rows):
    texts = [make_text(row, args.fields, args.strip_keyword_prefix) for row in training_rows]
    vocab = build_vocabulary(texts, args.max_vocab, args.min_freq, args.max_length)
    kwargs = model_hyperparameters(args, len(vocab))
    model = MODELS[args.model](**kwargs)
    print(f"Model={args.model} | Vocab={len(vocab)} | Parameters={sum(p.numel() for p in model.parameters()):,}")
    return vocab, kwargs, model


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
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break
    with (outdir / "history.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (outdir / "split_rows.json").open("w", encoding="utf-8") as out:
        json.dump({"train_csv_rows": [r["row_number"] for r in training_rows],
                   "validation_csv_rows": [r["row_number"] for r in validation_rows]}, out)
    print(f"Best epoch={best_epoch}; saved {checkpoint}")
    print(f"Validation metrics and error examples saved in {outdir}")


def train_final_command(args):
    """Train with ALL valid labelled rows after development model/epoch selection"""
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
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device, args.grad_clip)
        print(f"Final epoch {epoch:02d} | train_loss={loss:.4f}")
    path = Path(args.output_dir) / "final.pt"
    save_model(path, model, args, vocab, kwargs,
               {"final_training": True, "epochs": args.epochs,
                **({"input_mode": "handoff_text", "handoff_dataset": args.dataset, "handoff_split": args.split,
                    "handoff_variant": data["variant"], "class_weight": args.class_weight} if data else {})})
    print(f"Saved {path}")


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["labels"] != list(LABELS):
        raise ValueError("Checkpoint label order does not match this code version")
    model = MODELS[checkpoint["model_name"]](**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


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
    )
    scores = score_predictions(truths, guesses)
    print(f"Evaluated {len(rows)} rows; skipped {skipped} invalid targets")
    print(f"Accuracy={scores['accuracy']:.4f}, Macro-F1={scores['macro_f1']:.4f}, "
          f"Minimum class F1={scores['min_class_f1']:.4f}")
    if args.output_dir:
        write_analysis(Path(args.output_dir), scores, rows, guesses, truths, positions)
    else:
        for label, f1 in scores["per_class_f1"].items():
            print(f"  {label}: F1={f1:.3f}")


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
        p.add_argument("--epochs", type=int, default=12)
        p.add_argument("--lr", type=float, default=0.001)
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

    final = commands.add_parser("train-final", help="Refit on all labelled training rows")
    add_shared_training_flags(final)
    final.set_defaults(run=train_final_command)

    predict = commands.add_parser("predict", help="Predict the official HEADERLESS test file")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--test-csv", default=None, help="raw-mode checkpoints")
    add_handoff_flags(predict, False)
    predict.add_argument("--split", choices=["stratified", "group_aware"], default=None, help="default: the split stored in the checkpoint")
    predict.add_argument("--output", default="results.txt")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(run=predict_command)

    evaluate = commands.add_parser("evaluate", help="Evaluate on a separate LABELLED CSV")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-csv", required=True)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(run=evaluate_command)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = resolve_args(get_args())
    if parsed.command in {"train", "train-final"}:
        for name in ("max_length", "max_vocab", "min_freq", "batch_size", "epochs"):
            if getattr(parsed, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if parsed.max_vocab < 2:
            raise ValueError("--max-vocab must be at least 2")
    parsed.run(parsed)
