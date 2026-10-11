# Same preprocessing, training, evaluation and inference for 3 RNN models
# Input with semicolon-separated train.csv with header, test_no_labels.csv without header


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


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


LABEL_PREFIXES = {normalize_label(label) for label in LABELS}


def make_text(row: dict, fields: list[str] | tuple[str, ...],
              strip_keyword_prefix: bool = False) -> str:
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
def run_inference(model, loader, device, criterion=None):
    model.eval()
    predictions, actual, positions = [], [], []
    loss_sum, examples = 0.0, 0
    for padded, lengths, targets, row_positions in loader:
        logits = model(padded.to(device), lengths)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        actual.extend(targets.tolist())
        positions.extend(row_positions)
        if criterion is not None:
            loss_sum += criterion(logits, targets.to(device)).item() * len(targets)
            examples += len(targets)
    return predictions, actual, positions, (loss_sum / examples if examples else None)


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
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    set_seed(args.seed)
    device = device_from_arg(args.device)
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
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        guesses, truths, positions, valid_loss = run_inference(model, valid_loader, device, criterion)
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
                        "select_metric": args.select_metric})
            write_analysis(outdir, metrics, validation_rows, guesses, truths, positions)
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
               {"final_training": True, "epochs": args.epochs})
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
    rows = read_unlabelled_csv(args.test_csv)
    dataset = MedicalDataset(rows, checkpoint["vocab"], checkpoint["max_length"],
                             checkpoint["fields"], checkpoint["strip_keyword_prefix"],
                             labelled=False)
    loader = make_loader(dataset, args.batch_size, shuffle=False)
    predictions, _, positions, _ = run_inference(model, loader, device)
    if positions != list(range(len(rows))):
        raise RuntimeError("Unexpected row reordering during prediction")
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8", newline="\n") as out:
        for prediction in predictions:
            out.write(LABELS[prediction] + "\n")
    print(f"Saved {len(predictions)} predictions in original row order: {outpath}")
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


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_shared_training_flags(p):
        p.add_argument("--train-csv", required=True, help="train.csv with 5-column header")
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--fields", nargs="+", choices=FIELDS, default=list(FIELDS),
                       help="Concatenate selected fields in the given order")
        p.add_argument("--strip-keyword-prefix", action="store_true",
                       help="Remove the first keyword ONLY when it names a specialty")
        p.add_argument("--embedding-dim", type=int, default=128)
        p.add_argument("--hidden-dim", type=int, default=128)
        p.add_argument("--num-layers", type=int, default=1)
        p.add_argument("--dropout", type=float, default=0.3)
        p.add_argument("--max-length", type=int, default=512)
        p.add_argument("--max-vocab", type=int, default=20000)
        p.add_argument("--min-freq", type=int, default=2)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--epochs", type=int, default=12)
        p.add_argument("--lr", type=float, default=0.001)
        p.add_argument("--weight-decay", type=float, default=0.0)
        p.add_argument("--grad-clip", type=float, default=1.0)
        p.add_argument("--class-weighting", choices=["none", "sqrt_inv"], default="none")
        p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps...")
        p.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train with a labelled validation split")
    add_shared_training_flags(train)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--split", choices=["grouped", "stratified"], default="grouped")
    train.add_argument("--select-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    train.add_argument("--patience", type=int, default=4)
    train.set_defaults(run=train_command)

    final = commands.add_parser("train-final", help="Refit on all labelled training rows")
    add_shared_training_flags(final)
    final.set_defaults(run=train_final_command)

    predict = commands.add_parser("predict", help="Predict the official HEADERLESS test file")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--test-csv", required=True)
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
    parsed = get_args()
    if parsed.command in {"train", "train-final"}:
        for name in ("max_length", "max_vocab", "min_freq", "batch_size", "epochs"):
            if getattr(parsed, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if parsed.max_vocab < 2:
            raise ValueError("--max-vocab must be at least 2")
    parsed.run(parsed)
