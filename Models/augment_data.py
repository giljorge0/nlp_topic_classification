#!/usr/bin/env python3
"""Help the Doctor: local augmentation plus import/export for ChatGPT paraphrases.

Run from within Models using `python augment_data.py --help`.
No external connections or model/API keys are used by this code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from augmentation_pkg.io_utils import (
    FIELDS, MODELS_DIR, detect_csv_delimiter, discover_train_file, load_config, read_records,
    report_class_distribution, sha256_text, stable_row_ids, write_records,
    write_table,
)
from augmentation_pkg.medical_validation import validate_paraphrase
from augmentation_pkg.random_dropout import random_word_dropout
from augmentation_pkg.splitting import choose_source_indices, grouped_stratified_split

SPLITS = MODELS_DIR / "data" / "splits"
AUGMENTED = MODELS_DIR / "data" / "augmented"
REQUESTS = MODELS_DIR / "data" / "chat_requests"
REPORTS = MODELS_DIR / "reports" / "augmentation"
AUDIT_COLUMNS = (
    "source_row_id", "copy_index", "specialty", "method", "target_column",
    "status", "reason", "original", "augmented", "details", "attempts",
)


def suffix(tag: str) -> str:
    if tag and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", tag):
        raise ValueError("--tag must contain only letters, digits, _ or - (max 40 characters)")
    return f"_{tag}" if tag else ""


def base_path(train_input: str | None) -> Path:
    return Path(train_input).expanduser().resolve() if train_input else SPLITS / "train_base.csv"


def read_base(train_input: str | None) -> list[dict[str, str]]:
    path = base_path(train_input)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {path}. Run split or pass --train-input PATH."
        )
    rows = read_records(path)
    if not rows:
        raise ValueError(f"Training file is empty: {path}")
    return rows


def base_fingerprint(rows: list[dict[str, str]]) -> str:
    payload = [[r.get(f, "") for f in FIELDS] for r in rows]
    return sha256_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def distribution(base: list[dict[str, str]] | None = None) -> None:
    datasets = {}
    if base is not None:
        datasets["train_base"] = base
    elif (SPLITS / "train_base.csv").is_file():
        datasets["train_base"] = read_records(SPLITS / "train_base.csv")
    if (SPLITS / "validation.csv").is_file() and base is None:
        datasets["validation"] = read_records(SPLITS / "validation.csv")
    for method in ("rwd", "llm", "combined"):
        path = AUGMENTED / f"train_{method}.csv"
        if not path.is_file():
            continue
        rows = read_records(path)
        if "train_base" in datasets and rows[:len(datasets["train_base"])] == datasets["train_base"]:
            datasets[f"train_{method}"] = rows
    if datasets:
        report_class_distribution(REPORTS, datasets)


def selected_sources(base: list[dict[str, str]], config: dict,
                     limit: int | None = None) -> list[int]:
    ids = choose_source_indices(base, config["selection"], seed=int(config["seed"]),
                                override_limit=limit, text_field=config["source_column"])
    if not ids:
        print("WARNING: No source rows selected. Check selection strategy and text availability.")
    return ids


def write_augmented(method: str, base: list[dict[str, str]],
                    additions: list[dict[str, str]], audit: list[dict],
                    settings: dict, tag: str, delimiter: str = ",") -> None:
    suf = suffix(tag)
    path = AUGMENTED / f"train_{method}{suf}.csv"
    write_records(path, base + additions, delimiter=delimiter)
    write_table(REPORTS / f"{method}{suf}_audit.csv", AUDIT_COLUMNS, audit)
    write_table(REPORTS / f"{method}{suf}_rejected.csv", AUDIT_COLUMNS,
                [r for r in audit if r["status"] != "accepted"])
    info = {
        "method": method,
        "base_fingerprint": base_fingerprint(base),
        "original_rows": len(base),
        "accepted_synthetic_rows": len(additions),
        "augmented_rows": len(base) + len(additions),
        "accepted_by_specialty": dict(sorted(Counter(r["medical_specialty"] for r in additions).items())),
        "settings": settings,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{method}{suf}_summary.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if tag:
        report_class_distribution(REPORTS / f"distribution_{tag}",
                                  {"train_base": base, f"train_{method}": base + additions})
    else:
        distribution(base)
    print(f"{method.upper()}: {len(base)} originals + {len(additions)} accepted synthetic rows")
    print(f"Saved {path}")
    print(f"Logs: {REPORTS / (method + suf + '_audit.csv')}")


def cmd_split(args: argparse.Namespace, config: dict) -> None:
    path = discover_train_file(args.input)
    rows = read_records(path)
    # A CSV can parse to five fields yet still be corrupted by a spreadsheet
    # re-export: merged cases show up as whole clinical notes in the label.
    suspicious = [n for n, row in enumerate(rows, 2)
                  if len(row["medical_specialty"]) > 100
                  or len(row["description"]) > 2000]
    classes = {row["medical_specialty"] for row in rows}
    if len(classes) != 12 or suspicious:
        print(f"WARNING: Data-quality check found {len(classes)} distinct labels "
              f"(the project expects 12) and {len(suspicious)} suspicious rows. "
              "Possible CSV corruption. Use an untouched original export before training.",
              file=sys.stderr)
        if suspicious:
            print(f"WARNING: Suspicious logical CSV row numbers: {suspicious[:30]}",
                  file=sys.stderr)
    if (SPLITS / "train_base.csv").exists() and not args.force:
        raise ValueError("train_base.csv already exists. To avoid overwriting an existing split, "
                         "use --force only if you intend to recreate it.")
    train, val, warnings = grouped_stratified_split(
        rows, fraction=float(config["validation_fraction"]), seed=int(config["seed"])
    )
    delimiter = detect_csv_delimiter(path)
    write_records(SPLITS / "train_base.csv", train, delimiter=delimiter)
    write_records(SPLITS / "validation.csv", val, delimiter=delimiter)
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_path": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_fingerprint": base_fingerprint(train),
        "seed": config["seed"], "validation_fraction": config["validation_fraction"],
        "original_count": len(rows), "train_count": len(train), "validation_count": len(val),
        "warnings": warnings,
    }
    (REPORTS / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    distribution()
    print(f"Split {len(rows)} rows into {len(train)} training and {len(val)} validation rows.")
    print(f"Outputs: {SPLITS / 'train_base.csv'} and {SPLITS / 'validation.csv'}")
    for w in warnings:
        print("WARNING:", w)
    if args.force:
        print("WARNING: Old augmentation files, if any, may now be stale. Regenerate them.")


def cmd_rwd(args: argparse.Namespace, config: dict) -> None:
    base = read_base(args.train_input)
    source_indices = selected_sources(base, config, args.limit)
    ids = stable_row_ids(base)
    column = config["source_column"]
    params = config["random_dropout"]
    copies = int(params.get("copies_per_original", 1))
    if copies < 1:
        raise ValueError("random_dropout.copies_per_original must be >= 1")
    additions: list[dict[str, str]] = []
    audit: list[dict] = []
    seen = set()
    for idx in source_indices:
        src = base[idx]
        for n in range(copies):
            seed_text = f"{config['seed']}:{ids[idx]}:{n}:rwd"
            rng = random.Random(int(sha256_text(seed_text)[:16], 16))
            result = random_word_dropout(
                src[column], float(params["probability"]), rng,
                context=src, mode=params.get("mode", "safe"),
                max_attempts=int(params.get("max_attempts_per_copy", 20)),
            )
            unique_key = (ids[idx], result.text)
            accepted = result.success and unique_key not in seen
            if accepted:
                seen.add(unique_key)
                clone = dict(src)
                clone[column] = result.text
                additions.append(clone)
            audit.append({
                "source_row_id": ids[idx], "copy_index": n,
                "specialty": src["medical_specialty"], "method": "rwd",
                "target_column": column, "status": "accepted" if accepted else "rejected",
                "reason": "" if accepted else (result.reason or "duplicate_variant"),
                "original": src[column], "augmented": result.text if result.success else "",
                "details": ", ".join(result.deleted), "attempts": result.attempts,
            })
    write_augmented("rwd", base, additions, audit, {
        "selected_sources": len(source_indices), "requested": len(source_indices) * copies,
        "source_column": column, "random_dropout": params,
    }, args.tag, delimiter=detect_csv_delimiter(base_path(args.train_input)))


def _request_paths(tag: str) -> tuple[Path, Path]:
    suf = suffix(tag)
    return (REQUESTS / f"paraphrase_requests{suf}.csv",
            REQUESTS / f"request_manifest{suf}.json")


def cmd_chat_export(args: argparse.Namespace, config: dict) -> None:
    base = read_base(args.train_input)
    selection = selected_sources(base, config, args.limit)
    ids = stable_row_ids(base)
    column = config["source_column"]
    csv_path, manifest_path = _request_paths(args.tag)
    requests = [
        {"source_row_id": ids[i], "description": base[i][column]} for i in selection
    ]
    if not requests:
        raise ValueError("No eligible descriptions to export")
    write_table(csv_path, ["source_row_id", "description"], requests)
    manifest = {
        "base_fingerprint": base_fingerprint(base), "source_column": column,
        "row_count": len(base), "request_count": len(requests),
        "request_hashes": {r["source_row_id"]: sha256_text(r["description"]) for r in requests},
        "selection": config["selection"], "seed": config["seed"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(requests)} descriptions for ChatGPT: {csv_path}")
    print(f"Local verification manifest: {manifest_path}")
    print("Upload the requests CSV in ChatGPT. Ask for source_row_id,paraphrase CSV as output.")
    print("The export contains neither labels nor full transcriptions.")


def _read_paraphrase_input(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Paraphrases file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=detect_csv_delimiter(path, {"source_row_id", "paraphrase"}))
        if not reader.fieldnames or not {"source_row_id", "paraphrase"}.issubset(reader.fieldnames):
            raise ValueError("LLM response must be a CSV with headers: source_row_id,paraphrase")
        entries = []
        for n, record in enumerate(reader, 2):
            if None in record:
                raise ValueError(f"Malformed CSV line {n} in {path}")
            entries.append({k: (record.get(k) or "") for k in ("source_row_id", "paraphrase")})
    if not entries:
        raise ValueError("Paraphrases CSV contains no rows")
    return entries


def cmd_chat_import(args: argparse.Namespace, config: dict) -> None:
    base = read_base(args.train_input)
    _, manifest_path = _request_paths(args.tag)
    if not manifest_path.is_file():
        raise FileNotFoundError("No request manifest for this tag. Run chat-export first with "
                                "the same --tag and --train-input.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["base_fingerprint"] != base_fingerprint(base):
        raise ValueError("Training data changed since chat-export; refusing to misalign paraphrases")
    if manifest["source_column"] != config["source_column"]:
        raise ValueError("source_column changed after export; use the original configuration")
    ids = stable_row_ids(base)
    indexed = dict(zip(ids, base))
    allowed = manifest["request_hashes"]
    data = _read_paraphrase_input(Path(args.input).expanduser().resolve())
    seen_in_file = set()
    for item in data:
        ident = item["source_row_id"].strip()
        if not ident or ident not in allowed or ident not in indexed:
            raise ValueError(f"Unknown or non-exported source_row_id: {ident!r}. "
                             "Use the response to the matching paraphrase_requests CSV.")
        if sha256_text(indexed[ident][config["source_column"]]) != allowed[ident]:
            raise ValueError(f"Source text changed for row {ident}")
        if ident in seen_in_file:
            raise ValueError(f"Duplicate source_row_id in input file: {ident}")
        seen_in_file.add(ident)

    suf = suffix(args.tag)
    output_path = AUGMENTED / f"train_llm{suf}.csv"
    audit_path = REPORTS / f"llm{suf}_audit.csv"
    additions: list[dict[str, str]] = []
    audit: list[dict] = []
    if args.append:
        if not output_path.is_file() or not audit_path.is_file():
            raise FileNotFoundError("--append requires an existing chat-import output for this tag")
        prior = read_records(output_path)
        if prior[:len(base)] != base:
            raise ValueError("Existing augmented CSV does not belong to this training partition")
        additions = prior[len(base):]
        with audit_path.open("r", encoding="utf-8-sig", newline="") as f:
            audit = list(csv.DictReader(f))
    used_ids = {a["source_row_id"] for a in audit if a["status"] == "accepted"}
    column = config["source_column"]
    bounds = config.get("chat_paraphrasing", {})
    accepted_this_run = 0
    for entry in data:
        ident = entry["source_row_id"].strip()
        src = indexed[ident]
        candidate = entry["paraphrase"].strip()
        if ident in used_ids:
            accepted, reason = False, "source_id_already_accepted"
        else:
            review = validate_paraphrase(
                src[column], candidate,
                min_ratio=float(bounds.get("min_length_ratio", 0.5)),
                max_ratio=float(bounds.get("max_length_ratio", 2.0)),
            )
            accepted, reason = review.accepted, ";".join(review.reasons)
        if accepted:
            clone = dict(src)
            clone[column] = candidate
            additions.append(clone)
            used_ids.add(ident)
            accepted_this_run += 1
        audit.append({
            "source_row_id": ident, "copy_index": 0,
            "specialty": src["medical_specialty"], "method": "llm",
            "target_column": column, "status": "accepted" if accepted else "rejected",
            "reason": reason, "original": src[column], "augmented": candidate,
            "details": "ChatGPT-produced CSV, manually review clinical accuracy",
            "attempts": 1,
        })
    write_augmented("llm", base, additions, audit, {
        "source_column": column, "imported_rows_this_run": len(data),
        "accepted_this_run": accepted_this_run, "append": args.append,
        "automatic_checks_only": True,
    }, args.tag, delimiter=detect_csv_delimiter(base_path(args.train_input)))
    rejected = len(data) - accepted_this_run
    print(f"This import: {accepted_this_run} accepted, {rejected} rejected.")
    if rejected:
        print(f"Inspect {REPORTS / ('llm' + suf + '_rejected.csv')}")
    if accepted_this_run:
        print("IMPORTANT: Review paraphrases manually. Rules do not prove clinical equivalence.")


def cmd_combined(args: argparse.Namespace, _config: dict) -> None:
    base = read_base(args.train_input)
    suf = suffix(args.tag)
    additions: list[dict[str, str]] = []
    seen = set()
    for method in ("rwd", "llm"):
        path = AUGMENTED / f"train_{method}{suf}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Generate the {method} dataset first.")
        rows = read_records(path)
        if rows[:len(base)] != base:
            raise ValueError(f"Wrong training partition in {path.name}; regenerate it")
        for extra in rows[len(base):]:
            key = tuple(extra.get(field, "") for field in FIELDS)
            if key not in seen:
                seen.add(key)
                additions.append(extra)
    out = AUGMENTED / f"train_combined{suf}.csv"
    write_records(out, base + additions, delimiter=detect_csv_delimiter(base_path(args.train_input)))
    if not args.tag:
        distribution(base)
    print(f"Combined: {len(base)} original + {len(additions)} unique synthetic examples")
    print(f"Saved: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline medical augmentation + ChatGPT CSV integration")
    parser.add_argument("--config", help="Override Models/augmentation_config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="Prepare training/validation split once")
    split.add_argument("--input", help="Path to original labeled train.csv")
    split.add_argument("--force", action="store_true", help="Overwrite existing split intentionally")

    rwd = sub.add_parser("rwd", help="Generate Random Word Dropout examples locally")
    rwd.add_argument("--train-input", help="Use this labeled training CSV instead of train_base.csv")
    rwd.add_argument("--limit", type=int, help="Override max source rows (0 = unlimited)")
    rwd.add_argument("--tag", default="", help="Dataset suffix, e.g. final")

    export = sub.add_parser("chat-export", help="Export safe columns to upload for ChatGPT paraphrasing")
    export.add_argument("--train-input", help="Use this labeled training CSV instead of train_base.csv")
    export.add_argument("--limit", type=int, help="Maximum source rows exported (0 = unlimited)")
    export.add_argument("--tag", default="", help="Dataset suffix, e.g. final")

    imp = sub.add_parser("chat-import", help="Validate and merge ChatGPT-generated CSV locally")
    imp.add_argument("--input", required=True, help="CSV returned by ChatGPT: source_row_id,paraphrase")
    imp.add_argument("--train-input", help="Same labeled training CSV used for chat-export")
    imp.add_argument("--tag", default="", help="Must match the chat-export tag")
    imp.add_argument("--append", action="store_true", help="Accumulate multiple ChatGPT response batches")

    combined = sub.add_parser("combined", help="Join RWD and ChatGPT augmented datasets")
    combined.add_argument("--train-input", help="Same labeled training CSV used for rwd and chat-import")
    combined.add_argument("--tag", default="", help="Must match all input dataset tags")

    args = parser.parse_args(argv)
    try:
        conf = load_config(args.config)
        {"split": cmd_split, "rwd": cmd_rwd, "chat-export": cmd_chat_export,
         "chat-import": cmd_chat_import, "combined": cmd_combined}[args.command](args, conf)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
