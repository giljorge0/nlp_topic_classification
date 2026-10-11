#!/usr/bin/env python3
"""preprocess_data.py -- conservative record reconstruction + controlled preprocessing variants.

Example:
    python preprocess_data.py --train train.csv --test test_no_labels.csv --output-dir processed_data --seed 42

* Never touches the raw inputs; writes only to --output-dir.
* Does NOT train, split, or predict anything.
* Augmentation is applied ONLY to a dataset whose role is 'train'; test (and any validation)
  data is refused, with NO override. Augmented variants also require --split-manifest.
* Variant name grammar:  <fields>__<normalization>__<representation>__<augmentation>
    fields:         all_fields | clinical_fields | transcription_only | description_transcription(alias) |
                    all_fields_no_specialty_keyword_prefix | all_fields_without_keywords
    normalization:  raw | lowercase | clinical
    representation: full | windows | head_tail
    augmentation:   no_aug | token_deletion | back_translation_es | synonym
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import logging
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

import clinical_common as cc

log = logging.getLogger("preprocess")

FIELD_SETS: dict[str, tuple[str, ...]] = {
    "all_fields": ("description", "sample_name", "transcription", "keywords"),
    "clinical_fields": ("description", "transcription"),
    "transcription_only": ("transcription",),
    "description_transcription": ("description", "transcription"),
    "all_fields_no_specialty_keyword_prefix": ("description", "sample_name", "transcription", "keywords"),
    "all_fields_without_keywords": ("description", "sample_name", "transcription"),
}
ALIASES = {"description_transcription": "clinical_fields"}  # identical content & markers -> not duplicated on disk
NORMALIZATIONS = ("raw", "lowercase", "clinical")
REPRESENTATIONS = ("full", "windows", "head_tail")
AUGMENTATIONS = ("no_aug", "token_deletion", "back_translation_es", "synonym")
DEFAULT_MARKERS = {"description": "[DESCRIPTION]", "sample_name": "[SAMPLE_NAME]", "transcription": "[TRANSCRIPTION]", "keywords": "[KEYWORDS]"}
DEFAULT_PROTECTED = ["no", "not", "without", "denies", "denied", "negative", "none", "never", "absent", "neither", "nor", "cannot", "deny", "non", "w/o"]
DEFAULT_VARIANTS = [
    "all_fields__raw__full__no_aug",
    "clinical_fields__lowercase__full__no_aug",
    "transcription_only__raw__full__no_aug",
    "all_fields_no_specialty_keyword_prefix__raw__full__no_aug",
    "all_fields_without_keywords__raw__full__no_aug",
    "all_fields_without_keywords__clinical__full__no_aug",
    "transcription_only__raw__windows__no_aug",
]  # augmented variants are deliberately NOT in the default set: augmentation needs --split-manifest (split first, then augment train only)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file whose keys (same names as the CLI options, '-'->'_') provide defaults")
    p.add_argument("--train", default="train.csv"); p.add_argument("--test", default="test_no_labels.csv")
    p.add_argument("--output-dir", default="processed_data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="*", default=None, help="explicit variant names (default: small built-in set). Use 'all-no-aug' for every field/normalization/full combination")
    p.add_argument("--markers-json", default=None, help='JSON {"description": "[DESCRIPTION]", ...} to override field markers')
    p.add_argument("--partial-policy", choices=["include", "quarantine"], default="include",
                   help="records recovered only partially (cut transcription / missing keywords): keep flagged in cleaned_train (default) or quarantine")
    p.add_argument("--window-size", type=int, default=256, help="window length in whitespace words of the transcription")
    p.add_argument("--window-overlap", type=int, default=64, help="overlap between consecutive windows, in words")
    p.add_argument("--window-field-policy", choices=["repeat_meta", "transcription_only"], default="repeat_meta",
                   help="repeat_meta: the non-transcription fields are repeated on every window (they are short); transcription_only: windows carry only transcription text")
    p.add_argument("--head-words", type=int, default=128); p.add_argument("--tail-words", type=int, default=128)
    p.add_argument("--deletion-prob", type=float, default=0.05); p.add_argument("--aug-copies", type=int, default=1, help="augmented copies per training document")
    p.add_argument("--protected-terms-file", default=None, help="text file, one protected token per line (replaces default list)")
    p.add_argument("--synonym-map", default=None, help="vetted JSON {phrase: replacement}; synonym augmentation is disabled without it")
    p.add_argument("--back-translation-backend", default="none", help="'none' or 'module:function' implementing translate(texts, src, tgt)->list[str]")
    p.add_argument("--i-confirm-translation-backend-approved", action="store_true",
                   help="REQUIRED to run a back-translation backend: project rules + privacy/terms checked by a human")
    p.add_argument("--review-sample-size", type=int, default=30)
    p.add_argument("--split-manifest", default=None, help="split manifest CSV (source_record_id, partition). REQUIRED for any augmented variant: only partition=='train' records are augmented")
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        cfg = json.loads(Path(pre.config).read_text(encoding="utf-8"))
        p.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2212": "-"})
_COMMA_GLUE = re.compile(r"(?<=[.:;]),(?=[A-Za-z])")


def normalize_text(s: str, mode: str) -> str:
    """raw: NFC + control-char removal + whitespace collapse.  lowercase: raw + casefold-free lower().
    clinical: NFKC, typographic quotes/dashes -> ASCII, doubled-quote CSV artefact '""' -> '"',
    one space inserted after a comma glued to a sentence end ('.,She' -> '., She'; the comma is kept), whitespace collapse.
    Numbers, units, punctuation, case and negations are never touched by any mode except lower()."""
    if mode not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization {mode!r}")
    if mode == "clinical":
        s = unicodedata.normalize("NFKC", s).translate(_QUOTE_MAP).replace('""', '"')
        s = _COMMA_GLUE.sub(", ", s)
    else:
        s = unicodedata.normalize("NFC", s)
    s = _CTRL.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s.lower() if mode == "lowercase" else s


def strip_keyword_prefix(keywords: str) -> tuple[str, str | None]:
    """Remove ONE leading specialty name (+ its comma) from keywords; returns (new_keywords, removed_or_None)."""
    m = cc.KEYWORD_PREFIX_RE.match(keywords)
    if not m:
        return keywords, None
    return keywords[m.end():], keywords[:m.end()].strip()


# --------------------------------------------------------------------------- #
# Datasets & guards
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    role: str                    # 'train' | 'validation' | 'test'
    frame: pd.DataFrame          # must hold doc id + the four input fields


class AugmentationRefused(PermissionError):
    pass


def assert_augmentable(ds: Dataset) -> None:
    """Augmentation is train-only. There is intentionally NO override (validation/test are never augmented)."""
    if ds.role != "train":
        raise AugmentationRefused(f"refusing to augment a '{ds.role}' dataset; augmentation is train-only and cannot be overridden")


def load_partition_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"source_record_id", "partition"} <= set(m.columns) or not set(m.partition) <= {"train", "validation"}:
        raise ValueError(f"{path}: need columns source_record_id, partition in {{train, validation}}")
    if m.source_record_id.duplicated().any():
        raise ValueError(f"{path}: duplicated source_record_id")
    return dict(zip(m.source_record_id, m.partition))


def build_text(fields: dict[str, str], field_names: Sequence[str], markers: dict[str, str],
               transform: Callable[[str, str], str] | None = None) -> str:
    parts = []
    for f in field_names:
        v = fields.get(f, "")
        if transform is not None:
            v = transform(f, v)
        parts.append(f"{markers[f]} {v}".rstrip())
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Windows / head-tail
# --------------------------------------------------------------------------- #
def make_windows(words: list[str], size: int, overlap: int) -> list[tuple[int, int, str]]:
    """[(start_word, end_word_exclusive, text)]. Covers every word; never discards text."""
    if size <= 0 or not (0 <= overlap < size):
        raise ValueError("need window_size > 0 and 0 <= overlap < window_size")
    if len(words) <= size:
        return [(0, len(words), " ".join(words))]
    stride, out, start = size - overlap, [], 0
    while True:
        end = min(start + size, len(words))
        out.append((start, end, " ".join(words[start:end])))
        if end >= len(words):
            return out
        start += stride


def head_tail(words: list[str], head: int, tail: int) -> tuple[str, int]:
    """Lossy baseline. Returns (text, omitted_words)."""
    if len(words) <= head + tail:
        return " ".join(words), 0
    return " ".join(words[:head]) + " [...] " + " ".join(words[-tail:]), len(words) - head - tail


# --------------------------------------------------------------------------- #
# Augmentation backends
# --------------------------------------------------------------------------- #
def load_protected(path: str | None) -> set[str]:
    if path:
        return {l.strip().lower() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()}
    return set(DEFAULT_PROTECTED)


_UNIT = re.compile(r"\d")


def is_protected(tok: str, protected: set[str]) -> bool:
    core = tok.strip(".,;:()[]\"'").lower()
    # raw notes glue sentences with '.,' and no space ('abscess.,no'), so a whitespace token may CONTAIN a negation:
    # protect the whole token if any word-part inside it is protected (conservative; also covers 'no-show', 'w/o')
    parts = [p for p in re.split(r"[^\w]+", tok.lower()) if p]
    return (core in protected or any(p in protected for p in parts) or bool(_UNIT.search(tok))
            or (tok.startswith("[") and tok.endswith("]")))


def token_deletion(text: str, p: float, rng: random.Random, protected: set[str]) -> str:
    """Delete each unprotected whitespace token with prob p. Never returns an empty string for non-empty input."""
    toks = text.split()
    kept = [t for t in toks if is_protected(t, protected) or rng.random() >= p]
    return " ".join(kept) if kept else text


def synonym_replace(text: str, mapping: dict[str, str], protected: set[str]) -> str:
    """Whole-phrase, case-sensitive replacement from a VETTED map; skips matches touching protected tokens/digits."""
    out = text
    for src, dst in mapping.items():
        pat = re.compile(r"(?<!\w)" + re.escape(src) + r"(?!\w)")
        def sub(m: re.Match) -> str:
            if any(is_protected(t, protected) for t in m.group(0).split()) or _UNIT.search(m.group(0)):
                return m.group(0)
            return dst
        out = pat.sub(sub, out)
    return out


class BackendUnavailable(RuntimeError):
    pass


class BackTranslator:
    """Adapter for English -> pivot -> English back-translation. Subclass or supply 'module:function'.
    Nothing is downloaded or called unless a backend is explicitly configured AND approved by a human."""
    name = "unconfigured"
    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        raise BackendUnavailable("no back-translation backend configured")


class CallableBackTranslator(BackTranslator):
    def __init__(self, spec: str):
        mod, _, fn = spec.partition(":")
        if not mod or not fn:
            raise ValueError("backend spec must look like 'package.module:function'")
        self.fn = getattr(importlib.import_module(mod), fn); self.name = spec
    def translate(self, texts, src, tgt):
        return self.fn(texts, src, tgt)


def get_backtranslator(spec: str, approved: bool) -> BackTranslator | None:
    if spec in ("", "none"):
        return None
    if not approved:
        raise BackendUnavailable("a back-translation backend was named but --i-confirm-translation-backend-approved was not given. "
                                 "Confirm project rules (model/API allowed?) and privacy/terms before sending clinical text anywhere.")
    return CallableBackTranslator(spec)


def review_flags(orig: str, new: str) -> list[str]:
    neg = re.compile(r"\b(no|not|without|denies|denied|negative|none|never|absent)\b", re.I)
    flags = []
    if len(neg.findall(orig)) != len(neg.findall(new)): flags.append("negation_count_changed")
    if sorted(re.findall(r"\d+(?:\.\d+)?", orig)) != sorted(re.findall(r"\d+(?:\.\d+)?", new)): flags.append("numbers_changed")
    return flags


# --------------------------------------------------------------------------- #
# Variant parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VariantSpec:
    fields: str
    norm: str
    rep: str
    aug: str
    @property
    def name(self) -> str:
        return f"{self.fields}__{self.norm}__{self.rep}__{self.aug}"


def parse_variant(name: str) -> VariantSpec:
    parts = name.split("__")
    if len(parts) != 4:
        raise ValueError(f"variant {name!r} must look like fields__norm__representation__augmentation")
    f, n, r, a = parts
    if f not in FIELD_SETS or n not in NORMALIZATIONS or r not in REPRESENTATIONS or a not in AUGMENTATIONS:
        raise ValueError(f"unknown component in variant {name!r}")
    return VariantSpec(f, n, r, a)


# --------------------------------------------------------------------------- #
# Ingestion outputs
# --------------------------------------------------------------------------- #
def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, sep=",", encoding="utf-8", lineterminator="\n", quoting=1)  # QUOTE_ALL


def ingest_and_write(args, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recs, _ = cc.ingest_train(args.train)
    test_recs, tinfo = cc.ingest_test(args.test)
    if args.partial_policy == "quarantine":
        for r in recs:
            if r.status == "recovered_partial":
                r.status = "quarantined"; r.flags.append("quarantined_by_partial_policy")
    rows, qrows, audit = [], [], []
    seen: dict[str, str] = {}
    for r in recs:
        key = cc.dup_normalize(r.fields["transcription"])
        gid = seen.setdefault(key, r.record_id) if key else r.record_id
        base = {"record_id": r.record_id, "source_line_start": r.source_line_start, "source_line_end": r.source_line_end, "label": r.label}
        audit.append({**base, "status": r.status, "method": r.method, "evidence": r.evidence, "flags": "|".join(r.flags),
                      "fields_absent_from_file": "|".join(r.missing_fields),
                      "empty_or_whitespace_fields": "|".join(k for k, v in r.fields.items() if not v.strip()),
                      "output_file": "quarantined_train.csv" if r.status == "quarantined" else "cleaned_train.csv",
                      "raw_sha1": r.raw_sha1, "raw_lines_json": json.dumps(r.raw_lines, ensure_ascii=False) if r.raw_lines else ""})
        rec = {**base, **r.fields, "repair_status": r.status, "repair_method": r.method, "flags": "|".join(r.flags),
               "fields_absent_from_file": "|".join(r.missing_fields), "dup_group_id": gid}
        (qrows if r.status == "quarantined" else rows).append(rec)
    train = pd.DataFrame(rows)
    write_csv(train, out / "cleaned_train.csv")
    write_csv(pd.DataFrame(qrows), out / "quarantined_train.csv")
    write_csv(pd.DataFrame(audit), out / "record_repair_audit.csv")
    test = pd.DataFrame([{"test_record_id": t.test_record_id, "original_order": t.original_order, "source_line": t.source_line, **t.fields,
                          "flags": "|".join(t.flags), "empty_fields": "|".join(f[6:] for f in t.flags if f.startswith("empty:"))} for t in test_recs])
    write_csv(test, out / "cleaned_test.csv")
    write_csv(test[["test_record_id", "original_order", "source_line", "flags", "empty_fields"]].assign(
        all_fields_empty=lambda d: d["flags"].str.contains("all_fields_empty")), out / "test_record_audit.csv")
    log.info("train: %d cleaned, %d quarantined | test: %d records (header=%s)", len(train), len(qrows), len(test), tinfo["has_header"])
    for r in recs:
        if r.status != "clean":
            log.warning("%s lines %d-%d %s via %s", r.record_id, r.source_line_start, r.source_line_end, r.status, r.method)
    return train, pd.DataFrame(qrows), test


# --------------------------------------------------------------------------- #
# Variant construction
# --------------------------------------------------------------------------- #
def field_values(row: dict, spec: VariantSpec, removal_log: list | None, rid: str) -> dict[str, str]:
    vals = {f: normalize_text(str(row[f]), spec.norm) for f in FIELD_SETS[spec.fields]}
    if spec.fields == "all_fields_no_specialty_keyword_prefix":
        # strip on the *normalized* string so lowercase/clinical give identical decisions
        new, removed = strip_keyword_prefix(vals["keywords"])
        if removed is not None and removal_log is not None:
            removal_log.append({"record_id": rid, "removed_prefix": removed, "keywords_before": vals["keywords"][:80], "keywords_after": new[:80]})
        vals["keywords"] = new
    return vals


def build_variant(spec: VariantSpec, train: pd.DataFrame, test: pd.DataFrame, args, out: Path, markers: dict[str, str],
                  backtr: BackTranslator | None, protected: set[str], synonyms: dict[str, str] | None) -> dict:
    vdir = out / "variants" / spec.name
    vdir.mkdir(parents=True, exist_ok=True)
    fnames = FIELD_SETS[spec.fields]
    removal_log: list[dict] = []
    status: dict = {"variant": spec.name, "executed": True, "notes": []}
    train_ds, test_ds = Dataset("train", train), Dataset("test", test)
    part_map = load_partition_map(args.split_manifest)
    if part_map is not None and not set(train.record_id) <= set(part_map):
        raise SystemExit("split manifest does not cover every cleaned training record; rebuild the manifest from the same cleaned data")

    def docs(ds: Dataset, id_col: str, extra: list[str], log_removals: bool) -> list[dict]:
        out_rows = []
        for r in ds.frame.to_dict("records"):
            vals = field_values(r, spec, removal_log if log_removals else None, r[id_col])
            out_rows.append({"id": r[id_col], "vals": vals, **{k: r[k] for k in extra}})
        return out_rows

    tr_docs = docs(train_ds, "record_id", ["label"], True)
    te_docs = docs(test_ds, "test_record_id", ["original_order"], False)
    if spec.fields == "all_fields_no_specialty_keyword_prefix":
        te_log: list[dict] = []
        for r in test_ds.frame.to_dict("records"):
            _, rem = strip_keyword_prefix(normalize_text(str(r["keywords"]), spec.norm))
            if rem is not None: te_log.append({"record_id": r["test_record_id"], "removed_prefix": rem, "keywords_before": r["keywords"][:80], "keywords_after": ""})
        removal_log.extend(te_log)
        write_csv(pd.DataFrame(removal_log), vdir / "keyword_prefix_removal_log.csv")

    def render(doc: dict) -> list[dict]:
        """-> list of units (1 for full/head_tail, n for windows)."""
        v = doc["vals"]
        if spec.rep == "full":
            return [{"text": build_text(v, fnames, markers), "window_index": 0, "n_windows": 1, "w_start": 0, "w_end": None, "omitted_words": 0}]
        tw = v["transcription"].split() if "transcription" in v else []
        if spec.rep == "head_tail":
            t, om = head_tail(tw, args.head_words, args.tail_words)
            vv = dict(v); vv["transcription"] = t
            return [{"text": build_text(vv, fnames, markers), "window_index": 0, "n_windows": 1, "w_start": 0, "w_end": len(tw), "omitted_words": om}]
        wins = make_windows(tw, args.window_size, args.window_overlap)
        units = []
        for i, (s, e, t) in enumerate(wins):
            if args.window_field_policy == "repeat_meta":
                vv = dict(v); vv["transcription"] = t
                text = build_text(vv, fnames, markers)
            else:
                text = f"{markers['transcription']} {t}"
            units.append({"text": text, "window_index": i, "n_windows": len(wins), "w_start": s, "w_end": e, "omitted_words": 0})
        return units

    def unit_rows(docs_: list[dict], is_test: bool) -> tuple[list[dict], list[dict]]:
        rows, manifest = [], []
        for d in docs_:
            units = render(d)
            wids = []
            for u in units:
                wid = d["id"] if spec.rep != "windows" else f"{d['id']}_W{u['window_index']:03d}"
                wids.append(wid)
                row = {"unit_id": wid, "source_record_id": d["id"], "window_index": u["window_index"], "n_windows": u["n_windows"],
                       "aug_type": "none", "aug_params": "", "text": u["text"]}
                if is_test: row["original_order"] = d["original_order"]
                else:
                    row["label"] = d["label"]
                    if part_map is not None: row["partition"] = part_map[d["id"]]
                if spec.rep == "windows": row.update({"window_start_word": u["w_start"], "window_end_word": u["w_end"]})
                rows.append(row)
            tw_len = len(d["vals"].get("transcription", "").split())
            manifest.append({("test_record_id" if is_test else "record_id"): d["id"], **({"original_order": d["original_order"]} if is_test else {"label": d["label"]}),
                             "original_transcription_words": tw_len, "n_units": len(units), "unit_ids_json": json.dumps(wids),
                             "omitted_words": sum(u["omitted_words"] for u in units),
                             "text_omitted": any(u["omitted_words"] for u in units), "representation": spec.rep})
        return rows, manifest

    tr_rows, tr_manifest = unit_rows(tr_docs, False)
    te_rows, te_manifest = unit_rows(te_docs, True)

    # ---- augmentation (train only) -------------------------------------------------
    n_aug = 0
    if spec.aug != "no_aug":
        if part_map is None:
            raise SystemExit(f"{spec.name}: augmentation requires --split-manifest (create the split first, then augment the TRAIN partition only)")
        assert_augmentable(train_ds)
        aug_docs = [d for d in tr_docs if part_map[d["id"]] == "train"]
        base_rows = [r for r in tr_rows if part_map[r["source_record_id"]] == "train"]
        extra: list[dict] = []
        if spec.aug in ("token_deletion", "synonym"):
            if spec.aug == "synonym" and not synonyms:
                status.update(executed=False); status["notes"].append("synonym augmentation skipped: no --synonym-map supplied (disabled by default)")
            else:
                for d in aug_docs:
                    for c in range(args.aug_copies):
                        rng = random.Random(f"{args.seed}:{d['id']}:{spec.aug}:{c}")
                        if spec.aug == "token_deletion":
                            tf = lambda f, v: token_deletion(v, args.deletion_prob, rng, protected)
                            params = {"p": args.deletion_prob, "seed": args.seed, "copy": c, "protected_n": len(protected)}
                        else:
                            tf = lambda f, v: synonym_replace(v, synonyms, protected)
                            params = {"map_size": len(synonyms), "copy": c}
                        dd = {**d, "vals": {k: tf(k, v) for k, v in d["vals"].items()}}
                        for u in render(dd):
                            wid = f"{d['id']}_A{spec.aug[:3].upper()}{c}" + (f"_W{u['window_index']:03d}" if spec.rep == "windows" else "")
                            extra.append({"unit_id": wid, "source_record_id": d["id"], "window_index": u["window_index"], "n_windows": u["n_windows"],
                                          "aug_type": spec.aug, "aug_params": json.dumps(params, sort_keys=True), "text": u["text"], "label": d["label"], "partition": "train",
                                          **({"window_start_word": u["w_start"], "window_end_word": u["w_end"]} if spec.rep == "windows" else {})})
        elif spec.aug == "back_translation_es":
            if backtr is None:
                status.update(executed=False)
                status["notes"].append("back-translation NOT RUN: no approved backend configured (see README). A placeholder manifest was written.")
                (vdir / "BACK_TRANSLATION_NOT_RUN.json").write_text(json.dumps({"executed": False, "reason": "no approved backend configured",
                    "pivot_language": "es", "required_before_running": ["confirm project rules allow the tool", "privacy/terms check", "pass --i-confirm-translation-backend-approved"]}, indent=2))
            else:
                texts = [r["text"] for r in base_rows]
                try:
                    pivot = backtr.translate(texts, "en", "es"); back = backtr.translate(pivot, "es", "en")
                except Exception as exc:  # explicit: backend failures must be loud and logged, never silent
                    log.error("back-translation backend %s failed: %s", backtr.name, exc)
                    status.update(executed=False); status["notes"].append(f"back-translation failed: {exc!r}")
                    back = None
                if back is not None:
                    if len(back) != len(texts): raise RuntimeError("backend returned a different number of texts")
                    for r, b in zip(base_rows, back):
                        extra.append({**r, "partition": "train", "unit_id": r["unit_id"] + "_ABTes", "aug_type": "back_translation_es", "text": b,
                                      "aug_params": json.dumps({"engine": backtr.name, "pivot": "es", "seed": args.seed})})
                    status["notes"].append(f"back-translation via {backtr.name}: {len(extra)} generated texts (UNREVIEWED; see back_translation_review_sample.csv)")
                    rev = [{"unit_id": e["unit_id"], "source_record_id": e["source_record_id"], "original": o["text"][:600], "back_translated": e["text"][:600],
                            "auto_flags": "|".join(review_flags(o["text"], e["text"])), "reviewer_meaning_preserved(y/n)": "", "reviewer_notes": ""}
                           for o, e in zip(base_rows, extra)]
                    random.Random(args.seed).shuffle(rev)
                    write_csv(pd.DataFrame(rev[: args.review_sample_size]), vdir / "back_translation_review_sample.csv")
        tr_rows = tr_rows + extra; n_aug = len(extra)

    # ---- write ------------------------------------------------------------------------
    tr_df, te_df = pd.DataFrame(tr_rows), pd.DataFrame(te_rows)
    write_csv(tr_df, vdir / ("train_windows.csv" if spec.rep == "windows" else "train.csv"))
    write_csv(te_df, vdir / ("test_windows.csv" if spec.rep == "windows" else "test.csv"))
    write_csv(pd.DataFrame(tr_manifest), vdir / "train_document_manifest.csv")
    write_csv(pd.DataFrame(te_manifest), vdir / "test_document_manifest.csv")
    cfg = {"variant": spec.name, "fields": list(fnames), "normalization": spec.norm, "representation": spec.rep, "augmentation": spec.aug,
           "markers": markers, "seed": args.seed, "window_size_words": args.window_size, "window_overlap_words": args.window_overlap,
           "window_field_policy": args.window_field_policy, "head_words": args.head_words, "tail_words": args.tail_words,
           "deletion_prob": args.deletion_prob, "aug_copies": args.aug_copies, "protected_terms": sorted(protected),
           "text_regeneration": "text = join(f'{marker} {normalize_text(field, norm)}') over the listed fields, from cleaned_train.csv / cleaned_test.csv",
           "norm_definition": normalize_text.__doc__, "alias_of": None, "status_notes": status["notes"]}
    (vdir / "variant_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    rep = validate_variant(spec, tr_df, te_df, train, test, tr_manifest, te_manifest, n_aug, status, vdir)
    return rep


def validate_variant(spec, tr_df, te_df, train, test, tr_manifest, te_manifest, n_aug, status, vdir) -> dict:
    def lens(s: pd.Series) -> dict:
        w = s.map(lambda t: len(t.split())).tolist()
        return {k: round(float(v), 1) for k, v in cc.quantile_summary(w).items()}
    orig = tr_df[tr_df.aug_type == "none"]
    exp_ids = list(test.test_record_id)
    te_docs_order = [m["test_record_id"] for m in te_manifest]
    te_order_ok = te_docs_order == exp_ids and [m["original_order"] for m in te_manifest] == list(range(len(test)))
    rows_cover = set(te_df.source_record_id) == set(exp_ids)
    if spec.rep != "windows":
        te_row_mapping_ok = list(te_df.source_record_id) == exp_ids and list(te_df.original_order) == list(range(len(test)))
    else:
        te_row_mapping_ok = list(dict.fromkeys(te_df.source_record_id)) == exp_ids and bool((te_df.original_order.diff().dropna() >= 0).all())
    full_ok = None
    if spec.rep == "windows":
        full_ok = True
        tmap = {r.record_id: normalize_text(r.transcription, spec.norm).split() for r in train.itertuples()}
        for sid, g in tr_df[tr_df.aug_type == "none"].groupby("source_record_id", sort=False):
            g = g.sort_values("window_index"); rec: list[str] = []
            for r in g.itertuples():
                toks = r.text.split()[1:] if r.text.startswith("[TRANSCRIPTION]") else r.text.split()
                rec.extend(toks[len(rec) - r.window_start_word:] if rec else toks)
            if rec != tmap[sid]:
                full_ok = False; break
    aug_mapped = bool(tr_df[tr_df.aug_type != "none"].source_record_id.isin(set(train.record_id)).all()) if n_aug else True
    if n_aug and "partition" in tr_df.columns:
        aug_mapped = aug_mapped and bool((tr_df[tr_df.aug_type != "none"].partition == "train").all())
    if spec.rep == "full":
        full_ok = True
        norm_t = {r.record_id: normalize_text(r.transcription, spec.norm) for r in train.itertuples()}
        if "transcription" in FIELD_SETS[spec.fields]:
            full_ok = all(norm_t[r.source_record_id] in r.text for r in orig.itertuples())
    rep = {"variant": spec.name, "executed": status["executed"], "notes": status["notes"],
           "train_documents": int(len(train)), "train_rows_total": int(len(tr_df)), "train_rows_original": int(len(orig)), "train_rows_augmented": int(n_aug),
           "label_counts_original_documents": dict(collections.Counter(train.label)), "label_counts_rows": dict(collections.Counter(tr_df.label)),
           "empty_text_train_rows": int((tr_df.text.str.strip() == "").sum()),
           "test_documents_expected": int(len(test)), "test_documents_in_manifest": len(te_manifest), "test_rows": int(len(te_df)),
           "test_manifest_order_equals_original": te_order_ok, "test_all_ids_covered": rows_cover, "test_row_mapping_ok": te_row_mapping_ok,
           "test_augmented_rows": int((te_df.aug_type != "none").sum()),
           "every_augmented_row_maps_to_a_train_source": aug_mapped, "full_text_preserved_check": full_ok,
           "train_length_words": lens(tr_df.text), "test_length_words": lens(te_df.text),
           "omitted_words_total_train": int(sum(m["omitted_words"] for m in tr_manifest))}
    (vdir / "validation_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


# --------------------------------------------------------------------------- #
def expand_variant_args(v: list[str] | None) -> list[str]:
    if not v:
        return list(DEFAULT_VARIANTS)
    if v == ["all-no-aug"]:
        return [f"{f}__{n}__full__no_aug" for f in FIELD_SETS if f not in ALIASES for n in NORMALIZATIONS]
    return v


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "preprocess_log.txt", mode="w", encoding="utf-8")])
    for p in (Path(args.train), Path(args.test)):
        if out.resolve() in p.resolve().parents or p.resolve().parent == out.resolve():
            raise SystemExit(f"--output-dir {out} must not contain the raw input {p}; raw files are never overwritten")
    markers = {**DEFAULT_MARKERS, **(json.loads(Path(args.markers_json).read_text()) if args.markers_json else {})}
    labels = set(cc.EXPECTED_LABELS)
    if any(m.strip("[]") in labels or m in labels for m in markers.values()):
        raise SystemExit("field markers must not equal a target label")
    protected = load_protected(args.protected_terms_file)
    synonyms = json.loads(Path(args.synonym_map).read_text(encoding="utf-8")) if args.synonym_map else None
    try:
        backtr = get_backtranslator(args.back_translation_backend, args.i_confirm_translation_backend_approved)
    except BackendUnavailable as exc:
        raise SystemExit(f"ERROR: {exc}")
    train, quarantined, test = ingest_and_write(args, out)
    names = expand_variant_args(args.variants)
    reports, manifest = [], []
    for name in names:
        spec = parse_variant(name)
        if spec.fields in ALIASES:
            tgt = f"{ALIASES[spec.fields]}__{spec.norm}__{spec.rep}__{spec.aug}"
            vdir = out / "variants" / spec.name; vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "ALIAS.json").write_text(json.dumps({"alias_of": tgt, "reason": "identical fields and markers; not duplicated on disk"}, indent=2))
            manifest.append({"variant": name, "alias_of": tgt}); continue
        log.info("building %s", name)
        rep = build_variant(spec, train, test, args, out, markers, backtr, protected, synonyms)
        reports.append(rep)
        manifest.append({"variant": name, "alias_of": None, "executed": rep["executed"], "train_rows": rep["train_rows_total"], "test_rows": rep["test_rows"],
                         "fields": list(FIELD_SETS[spec.fields]), "normalization": spec.norm, "representation": spec.rep, "augmentation": spec.aug})
    (out / "variant_manifest.json").write_text(json.dumps({"seed": args.seed, "variants": manifest, "args": {k: v for k, v in vars(args).items()}}, indent=2, default=str))
    ok = all(r["test_manifest_order_equals_original"] and r["test_row_mapping_ok"] and r["test_all_ids_covered"] and r["test_augmented_rows"] == 0
             and r["every_augmented_row_maps_to_a_train_source"] and r["full_text_preserved_check"] in (True, None) and r["test_documents_in_manifest"] == r["test_documents_expected"]
             for r in reports)
    (out / "validation_report.json").write_text(json.dumps({"all_checks_passed": ok, "test_records": int(len(test)), "train_documents_cleaned": int(len(train)),
                                                            "train_documents_quarantined": int(len(quarantined)), "variants": reports}, indent=2))
    log.info("validation: %s", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- see validation_report.json")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
