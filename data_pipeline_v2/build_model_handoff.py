#!/usr/bin/env python3
"""build_model_handoff.py -- split-first, model-ready data hand-off (data component only; no model code).

    python build_model_handoff.py --train train.csv --test test_no_labels.csv --output-dir model_handoff --seed 42

Pipeline order (each step is recorded in handoff_build_log.json):
  1 read raw -> 2 repair defensible records -> 3 quarantine the rest -> 4 validate labels/columns ->
  5 stable source IDs -> 6 exact-duplicate groups -> 7 SPLIT MANIFEST (created once, reused) ->
  8 fit learned components on TRAIN ONLY (vocabulary, boilerplate rules, max-length stats) ->
  9 transform train / validation / test with the same fixed rules -> 10 optional augmentation (TRAIN only) ->
  11 export + checksums + zip.
Validation and test rows are never augmented; there is no override.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import random
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import clinical_common as cc
import preprocess_data as pp

log = logging.getLogger("handoff")
ALL4 = ("description", "sample_name", "transcription", "keywords")
VARIANTS: dict[str, dict] = {
    "baseline_all_fields": {"fields": ALL4, "strip_keyword_prefix": True, "boilerplate": False,
                            "desc": "description + sample_name + transcription + keywords; the specialty-name prefix is removed from keywords (logged). Recommended default."},
    "clinical_text": {"fields": ("description", "transcription"), "strip_keyword_prefix": False, "boilerplate": False,
                      "desc": "description + transcription only (no sample_name, no keywords)."},
    "compressed_unique_text": {"fields": ("description", "transcription"), "strip_keyword_prefix": False, "boilerplate": True,
                               "desc": "clinical_text with (a) repeated sentences inside a note collapsed and (b) train-derived boilerplate sentences removed (negation/number-bearing sentences protected)."},
    "diagnostic_original_keywords": {"fields": ALL4, "strip_keyword_prefix": False, "boilerplate": False,
                                     "desc": "all four fields with keywords UNCHANGED (specialty-name prefix kept). Leakage diagnostic only -- not the recommended dataset."},
}
STEPS = ["1_read_raw", "2_repair_records", "3_quarantine", "4_validate_labels_columns", "5_assign_source_ids", "6_duplicate_groups",
         "7_split_manifest", "8_fit_on_train_only", "9_transform_all_partitions", "10_augment_train_only", "11_export"]
EXPECTED_PREVIOUS = {"usable_records": 2662, "quarantined_records": 7, "transcription_duplicate_groups": 731, "records_in_transcription_duplicate_groups": 1530,
                     "transcription_groups_with_label_conflict": 719, "complete_input_conflict_groups": 82, "complete_input_conflict_records": 171,
                     "surgery_percent": 33.4, "dermatology_percent": 0.9, "test_records": 398, "median_transcription_words": 387}


class PipelineLog:
    def __init__(self): self.steps: list[dict] = []
    def mark(self, step: str, **info):
        self.steps.append({"step": step, "order": len(self.steps) + 1, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info})
        log.info("step %s %s", step, info or "")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=",", encoding="utf-8", lineterminator="\n", quoting=1)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# steps 1-6
# --------------------------------------------------------------------------- #
def load_and_validate(train: str, test: str, partial_policy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    recs, _ = cc.ingest_train(train)
    trecs, tinfo = cc.ingest_test(test)
    if partial_policy == "quarantine":
        for r in recs:
            if r.status == "recovered_partial": r.status = "quarantined"; r.flags.append("quarantined_by_partial_policy")
    rows, qrows = [], []
    for r in recs:
        d = {"source_record_id": r.record_id, "source_line_start": r.source_line_start, "source_line_end": r.source_line_end,
             "medical_specialty": r.label, **r.fields, "repair_status": r.status, "repair_method": r.method, "flags": "|".join(r.flags)}
        (qrows if r.status == "quarantined" else rows).append(d)
    df, q = pd.DataFrame(rows), pd.DataFrame(qrows)
    te = pd.DataFrame([{"test_record_id": t.test_record_id, "original_order": t.original_order, "source_line": t.source_line, **t.fields,
                        "flags": "|".join(t.flags)} for t in trecs])
    # step 4: validation (fail loudly, never silently map)
    bad = set(df.medical_specialty) - set(cc.EXPECTED_LABELS)
    if bad: raise ValueError(f"cleaned training data contains non-specialty labels {bad}")
    need = {"source_record_id", "medical_specialty", *ALL4}
    if not need <= set(df.columns): raise ValueError(f"missing columns {need - set(df.columns)}")
    if df.source_record_id.duplicated().any(): raise ValueError("duplicated source_record_id")
    raw_test_lines = cc.split_physical_lines(cc.read_raw_text(test))
    raw_test_records = len(raw_test_lines) - (1 if tinfo["has_header"] else 0)
    if len(te) != raw_test_records or list(te.original_order) != list(range(len(te))):
        raise ValueError(f"test record count/order mismatch: parsed {len(te)} vs {raw_test_records} raw lines")
    return df, q, te, {"test_has_header": tinfo["has_header"], "raw_test_records": raw_test_records, "ingested": recs}


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """duplicate_group_id = identical normalised transcription (NFKC+casefold+whitespace); empty transcription -> own group.
    complete_input_group_id = identical normalised (description, sample_name, transcription, keywords) incl. ORIGINAL keywords."""
    df = df.copy()
    tkey = df.transcription.map(cc.dup_normalize)
    ckey = df[list(ALL4)].apply(lambda r: "\x1f".join(cc.dup_normalize(x) for x in r), axis=1)
    def assign(keys: pd.Series, prefix: str) -> pd.Series:
        first: dict[str, str] = {}; out = []
        for sid, k in zip(df.source_record_id, keys):
            if not k.strip("\x1f"): out.append(f"{prefix}{sid[2:]}"); continue   # empty -> singleton
            if k not in first: first[k] = f"{prefix}{sid[2:]}"
            out.append(first[k])
        return pd.Series(out, index=df.index)
    df["duplicate_group_id"] = assign(tkey, "DG")
    df["complete_input_group_id"] = assign(ckey, "CG")
    for g, pre in (("duplicate_group_id", "transcription"), ("complete_input_group_id", "complete_input")):
        grp = df.groupby(g).medical_specialty
        df[f"{pre}_group_size"] = grp.transform("size")
        df[f"{pre}_group_n_labels"] = grp.transform("nunique")
        df[f"{pre}_group_label_conflict"] = df[f"{pre}_group_n_labels"] > 1
    return df


# --------------------------------------------------------------------------- #
# step 7: split manifests
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame, strategy: str, seed: int, frac: float, partial_policy: str) -> str:
    h = hashlib.sha256(json.dumps({"ids": list(zip(df.source_record_id, df.medical_specialty, df.duplicate_group_id)), "strategy": strategy,
                                   "seed": seed, "frac": frac, "partial_policy": partial_policy}).encode()).hexdigest()
    return h


def class_check(df: pd.DataFrame, val_ids: set[str]) -> dict:
    tot = df.medical_specialty.value_counts()
    v = df[df.source_record_id.isin(val_ids)].medical_specialty.value_counts()
    overall = len(val_ids) / len(df)
    rows, warnings = {}, []
    for c in cc.EXPECTED_LABELS:
        n_all, n_val = int(tot.get(c, 0)), int(v.get(c, 0))
        rows[c] = {"total": n_all, "validation": n_val, "train": n_all - n_val, "validation_share": round(n_val / n_all, 3) if n_all else None}
        if n_all == 0 or n_val == 0 or n_all - n_val == 0: warnings.append(f"{c}: empty in a partition (total={n_all}, val={n_val})")
        elif abs(n_val / n_all - overall) > 0.30 * overall + 1e-9: warnings.append(f"{c}: validation share {n_val / n_all:.2f} deviates >30% (relatively) from target {overall:.2f}")
    return {"per_class": rows, "warnings": warnings, "stratification_feasible": not warnings}


def make_split(df: pd.DataFrame, strategy: str, frac: float, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    d = df.sort_values("source_record_id").reset_index(drop=True)
    y = d.medical_specialty.values
    if strategy == "stratified_random":
        small = d.medical_specialty.value_counts(); small = small[small < 2]
        if len(small): raise ValueError(f"stratified split infeasible, classes with <2 records: {dict(small)}")
        _, va = train_test_split(np.arange(len(d)), test_size=frac, stratify=y, random_state=seed)
    elif strategy == "duplicate_group_aware":
        k = max(2, round(1 / frac))
        _, va = next(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(d)), y, d.duplicate_group_id.values))
    else:
        raise ValueError(strategy)
    part = np.array(["train"] * len(d), dtype=object); part[va] = "validation"
    return pd.DataFrame({"source_record_id": d.source_record_id, "partition": part, "medical_specialty": d.medical_specialty,
                         "duplicate_group_id": d.duplicate_group_id, "split_strategy": strategy, "split_seed": seed})


def load_or_create_split(df, strategy, frac, seed, partial_policy, sdir: Path, tag: str, force: bool) -> tuple[pd.DataFrame, dict, bool]:
    mp, jp = sdir / f"split_manifest_{tag}.csv", sdir / f"split_metadata_{tag}.json"
    fp = fingerprint(df, strategy, seed, frac, partial_policy)
    if mp.exists() and jp.exists() and not force:
        meta = json.loads(jp.read_text())
        if meta.get("config_fingerprint") != fp:
            raise SystemExit(f"{mp} exists but was made from different data/config. Refusing to silently resplit; pass --force-resplit to replace it "
                             "(this invalidates every model result that used the old split).")
        m = pd.read_csv(mp, dtype=str, keep_default_na=False)
        if set(m.source_record_id) != set(df.source_record_id): raise SystemExit(f"{mp} does not cover the same records")
        m["split_seed"] = m["split_seed"].astype(int)
        return m, meta, True
    m = make_split(df, strategy, frac, seed)
    val = set(m[m.partition == "validation"].source_record_id)
    chk = class_check(df, val)
    for w in chk["warnings"]: log.warning("[%s] %s", tag, w)
    meta = {"split_tag": tag, "strategy": strategy, "seed": seed, "validation_fraction": frac, "partial_policy": partial_policy, "config_fingerprint": fp,
            "records_total": len(m), "records_train": int((m.partition == "train").sum()), "records_validation": len(val),
            "per_class_counts": chk["per_class"], "stratification_feasible": chk["stratification_feasible"], "stratification_warnings": chk["warnings"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "PROPOSED -- needs owner approval" if tag.startswith("primary") else "OPTIONAL robustness check -- not a substitute for the primary split"}
    write_csv(m, mp); write_json(meta, jp)
    return m, meta, False


def overlap_report(df: pd.DataFrame, m: pd.DataFrame) -> dict:
    """Reports (never repairs) duplicate/conflict overlap across the split."""
    d = df.merge(m[["source_record_id", "partition"]], on="source_record_id")
    tr, va = d[d.partition == "train"], d[d.partition == "validation"]
    out: dict = {}
    for lvl, g in (("transcription", "duplicate_group_id"), ("complete_input", "complete_input_group_id")):
        trl = tr.groupby(g).medical_specialty.agg(set)
        cross = set(tr[g]) & set(va[g])
        conflict_cross = {x for x in cross if len(set(d[d[g] == x].medical_specialty)) > 1}
        same = sum(1 for r in va.itertuples() if getattr(r, g) in trl.index and r.medical_specialty in trl[getattr(r, g)] and (len(d[d[g] == getattr(r, g)]) > 1))
        diff = sum(1 for r in va.itertuples() if getattr(r, g) in trl.index and r.medical_specialty not in trl[getattr(r, g)])
        out[lvl] = {"groups_crossing_split": len(cross), "label_conflicting_groups_crossing_split": len(conflict_cross),
                    "validation_records_with_train_twin": int(sum(1 for x in va[g] if x in trl.index)),
                    "validation_twin_in_train_same_label": int(same), "validation_twin_in_train_only_different_labels": int(diff),
                    "note": "reported only; no record was removed or relabelled"}
    return out


def validation_subsets(df: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Per validation record flags for the model team's subset metrics."""
    d = df.merge(m[["source_record_id", "partition"]], on="source_record_id")
    tr = d[d.partition == "train"]
    tr_c, tr_t = set(tr.complete_input_group_id), tr.groupby("duplicate_group_id").medical_specialty.agg(set)
    v = d[d.partition == "validation"].copy()
    v["complete_input_label_conflict"] = v.complete_input_group_label_conflict & (v.complete_input_group_size > 1)
    v["complete_input_consistent_duplicate"] = (v.complete_input_group_size > 1) & ~v.complete_input_group_label_conflict
    v["no_complete_input_twin_in_train"] = ~v.complete_input_group_id.isin(tr_c)
    v["transcription_group_label_conflict"] = v.transcription_group_label_conflict & (v.transcription_group_size > 1)
    v["transcription_twin_in_train_only_different_labels"] = [g in tr_t.index and l not in tr_t[g] for g, l in zip(v.duplicate_group_id, v.medical_specialty)]
    return v[["source_record_id", "medical_specialty", "duplicate_group_id", "complete_input_group_id", "complete_input_group_size", "transcription_group_size",
              "complete_input_label_conflict", "complete_input_consistent_duplicate", "no_complete_input_twin_in_train",
              "transcription_group_label_conflict", "transcription_twin_in_train_only_different_labels"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# step 8/9: learned components (TRAIN ONLY) and transforms
# --------------------------------------------------------------------------- #
_SENT = re.compile(r"(?<=[.!?])[,\s]+(?=[A-Z0-9\[(])")
_TOK = re.compile(r"\[[A-Z_]+\]|[a-z0-9]+(?:[.\-'/][a-z0-9]+)*|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """Markers like [DESCRIPTION] kept as-is; everything else lower-cased word / punctuation tokens."""
    out = []
    for m in re.finditer(r"\[[A-Z_]+\]|\S+", text):
        t = m.group(0)
        out.extend([t] if re.fullmatch(r"\[[A-Z_]+\]", t) else _TOK.findall(t.lower()))
    return out


def _sent_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _boiler_eligible(s: str, min_words: int, protected: set[str]) -> bool:
    return len(s.split()) >= min_words and not re.search(r"\d", s) and not any(pp.is_protected(t, protected) for t in s.split())


def fit_boilerplate(train_transcriptions: list[str], min_df: int, min_words: int, protected: set[str]) -> dict[str, int]:
    """Document-frequency statistics from TRAIN transcriptions only -> fixed rule set {sentence: df}."""
    c: collections.Counter = collections.Counter()
    for t in train_transcriptions:
        c.update({_sent_key(s) for s in _SENT.split(t) if _boiler_eligible(s, min_words, protected)})
    return {k: v for k, v in c.items() if v >= min_df}


def compress_text(t: str, rules: dict[str, int], min_words: int = 6, protected: frozenset | set = frozenset()) -> tuple[str, list[tuple[str, str]]]:
    """Drop (a) sentences matching a TRAIN-fitted boilerplate rule and (b) exact within-note repeats of a sentence.
    Repeats are only collapsed for sentences that are *eligible* (>= min_words words, no digit, no negation/protected word-part):
    list numbers ('2.'), abbreviations ('Mr.') and any number/negation-bearing sentence are never removed.
    Returns (text, [(kind, removed_sentence)]). Unchanged text is returned verbatim; a note is never emptied."""
    sents = _SENT.split(t)
    seen: set[str] = set(); kept: list[str] = []; removed: list[tuple[str, str]] = []
    for s in sents:
        k = _sent_key(s)
        if k in rules:
            removed.append(("train_boilerplate_rule", s)); continue
        if k in seen and _boiler_eligible(s, min_words, set(protected)):
            removed.append(("within_note_repeat", s)); continue
        seen.add(k); kept.append(s)
    if not removed or not kept:
        return t, []
    return " ".join(kept), removed


def build_texts(df: pd.DataFrame, spec: dict, rules: dict[str, int] | None, markers: dict[str, str], id_col: str,
                min_words: int = 6, protected: frozenset | set = frozenset()) -> tuple[pd.DataFrame, list[dict]]:
    """Text for every row with the SAME fixed rules. Returns (frame, removed-sentence log rows)."""
    texts, info, log_rows = [], [], []
    for r in df.to_dict("records"):
        vals = {f: pp.normalize_text(str(r[f]), "clinical") for f in spec["fields"]}
        row = {"n_removed_repeat": 0, "n_removed_rule": 0, "words_before": 0, "words_after": 0}
        if spec["strip_keyword_prefix"] and "keywords" in vals: vals["keywords"], _ = pp.strip_keyword_prefix(vals["keywords"])
        if spec["boilerplate"]:
            before = vals["transcription"]
            vals["transcription"], removed = compress_text(before, rules or {}, min_words, protected)
            row = {"n_removed_repeat": sum(k == "within_note_repeat" for k, _ in removed), "n_removed_rule": sum(k == "train_boilerplate_rule" for k, _ in removed),
                   "words_before": len(before.split()), "words_after": len(vals["transcription"].split())}
            for kind, sent in removed:
                log_rows.append({"record_id": r[id_col], "kind": kind, "sentence": sent, "rule_train_df": (rules or {}).get(_sent_key(sent), "")})
        texts.append(pp.build_text(vals, spec["fields"], markers)); info.append(row)
    out = df[[id_col]].copy(); out["text"] = texts
    for k in ("n_removed_repeat", "n_removed_rule", "words_before", "words_after"): out[k] = [i[k] for i in info]
    out["sentences_removed"] = out.n_removed_repeat + out.n_removed_rule
    return out, log_rows


def fit_vocab(train_texts: list[str], min_freq: int) -> dict[str, int]:
    c: collections.Counter = collections.Counter(t for x in train_texts for t in tokenize(x))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        if n >= min_freq: vocab[tok] = len(vocab)
    return vocab


# --------------------------------------------------------------------------- #
# step 10: augmentation (train only, after split)
# --------------------------------------------------------------------------- #
AUG_METHODS = {"word_dropout": "wdrop", "sentence_dropout": "sdrop"}
NEG_RE = re.compile(r"\b(no|not|without|denies|denied|negative|none|never|absent|neither|nor|cannot)\b", re.I)
_MARK_SPLIT = re.compile(r"(\[(?:DESCRIPTION|SAMPLE_NAME|TRANSCRIPTION|KEYWORDS)\])")


def word_dropout(text: str, p: float, rng: random.Random, protected: set[str], max_drop_fraction: float = 0.5) -> str:
    """Conservative WORD-level dropout on whitespace tokens (never characters; surviving words are untouched).
    Protected (never dropped): negations (incl. glued 'abscess.,no', 'non-', 'w/o'), any token containing a digit, field markers.
    At most `max_drop_fraction` of the droppable words go, so even a short note cannot be emptied."""
    toks = text.split()
    cand = [i for i, t in enumerate(toks) if not pp.is_protected(t, protected)]
    drop = [i for i in cand if rng.random() < p]
    cap = int(max_drop_fraction * len(cand))
    if len(drop) > cap:
        drop = rng.sample(drop, cap)
    gone = set(drop)
    return " ".join(t for i, t in enumerate(toks) if i not in gone)


def sentence_dropout(text: str, p: float, rng: random.Random, protected: set[str], min_sentences: int = 8, min_words: int = 100,
                     max_drop_fraction: float = 0.25) -> str | None:
    """Drop whole TRANSCRIPTION sentences, keeping the order of the rest. Only for long notes (>= min_sentences sentences and
    >= min_words words); sentences that contain a negation are never dropped; at most `max_drop_fraction` of sentences go.
    Returns None when the note is not eligible or nothing was dropped."""
    parts = _MARK_SPLIT.split(text)
    idx = next((i + 1 for i, x in enumerate(parts) if x == "[TRANSCRIPTION]" and i + 1 < len(parts)), None)
    if idx is None: return None
    lead = len(parts[idx]) - len(parts[idx].lstrip()); trail = len(parts[idx]) - len(parts[idx].rstrip())
    body = parts[idx].strip()
    sents = _SENT.split(body)
    if len(sents) < min_sentences or len(body.split()) < min_words: return None
    cand = [i for i, s in enumerate(sents) if not any(w in protected for t in s.split() for w in re.split(r"[^\w]+", t.lower()) if w)]
    drop = [i for i in cand if rng.random() < p]
    cap = int(max_drop_fraction * len(sents))
    if len(drop) > cap: drop = rng.sample(drop, cap)
    if not drop: return None
    kept = [s for i, s in enumerate(sents) if i not in set(drop)]
    parts[idx] = parts[idx][:lead] + " ".join(kept) + (" " * trail if trail else "")
    return "".join(parts)


def augment_train_rows(train_rows: pd.DataFrame, partition_of: dict[str, str], group_of: dict[str, str], method: str, p: float, copies: int, seed: int,
                       protected: set[str], sentence_min_sentences: int = 8, sentence_min_words: int = 100) -> tuple[pd.DataFrame, dict]:
    """ORIGINAL train rows + appended augmented copies (never replaces). Refuses any source that is not in the TRAIN partition (no override).
    Copies inherit `duplicate_group_id` of their source. A copy identical to its source (nothing dropped / note too short) is NOT emitted."""
    if method not in AUG_METHODS: raise ValueError(f"unknown augmentation {method!r}; supported: {sorted(AUG_METHODS)}")
    bad = [s for s in train_rows.source_record_id if partition_of.get(s) != "train"]
    if bad: raise pp.AugmentationRefused(f"{len(bad)} rows are not in the TRAIN partition (e.g. {bad[:3]}); augmentation refused")
    if not 0 <= p < 1: raise ValueError("dropout probability must be in [0, 1)")
    base = train_rows.copy(); base["augmentation_probability"] = ""; base["duplicate_group_id"] = base.source_record_id.map(group_of)
    new, skipped = [], 0
    for r in train_rows.to_dict("records"):
        for c in range(copies):
            aseed = int(hashlib.sha256(f"{seed}:{method}:{r['source_record_id']}:{c}".encode()).hexdigest()[:8], 16)
            rng = random.Random(aseed)
            if method == "word_dropout":
                params = {"method": method, "p": p, "copy": c, "base_seed": seed, "max_drop_fraction": 0.5, "unit": "whitespace word"}
                text = word_dropout(r["text"], p, rng, protected)
            else:
                params = {"method": method, "p": p, "copy": c, "base_seed": seed, "min_sentences": sentence_min_sentences, "min_words": sentence_min_words,
                          "max_drop_fraction": 0.25, "negation_sentences_protected": True}
                text = sentence_dropout(r["text"], p, rng, protected, sentence_min_sentences, sentence_min_words)
            if text is None or text == r["text"] or not text.strip():
                skipped += 1; continue
            new.append({**r, "example_id": f"{r['source_record_id']}__{AUG_METHODS[method]}{c:02d}", "augmentation_type": method, "augmentation_probability": p,
                        "augmentation_seed": aseed, "augmentation_params": json.dumps(params, sort_keys=True), "duplicate_group_id": group_of[r["source_record_id"]], "text": text})
    out = pd.concat([base, pd.DataFrame(new)], ignore_index=True) if new else base
    return out, {"source_rows": len(train_rows), "copies_requested": len(train_rows) * copies, "copies_generated": len(new), "copies_skipped_unchanged_or_ineligible": skipped}


def check_augmentation(aug: pd.DataFrame, partition_of: dict[str, str], method: str, validation_groups: set[str] | None = None) -> dict:
    """Automatic invariants; `passed` is True only if every boolean check holds."""
    a, orig = aug[aug.augmentation_type != "none"], aug[aug.augmentation_type == "none"]
    src_text = dict(zip(orig.source_record_id, orig.text))
    neg_ok = all(len(NEG_RE.findall(t)) == len(NEG_RE.findall(src_text[s])) for s, t in zip(a.source_record_id, a.text))
    dig = lambda x: sorted(w for w in x.split() if re.search(r"\d", w))
    res = {"method": method, "augmented_rows": int(len(a)), "original_rows_preserved": int(len(orig)),
           "all_augmented_map_to_original_train_record": bool(a.source_record_id.isin(orig.source_record_id).all()),
           "all_augmented_sources_in_train_partition": bool(all(partition_of.get(s) == "train" for s in a.source_record_id)),
           "no_validation_rows": bool((aug.partition == "train").all()), "example_ids_unique": bool(aug.example_id.is_unique),
           "every_augmented_row_has_type_probability_seed": bool(len(a) == 0 or ((a.augmentation_type != "").all() and (a.augmentation_probability.astype(str) != "").all() and (a.augmentation_seed.astype(str) != "").all())),
           "no_empty_augmented_text": bool(a.text.str.strip().ne("").all()), "negation_counts_preserved": bool(neg_ok),
           "augmented_differs_from_source": bool(all(t != src_text[s] for s, t in zip(a.source_record_id, a.text)))}
    if method == "word_dropout":
        res["digit_tokens_preserved"] = bool(all(dig(t) == dig(src_text[s]) for s, t in zip(a.source_record_id, a.text)))
    if validation_groups is not None:
        res["augmented_groups_disjoint_from_validation_groups"] = bool(not (set(a.duplicate_group_id) & validation_groups))
    res["passed"] = all(v for k, v in res.items() if isinstance(v, bool))
    return res


# --------------------------------------------------------------------------- #
# class weights
# --------------------------------------------------------------------------- #
def _norm_mean1(w: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(w)) or np.any(w <= 0): raise ValueError("non-finite or non-positive weight")
    return w / w.mean()


def class_weight_table(counts: dict[str, int], alphas: Sequence[float]) -> pd.DataFrame:
    """freq_c = count_c / N (a PROPORTION). weight(alpha) = normalise((1-alpha) + alpha/freq_c), mean over classes = 1.
    alpha=0 -> all 1; alpha=1 -> inverse frequency. Strategies also: inverse_sqrt = normalise(1/sqrt(freq))."""
    labs = list(cc.EXPECTED_LABELS)
    n = np.array([counts.get(l, 0) for l in labs], dtype=float)
    if np.any(n <= 0): raise ValueError(f"classes with zero training records: {[l for l, x in zip(labs, n) if x <= 0]}")
    freq = n / n.sum()
    strategies: list[tuple[str, float | None, np.ndarray]] = [("none", None, _norm_mean1(np.ones(len(labs)))),
                                                               ("inverse_frequency", None, _norm_mean1(1 / freq)),
                                                               ("inverse_sqrt_frequency", None, _norm_mean1(1 / np.sqrt(freq)))]
    for a in alphas:
        if not 0 <= a <= 1: raise ValueError("alpha must be in [0, 1]")
        strategies.append((f"interpolated_alpha_{a:g}", a, _norm_mean1((1 - a) + a / freq)))
    rows = []
    for name, a, w in strategies:
        for i, l in enumerate(labs):
            rows.append({"strategy": name, "alpha": a, "class_index": i, "medical_specialty": l, "train_count": int(n[i]), "train_frequency": round(freq[i], 6), "weight": float(w[i])})
    return pd.DataFrame(rows)


def group_sample_weights(train_df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic duplicate-group weight: 1 / (#train records in the same transcription duplicate group), normalised to mean 1 over train rows."""
    size = train_df.groupby("duplicate_group_id").source_record_id.transform("size").astype(float)
    w = _norm_mean1(1 / size.values)
    return pd.DataFrame({"source_record_id": train_df.source_record_id.values, "duplicate_group_id": train_df.duplicate_group_id.values,
                         "train_group_size": size.astype(int).values, "group_weight": w})


# --------------------------------------------------------------------------- #
# manifest / checksums
# --------------------------------------------------------------------------- #
def write_manifest(root: Path, config: dict) -> dict:
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "handoff_manifest.json":
            files[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    man = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "configuration": config, "files": files}
    write_json(man, root / "handoff_manifest.json")
    return man


def verify_manifest(root: Path) -> list[str]:
    man = json.loads((root / "handoff_manifest.json").read_text())
    problems = []
    for rel, info in man["files"].items():
        p = root / rel
        if not p.exists(): problems.append(f"missing {rel}")
        elif sha256_file(p) != info["sha256"]: problems.append(f"checksum mismatch {rel}")
    return problems


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="train.csv"); p.add_argument("--test", default="test_no_labels.csv")
    p.add_argument("--output-dir", default="model_handoff")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--partial-policy", choices=["include", "quarantine"], default="include")
    p.add_argument("--force-resplit", action="store_true", help="replace an existing split manifest made from different data/config")
    p.add_argument("--split", choices=["primary", "group_aware"], default="primary",
                   help="which split manifest drives datasets, fitted components and augmentation (use a separate --output-dir for group_aware)")
    p.add_argument("--augment", nargs="*", choices=sorted(AUG_METHODS), default=[], help="opt-in, TRAIN partition only. word_dropout is the recommended main method; sentence_dropout optional")
    p.add_argument("--augment-variants", nargs="+", default=["baseline_all_fields"], choices=sorted(VARIANTS), help="variants that receive augmented files (default: baseline_all_fields only)")
    p.add_argument("--word-dropout-prob", type=float, default=0.05); p.add_argument("--aug-copies", type=int, default=1)
    p.add_argument("--sentence-dropout-prob", type=float, default=0.10); p.add_argument("--sentence-min-sentences", type=int, default=8); p.add_argument("--sentence-min-words", type=int, default=100)
    p.add_argument("--protected-terms-file", default=None)
    p.add_argument("--min-freq", type=int, default=2, help="vocabulary min frequency (train only)")
    p.add_argument("--boilerplate-min-df", type=int, default=15, help="sentence must occur in >= this many TRAIN notes")
    p.add_argument("--boilerplate-min-words", type=int, default=6)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--no-zip", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    for pth in (Path(a.train), Path(a.test)):
        if out.resolve() in pth.resolve().parents: raise SystemExit(f"--output-dir must not contain raw input {pth}")
    if not 0 < a.val_fraction < 0.5: raise SystemExit("--val-fraction must be in (0, 0.5)")
    L = PipelineLog()
    markers, protected = dict(pp.DEFAULT_MARKERS), pp.load_protected(a.protected_terms_file)
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]

    L.mark(STEPS[0], train=a.train, test=a.test)
    df, quarantined, test, info = load_and_validate(a.train, a.test, a.partial_policy)
    L.mark(STEPS[1], recovered=int(df.repair_status.isin(["recovered", "recovered_partial"]).sum()))
    L.mark(STEPS[2], quarantined=len(quarantined))
    L.mark(STEPS[3], labels_ok=True, test_records=len(test), raw_test_records=info["raw_test_records"], test_header=info["test_has_header"])
    L.mark(STEPS[4], first=df.source_record_id.iloc[0], last=df.source_record_id.iloc[-1])
    df = add_groups(df)
    L.mark(STEPS[5], transcription_groups=int(df.duplicate_group_id.nunique()), complete_input_groups=int(df.complete_input_group_id.nunique()))

    sdir = out / "splits"
    prim, pmeta, reused = load_or_create_split(df, "stratified_random", a.val_fraction, a.seed, a.partial_policy, sdir, "primary_stratified", a.force_resplit)
    grp, gmeta, _ = load_or_create_split(df, "duplicate_group_aware", a.val_fraction, a.seed, a.partial_policy, sdir, "optional_group_aware", a.force_resplit)
    L.mark(STEPS[6], primary_reused_existing=reused, primary_train=pmeta["records_train"], primary_validation=pmeta["records_validation"])
    active = prim if a.split == "primary" else grp           # the manifest that drives datasets, fitting and augmentation
    part = dict(zip(active.source_record_id, active.partition))
    group_of = dict(zip(df.source_record_id, df.duplicate_group_id))
    ov_p, ov_g = overlap_report(df, prim), overlap_report(df, grp)
    write_json({"primary_stratified": ov_p, "optional_group_aware": ov_g}, sdir / "split_overlap_report.json")
    vs_p, vs_g = validation_subsets(df, prim), validation_subsets(df, grp)
    write_csv(vs_p, sdir / "validation_subsets_primary.csv"); write_csv(vs_g, sdir / "validation_subsets_group_aware.csv")

    # ---- 8/9: fit on train only, transform everything with the fixed rules -----------------
    train_ids = set(active[active.partition == "train"].source_record_id)
    tr_df = df[df.source_record_id.isin(train_ids)]
    rules = fit_boilerplate([pp.normalize_text(t, "clinical") for t in tr_df.transcription], a.boilerplate_min_df, a.boilerplate_min_words, protected)
    L.mark(STEPS[7], boilerplate_rules_from_train_only=len(rules), fitted_on_records=len(tr_df))
    lab_of = dict(zip(df.source_record_id, df.medical_specialty))
    vcfgs, augchecks, train_frames = {}, {}, {}
    for vname, spec in VARIANTS.items():
        tx, txlog = build_texts(df, spec, rules, markers, "source_record_id", a.boilerplate_min_words, protected)
        tt, ttlog = build_texts(test.assign(source_record_id=test.test_record_id), spec, rules, markers, "source_record_id", a.boilerplate_min_words, protected)
        base = tx.assign(example_id=tx.source_record_id, medical_specialty=tx.source_record_id.map(lab_of), partition=tx.source_record_id.map(part),
                         augmentation_type="none", augmentation_seed="", augmentation_params="")
        cols = ["example_id", "source_record_id", "medical_specialty", "partition", "augmentation_type", "augmentation_seed", "augmentation_params", "text"]
        tr_rows, va_rows = base[base.partition == "train"][cols], base[base.partition == "validation"][cols]
        vocab = fit_vocab(tr_rows.text.tolist(), a.min_freq)
        tl = tr_rows.text.map(lambda s: len(tokenize(s)))
        va_tok = [t for s in va_rows.text for t in tokenize(s)]
        oov = float(np.mean([t not in vocab for t in va_tok])) if va_tok else 0.0
        mlp = {"unit": "vocabulary tokens (see tokenize())", "train_length_percentiles": {k: float(np.percentile(tl, k)) for k in (50, 90, 95, 99, 100)},
               "suggested_max_len": int(math.ceil(np.percentile(tl, 95))), "truncation_applied_in_data": False,
               "note": "No text is truncated in the exported CSVs. Apply the SAME max_len/truncation rule to train, validation and test inside the model input pipeline."}
        d = out / "datasets" / vname
        write_csv(tr_rows, d / "train.csv"); write_csv(va_rows, d / "validation.csv")
        write_csv(pd.DataFrame({"example_id": tt.source_record_id, "test_record_id": tt.source_record_id, "original_order": test.original_order.values, "text": tt.text}), d / "test.csv")
        write_json(vocab, d / "vocab_train_only.json")
        train_frames[vname] = tr_rows
        cfg = {"variant": vname, "description": spec["desc"], "fields": list(spec["fields"]), "markers": markers, "normalization": "clinical (pp.normalize_text)",
               "numeric_normalization": "none: numbers, dosages and units kept verbatim in every variant", "keyword_specialty_prefix_removed": spec["strip_keyword_prefix"],
               "boilerplate_compression": spec["boilerplate"], "boilerplate_min_df_train": a.boilerplate_min_df, "boilerplate_min_words": a.boilerplate_min_words,
               "boilerplate_rule_count": len(rules) if spec["boilerplate"] else 0, "unk_token": "<unk>", "unk_index": 1, "pad_token": "<pad>", "pad_index": 0,
               "vocab_min_freq": a.min_freq, "vocab_size": len(vocab), "vocab_fit_on": "train partition (original, unaugmented rows) only",
               "validation_oov_token_rate_diagnostic_only": round(oov, 4), "max_length_policy": mlp,
               "rows": {"train": len(tr_rows), "validation": len(va_rows), "test": len(tt)},
               "sentences_removed_total": {"train": int(tx[tx.source_record_id.isin(train_ids)].sentences_removed.sum()), "all": int(tx.sentences_removed.sum())}}
        vcfgs[vname] = cfg
        write_json(cfg, d / "preprocessing_config.json")
        if spec["boilerplate"]:
            write_json({"fit_on": "train partition only", "min_df": a.boilerplate_min_df, "min_words": a.boilerplate_min_words,
                        "rules": dict(sorted(rules.items(), key=lambda kv: -kv[1]))}, d / "boilerplate_rules_train_only.json")
            # per-record traceability of everything compression changed (rows are never removed; text is altered)
            allc = pd.concat([tx.assign(partition=tx.source_record_id.map(part)), tt.assign(partition="test")])
            clog = allc.rename(columns={"source_record_id": "record_id"})[["record_id", "partition", "n_removed_repeat", "n_removed_rule", "words_before", "words_after"]].copy()
            clog["text_changed"] = (clog.n_removed_repeat + clog.n_removed_rule) > 0
            write_csv(clog, d / "compression_log.csv")
            pm = {**part, **{t: "test" for t in test.test_record_id}}
            write_csv(pd.DataFrame([{**r, "partition": pm[r["record_id"]]} for r in txlog + ttlog], columns=["record_id", "partition", "kind", "sentence", "rule_train_df"]),
                      d / "compression_removed_sentences.csv")
        L.mark(STEPS[8], variant=vname, train=len(tr_rows), validation=len(va_rows), test=len(tt))
    # ---- 10: augmentation, strictly AFTER the split and only from TRAIN rows ----------------------
    val_groups = set(df[df.source_record_id.map(part) == "validation"].duplicate_group_id) if a.split == "group_aware" else None
    for vname in (a.augment_variants if a.augment else []):
        for method in a.augment:
            aug, stats = augment_train_rows(train_frames[vname], part, group_of, method, a.word_dropout_prob if method == "word_dropout" else a.sentence_dropout_prob,
                                            a.aug_copies, a.seed, protected, a.sentence_min_sentences, a.sentence_min_words)
            chk = check_augmentation(aug, part, method, val_groups)
            if not chk["passed"]: raise SystemExit(f"augmentation check FAILED for {vname}/{method}: {chk}")
            write_csv(aug, out / "datasets" / vname / f"train_augmented_{method}.csv"); augchecks[f"{vname}/{method}"] = {**chk, **stats}
            L.mark(STEPS[9], variant=vname, method=method, **stats)
    if not a.augment: L.mark(STEPS[9], skipped=True, reason="augmentation is opt-in (--augment word_dropout [sentence_dropout])")
    write_json(augchecks or {"augmentation": "not requested"}, out / "augmentation_check.json")

    # ---- class counts, weights ----------------------------------------------------------------
    cnt_full = df.medical_specialty.value_counts().reindex(cc.EXPECTED_LABELS, fill_value=0)
    ctab = pd.DataFrame({"medical_specialty": cnt_full.index, "class_index": range(len(cnt_full)), "full_cleaned_count": cnt_full.values,
                         "full_cleaned_percent": (cnt_full.values / cnt_full.sum() * 100).round(2)})
    for tag, m in (("primary_train", prim), ("primary_validation", prim), ("group_aware_train", grp), ("group_aware_validation", grp)):
        want = "train" if tag.endswith("train") else "validation"
        ctab[f"{tag}_count"] = [int(((m.partition == want) & (m.medical_specialty == c)).sum()) for c in ctab.medical_specialty]
    multi = df[df.transcription_group_size > 1]
    def groups_touching(sub: pd.DataFrame) -> list[int]:
        sets = sub.groupby("duplicate_group_id").medical_specialty.agg(set)
        return [int(sum(c in x for x in sets)) for c in ctab.medical_specialty]
    ctab["duplicate_groups_size_gt1_touching_class"] = groups_touching(multi)
    ctab["label_conflicting_groups_touching_class"] = groups_touching(multi[multi.transcription_group_label_conflict])
    wdir = out / "weights"
    write_csv(ctab, wdir / "class_counts.csv")
    long = []
    for basis, m in (("primary_stratified_train_partition", prim), ("group_aware_train_partition", grp)):
        c = m[m.partition == "train"].medical_specialty.value_counts().to_dict()
        long.append(class_weight_table(c, alphas).assign(basis=basis))
    wl = pd.concat(long, ignore_index=True)
    write_csv(wl, wdir / "class_weights_long.csv")
    prim_w = wl[wl.basis == ("primary_stratified_train_partition" if a.split == "primary" else "group_aware_train_partition")]
    wide = prim_w.pivot(index=["class_index", "medical_specialty"], columns="strategy", values="weight").reset_index()
    write_csv(wide, wdir / "class_weights_primary_split_wide.csv")
    write_json({s: {r.medical_specialty: r.weight for r in g.itertuples()} for s, g in prim_w.groupby("strategy")}, wdir / "class_weights_primary_split.json")   # NB: file name kept; basis = the ACTIVE split (see --split)
    write_json({l: i for i, l in enumerate(cc.EXPECTED_LABELS)}, wdir / "label_to_index.json")
    gsw = group_sample_weights(df[df.source_record_id.isin(train_ids)])
    write_csv(gsw, wdir / "duplicate_group_sample_weights_primary_train.csv")

    # ---- mappings + audit -----------------------------------------------------------------------
    maps = df[["source_record_id", "source_line_start", "source_line_end", "medical_specialty", "repair_status", "repair_method", "flags", "duplicate_group_id", "transcription_group_size",
               "transcription_group_n_labels", "transcription_group_label_conflict", "complete_input_group_id", "complete_input_group_size", "complete_input_group_n_labels",
               "complete_input_group_label_conflict"]].merge(prim[["source_record_id", "partition"]].rename(columns={"partition": "primary_partition"}), on="source_record_id").merge(
        grp[["source_record_id", "partition"]].rename(columns={"partition": "group_aware_partition"}), on="source_record_id")
    write_csv(maps, out / "mappings" / "record_mapping.csv")
    write_csv(df[["source_record_id", "source_line_start", "source_line_end", "medical_specialty", *ALL4, "repair_status", "repair_method", "flags"]], out / "audit" / "cleaned_train_source_fields.csv")
    write_csv(test, out / "audit" / "cleaned_test_source_fields.csv")
    write_csv(quarantined[["source_record_id", "source_line_start", "source_line_end", "medical_specialty", "description", "repair_method", "flags"]] if len(quarantined) else quarantined, out / "audit" / "quarantined_train.csv")

    # ---- verification of previous findings ------------------------------------------------------
    tg = df[df.transcription_group_size > 1].drop_duplicates("duplicate_group_id")
    cg = df[(df.complete_input_group_size > 1) & df.complete_input_group_label_conflict]
    vc = df.medical_specialty.value_counts(normalize=True) * 100
    measured = {"usable_records": len(df), "quarantined_records": len(quarantined), "transcription_duplicate_groups": len(tg),
                "records_in_transcription_duplicate_groups": int(df[df.transcription_group_size > 1].shape[0]),
                "transcription_groups_with_label_conflict": int(tg.transcription_group_label_conflict.sum()),
                "complete_input_conflict_groups": int(cg.complete_input_group_id.nunique()), "complete_input_conflict_records": len(cg),
                "surgery_percent": round(float(vc["Surgery"]), 1), "dermatology_percent": round(float(vc["Dermatology"]), 1),
                "test_records": len(test), "median_transcription_words": float(np.median([len(t.split()) for t in df.transcription]))}
    kwp = df.keywords.map(lambda s: bool(cc.KEYWORD_PREFIX_RE.match(s)))
    cmp = {k: {"previously_reported": EXPECTED_PREVIOUS[k], "measured_now": measured[k], "matches": measured[k] == EXPECTED_PREVIOUS[k]} for k in EXPECTED_PREVIOUS}
    audit = {"verification_of_previous_findings": cmp, "all_previous_findings_match": all(v["matches"] for v in cmp.values()),
             "keyword_specialty_prefix_records": int(kwp.sum()), "keyword_specialty_prefix_percent": round(float(kwp.mean() * 100), 2),
             "keyword_prefix_equals_label": int(sum(1 for k, l in zip(df.keywords, df.medical_specialty) if (m := cc.KEYWORD_PREFIX_RE.match(k)) and m.group("spec").lower() in cc.KEYWORD_SPECIALTY_STRINGS[l])),
             "test_keyword_specialty_prefix_percent": round(float(test.keywords.map(lambda s: bool(cc.KEYWORD_PREFIX_RE.match(s))).mean() * 100), 2),
             "split_overlap": {"primary_stratified": ov_p, "optional_group_aware": ov_g},
             "stratification": {"primary": pmeta["stratification_warnings"], "group_aware": gmeta["stratification_warnings"]},
             "validation_subset_sizes_primary": {c: int(vs_p[c].sum()) for c in ("complete_input_label_conflict", "complete_input_consistent_duplicate", "no_complete_input_twin_in_train",
                                                                                 "transcription_group_label_conflict", "transcription_twin_in_train_only_different_labels")},
             "validation_subset_sizes_group_aware": {c: int(vs_g[c].sum()) for c in ("complete_input_label_conflict", "complete_input_consistent_duplicate", "no_complete_input_twin_in_train",
                                                                                     "transcription_group_label_conflict", "transcription_twin_in_train_only_different_labels")},
             "quarantined_ids": list(quarantined.source_record_id) if len(quarantined) else [],
             "partial_records": list(df[df.repair_status == "recovered_partial"].source_record_id)}
    write_json(audit, out / "audit" / "audit_summary.json")
    L.mark(STEPS[10])
    write_json(L.steps, out / "handoff_build_log.json")

    import handoff_docs
    handoff_docs.write_docs(out, a, audit, pmeta, gmeta, vcfgs, ctab, wl, measured)
    write_manifest(out, {"args": vars(a), "variants": vcfgs, "split_primary": pmeta, "split_group_aware": gmeta})
    problems = verify_manifest(out)
    if problems: raise SystemExit(f"manifest verification failed: {problems}")
    if not a.no_zip:
        zp = out.parent / "model_handoff.zip" if out.name == "model_handoff" else out.parent / f"{out.name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file(): z.write(p, Path("model_handoff") / p.relative_to(out))
        log.info("wrote %s", zp)
    log.info("previous findings all verified: %s", audit["all_previous_findings_match"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
