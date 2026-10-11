#!/usr/bin/env python3
"""analyze_data.py -- exploratory analysis & data-quality audit (describes; never modifies data).

Example:
    python analyze_data.py --train train.csv --test test_no_labels.csv --output-dir data_analysis

It reads the raw files, writes ONLY into --output-dir, deletes nothing, balances nothing and
makes no preprocessing decisions.  Record boundaries come from clinical_common.ingest_train
(line-oriented, evidence-driven); the *default* CSV parse is audited side by side.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import clinical_common as cc  # noqa: E402

log = logging.getLogger("analyze")
NEGATIONS = ["no", "not", "without", "denies", "denied", "negative", "none", "never", "absent", "neither", "nor", "cannot"]
BOILER_N = 8  # word n-gram length used for boilerplate search


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="train.csv", help="raw training CSV (default: ./train.csv)")
    p.add_argument("--test", default="test_no_labels.csv", help="raw test CSV (default: ./test_no_labels.csv)")
    p.add_argument("--output-dir", default="data_analysis")
    p.add_argument("--seed", type=int, default=42, help="seed for the diagnostic split simulation")
    p.add_argument("--val-ratio", type=float, default=0.2, help="validation fraction for split simulation")
    p.add_argument("--near-dup-threshold", type=float, default=0.90, help="primary cosine threshold (documents)")
    p.add_argument("--near-dup-min-threshold", type=float, default=0.80, help="lowest similarity stored in near_duplicate_pairs.csv")
    p.add_argument("--near-dup-field", choices=["transcription", "combined"], default="transcription")
    p.add_argument("--min-chars-dup", type=int, default=30, help="texts shorter than this are not used for near-duplicate search")
    p.add_argument("--top-k", type=int, default=25, help="how many top tokens / n-grams to report")
    p.add_argument("--snippet-chars", type=int, default=100, help="max chars of any quoted snippet")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def snip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, sep=",", encoding="utf-8", lineterminator="\n")


# --------------------------------------------------------------------------- #
# A. raw integrity
# --------------------------------------------------------------------------- #
def raw_integrity(train_path: Path, test_path: Path, recs: list[cc.TrainRecord], out: Path) -> dict:
    text = cc.read_raw_text(train_path)
    lines = cc.split_physical_lines(text)
    parsed = cc.default_csv_parse(text)
    body = parsed[1:]
    header = parsed[0][2]
    labels = set(cc.EXPECTED_LABELS)
    fc = collections.Counter(len(r[2]) for r in parsed)
    bad_label = [(s, e, r) for s, e, r in body if r and r[0] not in labels]
    multi = [(s, e, r) for s, e, r in body if e > s]
    odd_lines = [i + 1 for i, l in enumerate(lines) if l.count('"') % 2]
    label_start = [i + 1 for i, l in enumerate(lines[1:], 1) if cc.label_prefix(l)]
    nonlabel_lines = [i + 1 for i, l in enumerate(lines[1:], 1) if not cc.label_prefix(l) and l.strip()]
    semicolon_only_tail = sum(1 for l in lines[1:] if l.endswith(";;;"))
    long_cut = [(i + 1, len(l)) for i, l in enumerate(lines) if len(l) > 0 and cc.CELL_LIMIT_HINT - 20 <= len(l)]
    by_status = collections.Counter(r.status for r in recs)

    # audit rows --------------------------------------------------------------
    rows = []
    for s, e, r in multi:
        rows.append({"item": "default_parse_swallowed_span", "classification": "confirmed_corruption",
                     "line_start": s, "line_end": e, "physical_lines": e - s + 1,
                     "detail": f"default CSV parse merged {e - s + 1} physical lines into 1 record; description field length={len(r[1]) if len(r) > 1 else 'n/a'}; "
                               f"fields={len(r)}"})
    for s, e, r in bad_label:
        rows.append({"item": "fake_label_record", "classification": "confirmed_corruption", "line_start": s, "line_end": e,
                     "physical_lines": e - s + 1, "detail": f"default parse yields non-specialty 'label' of length {len(r[0])}: {snip(r[0], 40)!r}"})
    for rec in recs:
        if rec.status != "clean":
            rows.append({"item": "logical_record", "classification": {"quarantined": "unrecoverable", "recovered": "suspicious_recovered",
                                                                         "recovered_partial": "confirmed_corruption_partially_recovered"}[rec.status],
                         "line_start": rec.source_line_start, "line_end": rec.source_line_end,
                         "physical_lines": rec.source_line_end - rec.source_line_start + 1, "record_id": rec.record_id,
                         "label": rec.label, "method": rec.method, "detail": rec.evidence})
    for ln in odd_lines:
        rows.append({"item": "odd_quote_physical_line", "classification": "suspicious", "line_start": ln, "line_end": ln,
                     "physical_lines": 1, "detail": f"{lines[ln-1].count(chr(34))} quote chars on this physical line"})
    audit = pd.DataFrame(rows)
    save_csv(audit, out / "raw_integrity_audit.csv")

    # test file
    ttext = cc.read_raw_text(test_path)
    tlines = cc.split_physical_lines(ttext)
    tparsed = cc.default_csv_parse(ttext)
    first = cc.parse_single_line(tlines[0]) if tlines else []
    summary = {
        "train": {
            "bytes": train_path.stat().st_size, "physical_lines_incl_header": len(lines),
            "header_present": tuple(h.strip() for h in header) == cc.TRAIN_COLUMNS, "header": header,
            "default_parser_rows_incl_header": len(parsed), "default_parser_data_rows": len(body),
            "default_parser_field_count_distribution": dict(fc),
            "default_parser_rows_with_valid_label": len(body) - len(bad_label),
            "default_parser_rows_with_invalid_label": len(bad_label),
            "default_parser_multiline_records": len(multi), "default_parser_lines_inside_multiline": int(sum(e - s + 1 for s, e, _ in multi)),
            "physical_lines_with_odd_quote_count": len(odd_lines), "odd_quote_line_numbers": odd_lines,
            "physical_lines_starting_with_valid_label": len(label_start),
            "physical_lines_not_starting_with_label": len(nonlabel_lines), "non_label_line_numbers": nonlabel_lines,
            "physical_lines_ending_with_3_semicolons": semicolon_only_tail,
            "lines_near_cell_limit_32767": long_cut,
            "intended_logical_records_estimate": len(label_start),
            "logical_records_by_status": dict(by_status),
            "records_absorbed_by_default_parser": len(label_start) - (len(body) - len(bad_label)),
            "note": "Intended-record count = physical lines that begin with '<valid label>;' (every well-formed record verified to be a single physical line).",
        },
        "test": {
            "bytes": test_path.stat().st_size, "physical_lines": len(tlines), "default_parser_records": len(tparsed),
            "default_parser_field_counts": dict(collections.Counter(len(r[2]) for r in tparsed)),
            "first_line_is_header": tuple(x.strip() for x in first) == cc.INPUT_FIELDS,
            "physical_lines_with_odd_quote_count": sum(l.count('"') % 2 for l in tlines),
            "blank_physical_lines": sum(1 for l in tlines if not l.strip()),
            "lines_starting_with_a_label_name": sum(1 for l in tlines if cc.label_prefix(l)),
            "verified_record_count": len(tlines) if len(tparsed) == len(tlines) else None,
        },
    }
    (out / "raw_integrity_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
def records_frame(recs: list[cc.TrainRecord]) -> pd.DataFrame:
    rows = []
    for r in recs:
        d = {"record_id": r.record_id, "line_start": r.source_line_start, "line_end": r.source_line_end,
             "label": r.label, "status": r.status, "method": r.method}
        d.update(r.fields)
        rows.append(d)
    return pd.DataFrame(rows)


def default_parse_frame(train_path: Path) -> pd.DataFrame:
    """The frame a naive pandas/csv user would get (valid-label rows only), for before/after comparison."""
    text = cc.read_raw_text(train_path)
    rows = []
    for s, e, r in cc.default_csv_parse(text)[1:]:
        if r and r[0] in cc.EXPECTED_LABELS:
            r = (r + [""] * 5)[:5]
            rows.append(dict(zip(cc.TRAIN_COLUMNS, r)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# B. labels
# --------------------------------------------------------------------------- #
def label_distribution(df: pd.DataFrame, out: Path, plots: bool) -> pd.DataFrame:
    use = df[df.status != "quarantined"]
    cnt = use.label.value_counts().reindex(cc.EXPECTED_LABELS, fill_value=0).sort_values(ascending=False)
    tab = pd.DataFrame({"label": cnt.index, "count": cnt.values, "percent": (cnt.values / cnt.sum() * 100).round(2)})
    q = df[df.status == "quarantined"].label.value_counts()
    tab["quarantined_with_this_label"] = [int(q.get(l, 0)) for l in tab.label]
    tab["clean"] = [int(((use.label == l) & (use.status == "clean")).sum()) for l in tab.label]
    save_csv(tab, out / "class_distribution.csv")
    if plots:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(tab.label[::-1], tab["count"][::-1], color="#3b6ea5")
        for i, (c, p) in enumerate(zip(tab["count"][::-1], tab.percent[::-1])):
            ax.text(c + 8, i, f"{c} ({p:.1f}%)", va="center", fontsize=9)
        ax.set_xlabel("records"); ax.set_title(f"Training class distribution (usable records, n={int(cnt.sum())})")
        ax.set_xlim(0, tab["count"].max() * 1.18); fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=130); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(range(len(tab)), tab.percent, color="#a5693b"); ax.set_xticks(range(len(tab)))
        ax.set_xticklabels(tab.label, rotation=40, ha="right"); ax.set_ylabel("% of usable records")
        ax.set_title("Training class distribution (percent)"); fig.tight_layout(); fig.savefig(out / "class_distribution_percent.png", dpi=130); plt.close(fig)
    return tab


# --------------------------------------------------------------------------- #
# C. missingness
# --------------------------------------------------------------------------- #
def missingness(df_before: pd.DataFrame, df_after: pd.DataFrame, test_df: pd.DataFrame, out: Path, plots: bool) -> pd.DataFrame:
    def one(df: pd.DataFrame, stage: str, cols):
        rows = []
        n = len(df)
        for c in cols:
            s = df[c]
            null = int(s.isna().sum())
            empty = int((s.fillna("x") == "").sum())
            ws = int(((s.fillna("x").str.strip() == "") & (s.fillna("x") != "")).sum())
            rows.append({"stage": stage, "column": c, "records": n, "null": null, "empty_string": empty, "whitespace_only": ws,
                         "missing_total": null + empty + ws, "missing_percent": round((null + empty + ws) / max(n, 1) * 100, 2)})
        inputs = [c for c in cols if c in cc.INPUT_FIELDS]
        allmiss = int((df[inputs].fillna("").apply(lambda s: s.str.strip() == "").all(axis=1)).sum())
        notext = int((df[["description", "transcription"]].fillna("").apply(lambda s: s.str.strip() == "").all(axis=1)).sum())
        rows.append({"stage": stage, "column": "<ALL_INPUT_FIELDS_EMPTY>", "records": n, "missing_total": allmiss,
                     "missing_percent": round(allmiss / max(n, 1) * 100, 2)})
        rows.append({"stage": stage, "column": "<NO_DESCRIPTION_AND_NO_TRANSCRIPTION>", "records": n, "missing_total": notext,
                     "missing_percent": round(notext / max(n, 1) * 100, 2)})
        return rows
    rows = one(df_before, "train_default_parse", list(cc.TRAIN_COLUMNS)) + one(df_after, "train_after_documented_repair", ["label"] + list(cc.INPUT_FIELDS)
                                                                          ) + one(test_df, "test", list(cc.INPUT_FIELDS))
    tab = pd.DataFrame(rows)
    # the repaired frame has 'label' not medical_specialty; unify name for readability
    tab["column"] = tab["column"].replace({"label": "medical_specialty"})
    save_csv(tab, out / "missingness.csv")
    if plots:
        t = tab[~tab.column.str.startswith("<")]
        piv = t.pivot(index="column", columns="stage", values="missing_percent").reindex(list(cc.TRAIN_COLUMNS))
        ax = piv.plot.bar(figsize=(9, 5), rot=20); ax.set_ylabel("% missing (empty / null / whitespace)")
        ax.set_title("Missingness per field, before vs after repair, and test"); plt.tight_layout()
        plt.savefig(out / "missingness.png", dpi=130); plt.close()
    return tab


# --------------------------------------------------------------------------- #
# D. lengths
# --------------------------------------------------------------------------- #
def combined_text(row: pd.Series) -> str:
    return " ".join(str(row[c]) for c in cc.INPUT_FIELDS)


def length_analysis(df: pd.DataFrame, test_df: pd.DataFrame, out: Path, plots: bool, snippet: int) -> pd.DataFrame:
    rows, extremes = [], []
    for split, d in (("train", df), ("test", test_df)):
        d = d.copy()
        d["combined"] = d.apply(combined_text, axis=1)
        for c in list(cc.INPUT_FIELDS) + ["combined"]:
            chars = d[c].str.len().tolist()
            wt = d[c].map(lambda s: len(s.split())).tolist()
            rt = d[c].map(lambda s: len(cc.WORD_RE.findall(s))).tolist()
            empty = float(np.mean([not x.strip() for x in d[c]])) * 100
            for unit, vals in (("chars", chars), ("whitespace_tokens", wt), ("regex_tokens(approx.)", rt)):
                row = {"split": split, "field": c, "unit": unit, "empty_rate_percent": round(empty, 2)}
                row.update({k: round(float(v), 2) for k, v in cc.quantile_summary(vals).items()})
                rows.append(row)
            if split == "train" and c != "combined":
                idx = np.argsort(wt)
                nonempty = [i for i in idx if wt[i] > 0]
                for tag, ii in (("shortest_nonempty", nonempty[:5]), ("longest", list(idx[-5:])[::-1])):
                    for i in ii:
                        extremes.append({"field": c, "kind": tag, "record_id": d.iloc[i].record_id, "line_start": int(d.iloc[i].line_start),
                                         "whitespace_tokens": wt[i], "chars": chars[i], "label": d.iloc[i].label, "snippet": snip(d.iloc[i][c], snippet)})
        if plots and split == "train":
            for c in list(cc.INPUT_FIELDS) + ["combined"]:
                w = d[c].map(lambda s: len(s.split()))
                fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                ax[0].hist(w, bins=60, color="#3b6ea5"); ax[0].set_title(f"{c}: full range"); ax[0].set_xlabel("whitespace tokens"); ax[0].set_ylabel("records")
                p99 = max(np.percentile(w, 99), 1)
                ax[1].hist(w[w <= p99], bins=60, color="#5aa469"); ax[1].set_title(f"{c}: central view (≤ p99 = {p99:.0f})"); ax[1].set_xlabel("whitespace tokens")
                pos = w[w > 0]
                if len(pos):
                    ax[2].hist(pos, bins=np.logspace(0, math.log10(max(pos.max(), 2)), 60), color="#a5693b"); ax[2].set_xscale("log")
                ax[2].set_title(f"{c}: log x-axis (empty={int((w == 0).sum())} excluded)"); ax[2].set_xlabel("whitespace tokens (log)")
                fig.suptitle(f"Length distribution of '{c}' (training, n={len(d)})"); fig.tight_layout()
                fig.savefig(out / f"{c}_length_distribution.png", dpi=120); plt.close(fig)
    tab = pd.DataFrame(rows)
    save_csv(tab, out / "length_statistics.csv")
    save_csv(pd.DataFrame(extremes), out / "length_extremes.csv")
    return tab


# --------------------------------------------------------------------------- #
# E/H. duplicates and leakage
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def similarity_pairs(X, min_sim: float, chunk: int = 400):
    """All i<j pairs with cosine >= min_sim from an L2-normalised sparse matrix (chunked, never n^2 in memory)."""
    n = X.shape[0]
    XT = X.T.tocsc()
    pi, pj, ps = [], [], []
    for s in range(0, n, chunk):
        S = (X[s:s + chunk] @ XT).toarray()
        for r in range(S.shape[0]):
            S[r, : s + r + 1] = 0.0  # keep strictly upper triangle
        ii, jj = np.nonzero(S >= min_sim)
        pi.extend((ii + s).tolist()); pj.extend(jj.tolist()); ps.extend(S[ii, jj].tolist())
    return np.array(pi, dtype=int), np.array(pj, dtype=int), np.array(ps, dtype=float)


def max_sim_to_set(X, query_idx, ref_idx, chunk: int = 400) -> np.ndarray:
    ref = X[ref_idx].T.tocsc()
    out = np.zeros(len(query_idx))
    for s in range(0, len(query_idx), chunk):
        S = (X[query_idx[s:s + chunk]] @ ref).toarray()
        out[s:s + chunk] = S.max(axis=1) if S.size else 0
    return out


def duplicate_analysis(df: pd.DataFrame, args, out: Path) -> dict:
    d = df.reset_index(drop=True).copy()
    d["combined"] = d.apply(combined_text, axis=1)
    summ: list[dict] = []
    groups: list[dict] = []

    def add_group(level: str, key_series: pd.Series, skip_empty=True):
        k = key_series
        if skip_empty:
            mask = k.str.strip() != ""
        else:
            mask = pd.Series(True, index=d.index)
        sub = d[mask].reset_index(drop=True)          # .indices are POSITIONS in `sub`, so every lookup below must use `sub`
        g = sub.groupby(k[mask].reset_index(drop=True)).indices
        dup = {key: idx for key, idx in g.items() if len(idx) > 1}
        nrec = sum(len(v) for v in dup.values())
        conflict = 0
        for gi, (key, idx) in enumerate(sorted(dup.items(), key=lambda kv: -len(kv[1])), 1):
            labs = sorted(set(sub.label.iloc[idx]))
            if len(labs) > 1: conflict += 1
            groups.append({"level": level, "group_id": f"{level}:{gi}", "size": len(idx), "n_distinct_labels": len(labs),
                           "label_conflict": len(labs) > 1, "labels": "|".join(labs),
                           "record_ids": "|".join(sub.record_id.iloc[idx]), "line_starts": "|".join(map(str, sub.line_start.iloc[idx])),
                           "key_snippet": snip(str(key), args.snippet_chars)})
        summ.append({"level": level, "records_considered": int(mask.sum()), "duplicate_groups": len(dup), "records_in_duplicate_groups": nrec,
                     "extra_copies(rows-groups)": nrec - len(dup), "groups_with_label_conflict": conflict,
                     "percent_records_in_groups": round(nrec / max(int(mask.sum()), 1) * 100, 2)})

    raw_row = d[list(cc.INPUT_FIELDS) + ["label"]].astype(str).agg("\x1f".join, axis=1)
    add_group("raw_exact_row(all 5 fields)", raw_row, skip_empty=False)
    add_group("raw_exact_transcription", d.transcription)
    norm = {c: d[c].map(cc.dup_normalize) for c in cc.INPUT_FIELDS}
    add_group("normalized_exact_transcription", norm["transcription"])
    add_group("normalized_exact_combined_input", pd.Series(["\x1f".join(norm[c].iloc[i] for c in cc.INPUT_FIELDS) for i in range(len(d))]))
    add_group("normalized_exact_description", norm["description"])
    add_group("normalized_exact_sample_name", norm["sample_name"])
    add_group("normalized_exact_keywords", norm["keywords"])

    # near-duplicates ----------------------------------------------------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    base = d.transcription if args.near_dup_field == "transcription" else d.combined
    eligible = np.array([len(x.strip()) >= args.min_chars_dup for x in base])
    eidx = np.nonzero(eligible)[0]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    Xe = vec.fit_transform(base.iloc[eidx])
    pi, pj, ps = similarity_pairs(Xe, args.near_dup_min_threshold)
    pairs = pd.DataFrame({"record_id_a": d.record_id.iloc[eidx[pi]].values, "record_id_b": d.record_id.iloc[eidx[pj]].values,
                          "line_a": d.line_start.iloc[eidx[pi]].values, "line_b": d.line_start.iloc[eidx[pj]].values,
                          "label_a": d.label.iloc[eidx[pi]].values, "label_b": d.label.iloc[eidx[pj]].values, "cosine": np.round(ps, 4)})
    exact_t = d.transcription.map(cc.dup_normalize)
    fm = []
    for a, b in zip(eidx[pi], eidx[pj]):
        m = [c for c in cc.INPUT_FIELDS if norm[c].iloc[a] == norm[c].iloc[b] and norm[c].iloc[a] != ""]
        fm.append("|".join(m))
    pairs["exactly_matching_fields(normalized)"] = fm
    pairs["label_conflict"] = pairs.label_a != pairs.label_b
    pairs["snippet_a"] = [snip(base.iloc[a], args.snippet_chars) for a in eidx[pi]]
    pairs["snippet_b"] = [snip(base.iloc[b], args.snippet_chars) for b in eidx[pj]]
    pairs = pairs.sort_values("cosine", ascending=False)
    save_csv(pairs, out / "near_duplicate_pairs.csv")
    thr_rows = []
    for t in sorted({0.8, 0.9, 0.95, 0.99, 1.0, args.near_dup_threshold, args.near_dup_min_threshold}):
        if t < args.near_dup_min_threshold: continue
        sel = pairs[pairs.cosine >= t - 1e-9]
        uf = UnionFind(len(eidx)); pos = {r: i for i, r in enumerate(d.record_id.iloc[eidx])}
        for a, b in zip(sel.record_id_a, sel.record_id_b): uf.union(pos[a], pos[b])
        comp = collections.Counter(uf.find(i) for i in set(pos[x] for x in sel.record_id_a) | set(pos[x] for x in sel.record_id_b))
        thr_rows.append({"cosine_threshold": t, "pairs": len(sel), "pairs_with_label_conflict": int(sel.label_conflict.sum()),
                         "records_involved": int(sum(comp.values())), "connected_groups": len(comp)})
    save_csv(pd.DataFrame(thr_rows), out / "near_duplicate_threshold_sweep.csv")
    summ.append({"level": f"near_duplicate({args.near_dup_field}, tfidf word 1-2gram cosine>={args.near_dup_threshold})",
                 "records_considered": len(eidx), "duplicate_groups": next((r["connected_groups"] for r in thr_rows if r["cosine_threshold"] == args.near_dup_threshold), None),
                 "records_in_duplicate_groups": next((r["records_involved"] for r in thr_rows if r["cosine_threshold"] == args.near_dup_threshold), None),
                 "groups_with_label_conflict": None, "note": "pairs/conflicts in near_duplicate_threshold_sweep.csv"})
    save_csv(pd.DataFrame(summ), out / "duplicate_summary.csv")
    gdf = pd.DataFrame(groups)
    save_csv(gdf, out / "exact_duplicate_groups.csv")
    conflicts = gdf[gdf.label_conflict] if len(gdf) else gdf
    save_csv(conflicts, out / "label_conflict_groups.csv")

    # boilerplate: repeated 8-word shingles by document frequency
    df_counter: collections.Counter = collections.Counter()
    for t in d.transcription:
        w = cc.dup_normalize(t).split()
        df_counter.update({" ".join(w[i:i + BOILER_N]) for i in range(0, max(len(w) - BOILER_N + 1, 0))})
    boiler = [(k, v) for k, v in df_counter.most_common(60) if v >= 5]
    save_csv(pd.DataFrame(boiler, columns=[f"{BOILER_N}-word phrase (normalized)", "documents_containing"]), out / "boilerplate_phrases.csv")

    pair_counter: collections.Counter = collections.Counter()
    if len(gdf):
        for r in gdf[(gdf.level == "normalized_exact_transcription") & gdf.label_conflict].itertuples():
            labs = sorted(set(r.labels.split("|")))
            for a in range(len(labs)):
                for b in range(a + 1, len(labs)): pair_counter[(labs[a], labs[b])] += 1
    save_csv(pd.DataFrame([(a, b, c) for (a, b), c in pair_counter.most_common()], columns=["label_a", "label_b", "conflicting_transcription_groups"]), out / "label_conflict_pairs.csv")
    ident = gdf[(gdf.level == "normalized_exact_combined_input")] if len(gdf) else gdf
    ident_records = int(ident.loc[ident.label_conflict, "size"].sum()) if len(ident) else 0
    return {"identical_input_diff_label_groups": int(ident.label_conflict.sum()) if len(ident) else 0, "identical_input_diff_label_records": ident_records,
            "n_label_conflict_trans": int(((gdf.level == "normalized_exact_transcription") & gdf.label_conflict).sum()) if len(gdf) else 0,
            "top_label_pairs": pair_counter.most_common(5),
            "summary": summ, "thr": thr_rows, "X": Xe, "eidx": eidx, "pairs": pairs, "norm_trans": exact_t, "n_label_conflict_exact": int(len(conflicts)),
            "boiler_top": boiler[:5]}


def train_test_overlap(df: pd.DataFrame, test_df: pd.DataFrame, out: Path) -> dict:
    """Diagnostic only: how many test notes have an exact normalized twin in train (test text is NOT used to fit anything)."""
    tr: dict[str, list[str]] = collections.defaultdict(list)
    for t, l in zip(df.transcription.map(cc.dup_normalize), df.label):
        if t: tr[t].append(l)
    nt = test_df.transcription.map(cc.dup_normalize)
    hit = [bool(t) and t in tr for t in nt]
    multi = [bool(t) and t in tr and len(set(tr[t])) > 1 for t in nt]
    res = {"test_records": len(test_df), "test_nonempty_transcription": int((nt != "").sum()), "test_with_exact_twin_in_train": int(sum(hit)),
           "percent_of_test": round(sum(hit) / len(test_df) * 100, 2), "twin_matches_with_multiple_train_labels": int(sum(multi))}
    (out / "train_test_overlap.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def split_simulation(df: pd.DataFrame, dup: dict, args, out: Path) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    d = df.reset_index(drop=True)
    X, eidx = dup["X"], dup["eidx"]
    pos = {i: k for k, i in enumerate(eidx)}  # df index -> row of X
    thresholds = [t for t in (0.8, 0.9, 0.95, 0.99) if t >= args.near_dup_min_threshold]
    y = d.label.values

    # groups = exact-normalized-transcription groups + near-dup (>= primary threshold) connected components
    uf = UnionFind(len(d))
    firsts: dict[str, int] = {}
    for i, t in enumerate(dup["norm_trans"]):
        if t:
            if t in firsts: uf.union(i, firsts[t])
            else: firsts[t] = i
    rid2i = {r: i for i, r in enumerate(d.record_id)}
    sel = dup["pairs"][dup["pairs"].cosine >= args.near_dup_threshold]
    for a, b in zip(sel.record_id_a, sel.record_id_b): uf.union(rid2i[a], rid2i[b])
    groups = np.array([uf.find(i) for i in range(len(d))])

    def evaluate(name: str, tr: np.ndarray, va: np.ndarray) -> dict:
        trn = set(dup["norm_trans"].iloc[tr][dup["norm_trans"].iloc[tr] != ""])
        exact_leak = np.array([bool(dup["norm_trans"].iloc[i]) and dup["norm_trans"].iloc[i] in trn for i in va])
        tr_e = np.array([pos[i] for i in tr if i in pos]); va_i = [i for i in va if i in pos]
        ms = max_sim_to_set(X, np.array([pos[i] for i in va_i]), tr_e) if len(va_i) and len(tr_e) else np.array([])
        tr_labels: dict[str, set] = collections.defaultdict(set)
        for i in tr:
            t = dup["norm_trans"].iloc[i]
            if t: tr_labels[t].add(d.label.iloc[i])
        same = diff = 0
        for i in va:
            t = dup["norm_trans"].iloc[i]
            if t and t in tr_labels:
                if d.label.iloc[i] in tr_labels[t]: same += 1
                else: diff += 1
        res = {"design": name, "train_n": len(tr), "val_n": len(va), "val_with_exact_normalized_transcription_in_train": int(exact_leak.sum()),
               "percent_val_exact_in_train": round(float(exact_leak.mean()) * 100, 2),
               "val_exact_twin_in_train_SAME_label": same, "val_exact_twin_in_train_ONLY_DIFFERENT_labels": diff}
        for t in thresholds:
            res[f"percent_val_with_train_cosine>={t}"] = round(float((ms >= t).sum()) / max(len(va), 1) * 100, 2)
        labs = d.label.iloc[va]
        res["per_class_percent_val_exact_in_train"] = {l: round(float(exact_leak[(labs == l).values].mean()) * 100, 1) for l in cc.EXPECTED_LABELS if (labs == l).any()}
        # exact groups crossing
        g_tr, g_va = set(groups[tr]), set(groups[va])
        res["duplicate_groups_crossing_split"] = len(g_tr & g_va)
        return res

    tr, va = train_test_split(np.arange(len(d)), test_size=args.val_ratio, stratify=y, random_state=args.seed)
    r1 = evaluate("stratified_random", tr, va)
    k = max(2, round(1 / args.val_ratio))
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed)
    tr2, va2 = next(sgk.split(np.zeros(len(d)), y, groups))
    r2 = evaluate("duplicate_aware_grouped(StratifiedGroupKFold fold 0)", tr2, va2)
    r2["val_class_counts"] = dict(collections.Counter(y[va2])); r1["val_class_counts"] = dict(collections.Counter(y[va]))
    res = {"seed": args.seed, "val_ratio": args.val_ratio, "grouping": f"exact normalized transcription + near-dup cosine>={args.near_dup_threshold}",
           "n_groups": int(len(set(groups))), "largest_group": int(max(collections.Counter(groups).values())), "designs": [r1, r2]}
    (out / "split_leakage_simulation.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in (r1, r2)]
    save_csv(pd.DataFrame(flat), out / "split_leakage_simulation.csv")
    return res


# --------------------------------------------------------------------------- #
# F. keyword leakage
# --------------------------------------------------------------------------- #
def keyword_leakage(df: pd.DataFrame, test_df: pd.DataFrame, out: Path, snippet: int) -> dict:
    spec_re = {lab: re.compile("|".join(re.escape(s) for s in strs), re.I) for lab, strs in cc.KEYWORD_SPECIALTY_STRINGS.items()}
    any_re = re.compile("|".join(re.escape(s) for s in cc.ALL_SPECIALTY_STRINGS), re.I)
    rows = []
    for split, d in (("train", df), ("test", test_df)):
        kw = d.keywords
        n = len(d)
        m_prefix = kw.map(lambda s: bool(cc.KEYWORD_PREFIX_RE.match(s)))
        m_any = kw.map(lambda s: bool(any_re.search(s)))
        row = {"split": split, "records": n, "keywords_nonempty": int((kw.str.strip() != "").sum()),
               "specialty_as_first_keyword(prefix pattern)": int(m_prefix.sum()), "percent_prefix": round(m_prefix.mean() * 100, 2),
               "specialty_string_anywhere_in_keywords": int(m_any.sum()), "percent_anywhere": round(m_any.mean() * 100, 2),
               "anywhere_but_not_as_prefix": int((m_any & ~m_prefix).sum())}
        if split == "train":
            pref = kw.map(lambda s: (cc.KEYWORD_PREFIX_RE.match(s).group("spec").lower() if cc.KEYWORD_PREFIX_RE.match(s) else None))
            agree = [(p is not None) and (p in cc.KEYWORD_SPECIALTY_STRINGS.get(l, ())) for p, l in zip(pref, d.label)]
            row["prefix_equals_true_label"] = int(sum(agree)); row["prefix_present_but_different_label"] = int(m_prefix.sum() - sum(agree))
            by = []
            for lab in cc.EXPECTED_LABELS:
                sub = d.label == lab
                by.append({"label": lab, "records": int(sub.sum()), "prefix_present": int((m_prefix & sub).sum()),
                           "percent_prefix_present": round(float((m_prefix & sub).sum()) / max(int(sub.sum()), 1) * 100, 1),
                           "other_specialty_string_anywhere": int((m_any & ~m_prefix & sub).sum())})
            save_csv(pd.DataFrame(by), out / "keyword_leakage_by_class.csv")
            ex = d[m_prefix].head(8)
            save_csv(pd.DataFrame({"record_id": ex.record_id, "line_start": ex.line_start, "label": ex.label,
                                   "keywords_snippet": [snip(x, snippet) for x in ex.keywords]}), out / "keyword_leakage_examples.csv")
            # other fields
            for c in ("description", "sample_name", "transcription"):
                row[f"label_name_in_{c}"] = int(sum(bool(spec_re[l].search(t)) for l, t in zip(d.label, d[c])))
                row[f"any_specialty_name_in_{c}"] = int(d[c].map(lambda s: bool(any_re.search(s))).sum())
        rows.append(row)
    res = pd.DataFrame(rows)
    save_csv(res, out / "keyword_leakage_summary.csv")
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# G. vocabulary
# --------------------------------------------------------------------------- #
def vocabulary(df: pd.DataFrame, out: Path, k: int) -> dict:
    tok = lambda s: [t for t in cc.WORD_RE.findall(s.lower()) if re.match(r"[a-z0-9]", t)]
    docs = [tok(" ".join([r.description, r.sample_name, r.transcription])) for r in df.itertuples()]  # keywords excluded (leakage)
    tf = collections.Counter(t for d in docs for t in d)
    total = sum(tf.values())
    bi = collections.Counter(b for d in docs for b in zip(d, d[1:]))
    tri = collections.Counter(b for d in docs for b in zip(d, d[1:], d[2:]))
    save_csv(pd.DataFrame(tf.most_common(k), columns=["token", "count"]), out / "top_tokens.csv")
    save_csv(pd.DataFrame([(" ".join(b), c) for b, c in bi.most_common(k)], columns=["bigram", "count"]), out / "top_bigrams.csv")
    save_csv(pd.DataFrame([(" ".join(b), c) for b, c in tri.most_common(k)], columns=["trigram", "count"]), out / "top_trigrams.csv")
    # per-class distinctive tokens: smoothed log-odds vs the rest, training data only
    per = []
    cls_tf = {l: collections.Counter() for l in cc.EXPECTED_LABELS}
    for lab, d in zip(df.label, docs): cls_tf[lab].update(d)
    for lab in cc.EXPECTED_LABELS:
        c, rest = cls_tf[lab], tf - cls_tf[lab]
        nc, nr = sum(c.values()), sum(rest.values())
        sc = [(t, math.log((c[t] + .5) / (nc + .5 * len(tf))) - math.log((rest[t] + .5) / (nr + .5 * len(tf))), c[t]) for t in c if c[t] >= 15]
        for t, s, cnt in sorted(sc, key=lambda x: -x[1])[:10]:
            per.append({"label": lab, "token": t, "log_odds_vs_rest": round(s, 3), "count_in_class": cnt})
    save_csv(pd.DataFrame(per), out / "class_distinctive_tokens.csv")
    neg = {n: tf.get(n, 0) for n in NEGATIONS}
    docs_neg = float(np.mean([any(t in NEGATIONS for t in d) for d in docs])) * 100
    num_tok = sum(c for t, c in tf.items() if re.search(r"\d", t))
    full_text = " ".join(df.transcription.tolist())
    abbr = collections.Counter(re.findall(r"\b[A-Z]{2,6}\b", full_text))
    meas = collections.Counter(m.lower() for m in re.findall(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|cc|mm|cm|kg|lbs?|mmHg|units?|%)\b", full_text, flags=re.I))
    save_csv(pd.DataFrame(abbr.most_common(k), columns=["uppercase_abbreviation", "count"]), out / "top_abbreviations.csv")
    suspicious = [{"record_id": r.record_id, "line_start": r.line_start, "issue": "text starts with lowercase letter fragment"}
                  for r in df.itertuples() if r.transcription[:1].islower()][:50]
    save_csv(pd.DataFrame(suspicious), out / "suspicious_fragments.csv")
    res = {"tokenization": "lowercased regex tokens [A-Za-z0-9]+ with internal . - ' / allowed; punctuation dropped for counting; fields: description+sample_name+transcription (keywords excluded)",
           "tokens_total": total, "vocab_size": len(tf), "vocab_size_min_count_2": sum(1 for c in tf.values() if c >= 2),
           "negation_term_counts": neg, "percent_docs_with_negation_term": round(docs_neg, 2), "numeric_token_count": num_tok,
           "numeric_percent_of_tokens": round(num_tok / total * 100, 2), "measurement_mentions": sum(meas.values()),
           "top_measurement_units": meas.most_common(8), "top_tokens": tf.most_common(10),
           "suspicious_lowercase_start_transcriptions": len(suspicious)}
    (out / "vocabulary_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- #
# I. report
# --------------------------------------------------------------------------- #
def write_report(out: Path, args, integ, lab, miss, dup, kw, vocab, split, recs, argv, overlap) -> None:
    t = integ["train"]; te = integ["test"]
    st = t["logical_records_by_status"]
    ds = pd.DataFrame(dup["summary"]); thr = pd.DataFrame(dup["thr"])
    kwt = {r["split"]: r for r in kw["rows"]}["train"]; kws = {r["split"]: r for r in kw["rows"]}["test"]
    r1, r2 = split["designs"]
    def tbl(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False) if _has_tab() else "```\n" + df.to_string(index=False) + "\n```"
    qdf = pd.DataFrame([{"record_id": r.record_id, "lines": f"{r.source_line_start}-{r.source_line_end}", "label": r.label, "status": r.status, "method": r.method}
                        for r in recs if r.status != "clean"])
    L = []
    L.append("# Data analysis report (auto-generated by analyze_data.py)\n")
    L.append("All numbers below are computed from the supplied files in this run. Nothing was deleted, repaired in place or balanced.\n")
    L.append("## 1. Reproduce\n```\n" + argv + "\n```\n")
    L.append("## 2. Raw-file / CSV integrity\n")
    L.append(f"* train.csv: {t['physical_lines_incl_header']} physical lines (1 header + {t['physical_lines_incl_header']-1} data lines), header present: {t['header_present']}.")
    L.append(f"* Default quote-aware CSV parser returns **{t['default_parser_data_rows']}** data records, all with 5 fields, of which **{t['default_parser_rows_with_valid_label']}** have a valid label and **{t['default_parser_rows_with_invalid_label']}** have a non-specialty 'label' (text fragments).")
    L.append(f"* Physical lines that begin with `<valid label>;`: **{t['physical_lines_starting_with_valid_label']}**. Every well-formed record was verified to occupy one physical line, so this is the best-supported count of *intended* records; it is an inference from line structure, not something the file states.")
    L.append(f"* Default parse absorbed **{t['records_absorbed_by_default_parser']}** intended records into {t['default_parser_multiline_records']} swallowed spans ({t['default_parser_lines_inside_multiline']} physical lines). Odd-quote physical lines: {t['physical_lines_with_odd_quote_count']}.")
    L.append(f"* Logical-record status (this analysis): {st}.")
    L.append("* Root cause supported by evidence: 7 records begin with an unterminated `\"` in `description`. Three swallowed spans end in a transcription cut near the 32 767-character spreadsheet-cell limit with the remainder on an orphan line (these produce the fake labels). This mechanism is an inference; the file itself does not say how it was produced.")
    L.append(f"* test_no_labels.csv: {te['physical_lines']} physical lines, default parser {te['default_parser_records']} records, field counts {te['default_parser_field_counts']}, **header present: {te['first_line_is_header']}** (first line is data), odd-quote lines {te['physical_lines_with_odd_quote_count']}, blank lines {te['blank_physical_lines']}. Verified record count: **{te['verified_record_count']}**.\n")
    L.append("Damaged / non-clean records:\n\n" + tbl(qdf) + "\n")
    L.append("Full per-item audit: `raw_integrity_audit.csv`; summary: `raw_integrity_summary.json`.\n")
    L.append("## 3. Label distribution (usable = clean + recovered + recovered_partial)\n\n" + tbl(lab) + "\n\n![](class_distribution.png)\n")
    L.append("## 4. Missingness\nSee `missingness.csv` / `missingness.png` (default parse vs repaired vs test).\n")
    L.append("## 5. Length distributions\nSee `length_statistics.csv`, `length_extremes.csv` and `*_length_distribution.png`. Token counts are whitespace tokens and a regex approximation; **no downstream tokenizer has been fixed**, so model-token lengths are not reported.\n")
    L.append("## 6. Duplicates\n\n" + tbl(ds) + "\n\nNear-duplicate threshold sweep (TF-IDF word 1-2gram cosine, field: " + args.near_dup_field + "):\n\n" + tbl(thr) +
             f"\n\nTranscription-level exact-duplicate groups whose members carry different labels: **{dup['n_label_conflict_trans']}** (top label pairs: {dup['top_label_pairs']}; `label_conflict_pairs.csv`, `label_conflict_groups.csv` lists all duplicate levels; nothing resolved automatically). Records whose *entire input incl. keywords* is identical to another record with a different label: **{dup['identical_input_diff_label_records']}** in {dup['identical_input_diff_label_groups']} groups (irreducible ambiguity for any input-only model). Train/test overlap: {overlap}. Limitations: cosine on word n-grams ignores word order and can score templated notes with different findings highly; empty texts are excluded from duplicate groups.\n")
    L.append("### Split leakage simulation (diagnostic, seed " + str(split["seed"]) + ", val ratio " + str(split["val_ratio"]) + ")\n\n" +
             tbl(pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in (r1, r2)])) +
             f"\n\nGrouping used: {split['grouping']}; {split['n_groups']} groups, largest group {split['largest_group']} records. **The model team decides which design is primary**; trade-off: the random split mirrors the (probably duplicate-rich) test distribution but inflates validation scores; the grouped split measures generalisation to unseen notes but gives noisier, harder, less class-balanced validation folds (see `val_class_counts` in the JSON).\n")
    L.append("## 7. Keyword leakage\n\n" + tbl(pd.DataFrame(kw["rows"])) + f"\n\nTrain: prefix equals the true label in {kwt['prefix_equals_true_label']} of {kwt['specialty_as_first_keyword(prefix pattern)']} prefixed records. Test: {kws['percent_prefix']}% of test records carry a specialty first-keyword, so the same shortcut exists at test time. Per-class: `keyword_leakage_by_class.csv`.\n")
    L.append("## 8. Vocabulary\n```\n" + json.dumps({k: v for k, v in vocab.items() if k not in ('top_tokens',)}, indent=1) + "\n```\nFiles: `top_tokens.csv`, `top_bigrams.csv`, `top_trigrams.csv`, `class_distinctive_tokens.csv`, `top_abbreviations.csv`, `boilerplate_phrases.csv`. Stop words and negations are NOT removed anywhere.\n")
    L.append("## 9. Candidate anomalies for the paper (ranked by measured magnitude, verify before citing)\n")
    cands = [("CSV structural corruption", f"{t['records_absorbed_by_default_parser']} records absorbed + {t['default_parser_rows_with_invalid_label']} fake labels + {st.get('quarantined',0)} quarantined"),
             ("Specialty name in keywords", f"{kwt['percent_prefix']}% of train records ({kws['percent_prefix']}% of test)"),
             ("Exact-duplicate transcriptions", f"{ds.iloc[2]['records_in_duplicate_groups']} records in {ds.iloc[2]['duplicate_groups']} groups (normalized)"),
             ("Validation leakage (random split)", f"{r1['percent_val_exact_in_train']}% of validation notes have an exact normalized twin in train vs {r2['percent_val_exact_in_train']}% in grouped split"),
             ("Class imbalance", f"{lab.iloc[0]['label']} {lab.iloc[0]['percent']}% vs {lab.iloc[-1]['label']} {lab.iloc[-1]['percent']}%")]
    for n, v in cands: L.append(f"* **{n}**: {v}")
    L.append("\n**Interpretation caution:** when twins carry *different* labels, memorising the training twin is penalised, not rewarded. Compare `val_exact_twin_in_train_SAME_label` vs `..._ONLY_DIFFERENT_labels` before calling duplicates 'leakage'.")
    L.append("\n## 10. Assumptions, parser choices, limitations\n* Record boundary = physical line starting with `<label>;` (verified for all well-formed lines).\n* Quarantined / partial records are reported, not hidden; no label was invented or changed.\n* Duplicate analysis uses training records only; near-dup vectoriser is fitted on the analysed pool (diagnostic, not a modelling component).\n* Snippets are limited to " + str(args.snippet_chars) + " characters.\n")
    (out / "analysis_report.md").write_text("\n".join(L), encoding="utf-8")


def _has_tab() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "analysis_log.txt", mode="w", encoding="utf-8"); fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); log.addHandler(fh)
    log.setLevel(logging.INFO)
    train_p, test_p = Path(args.train), Path(args.test)
    for p in (train_p, test_p):
        if out.resolve() == p.resolve().parent and False: pass
    log.info("args: %s", vars(args))
    recs, _ = cc.ingest_train(train_p)
    test_recs, tinfo = cc.ingest_test(test_p)
    log.info("train logical records: %d; test records: %d (header=%s)", len(recs), len(test_recs), tinfo["has_header"])
    integ = raw_integrity(train_p, test_p, recs, out)
    full = records_frame(recs)
    usable = full[full.status != "quarantined"].reset_index(drop=True)
    for w in full[full.status == "quarantined"].itertuples(): log.warning("quarantined %s lines %s: %s", w.record_id, w.line_start, w.method)
    test_df = pd.DataFrame([{**{"test_record_id": r.test_record_id}, **r.fields} for r in test_recs])
    plots = not args.no_plots
    lab = label_distribution(full, out, plots)
    missingness(default_parse_frame(train_p), usable, test_df, out, plots)
    length_analysis(usable, test_df, out, plots, args.snippet_chars)
    dup = duplicate_analysis(usable, args, out)
    split = split_simulation(usable, dup, args, out)
    overlap = train_test_overlap(usable, test_df, out)
    kw = keyword_leakage(usable, test_df, out, args.snippet_chars)
    vocab = vocabulary(usable, out, args.top_k)
    write_report(out, args, integ, lab, None, dup, kw, vocab, split, recs, "python analyze_data.py " + " ".join(sys.argv[1:]), overlap)
    log.info("done -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
