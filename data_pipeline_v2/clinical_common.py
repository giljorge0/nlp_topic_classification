"""Shared, policy-free utilities for the clinical-note data pipeline.

Both ``analyze_data.py`` and ``preprocess_data.py`` import ONLY raw ingestion
and text helpers from here.  No analysis conclusions and no preprocessing
policies (augmentation, field selection, ...) live in this module.

Raw-file findings that drive the ingestion design (verified on the supplied
files, see README / analysis_report.md):

* Physical lines end in CRLF; no BOM; ASCII + a few non-ASCII chars; UTF-8.
* Every *intended* training record occupies exactly ONE physical line and
  starts with ``<label>;``.  A few records contain an unbalanced ``"`` which
  makes a quote-aware CSV parser swallow the following physical lines.
* Swallowed spans are reproducible (an "open" line with only 2 fields and one
  quote, then ordinary records, then a "terminal" line).  Three terminal lines
  were additionally cut at ~32 767 characters (the spreadsheet cell limit) and
  their remainder sits on the next physical line as an orphan fragment.
* The test file has no header and 4 fields per physical line.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

csv.field_size_limit(10**9)

DELIMITER = ";"
EXPECTED_LABELS: tuple[str, ...] = (
    "Surgery", "Cardiovascular-Pulmonary", "Orthopedic", "Radiology",
    "General Medicine", "Gastroenterology", "Neurology", "Obstetrics-Gynecology",
    "Neurosurgery", "Ophthalmology", "Psychiatry-Psychology", "Dermatology",
)
TRAIN_COLUMNS = ("medical_specialty", "description", "sample_name", "transcription", "keywords")
INPUT_FIELDS = ("description", "sample_name", "transcription", "keywords")
CELL_LIMIT_HINT = 32767  # spreadsheet cell limit; used only as supporting evidence


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class TrainRecord:
    """One logical training record with full provenance."""
    record_id: str                      # TR000001... assigned in source order of record start
    source_line_start: int              # 1-based physical line (header = line 1)
    source_line_end: int
    label: str | None
    fields: dict[str, str]              # the four input fields (missing -> "")
    status: str                         # clean | recovered | recovered_partial | quarantined
    method: str                         # how it was obtained
    evidence: str = ""                  # human-readable evidence
    flags: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)  # absent in the FILE (not merely empty)
    raw_lines: list[str] = field(default_factory=list)       # raw text (kept for non-clean only)
    raw_sha1: str = ""


@dataclass
class TestRecord:
    test_record_id: str                 # TE000001...
    original_order: int                 # 0-based row position in the file
    source_line: int
    fields: dict[str, str]
    status: str                         # ok | flagged
    flags: list[str] = field(default_factory=list)
    raw_line: str = ""


# --------------------------------------------------------------------------- #
# Raw reading
# --------------------------------------------------------------------------- #
def read_raw_text(path: str | Path) -> str:
    """Read the file as UTF-8 text WITHOUT newline translation."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    try:
        return p.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{p} is not valid UTF-8: {exc}") from exc


def split_physical_lines(text: str) -> list[str]:
    """Split on CRLF / LF / CR. A trailing terminator does not create an extra line."""
    lines = re.split(r"\r\n|\n|\r", text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def parse_single_line(line: str) -> list[str]:
    """Parse ONE physical line with the csv module (never lets a quote cross lines)."""
    rows = list(csv.reader([line], delimiter=DELIMITER))
    return rows[0] if rows else []


def default_csv_parse(text: str, delimiter: str = DELIMITER) -> list[tuple[int, int, list[str]]]:
    """The 'naive' quote-aware parse: [(first_line, last_line, fields)], 1-based."""
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    out, start = [], 1
    for rec in reader:
        out.append((start, reader.line_num, rec))
        start = reader.line_num + 1
    return out


def label_prefix(line: str, labels: Iterable[str] = EXPECTED_LABELS) -> str | None:
    """Return the label if the physical line begins with '<label>;' else None."""
    for lab in labels:
        if line.startswith(lab + DELIMITER):
            return lab
    return None


# --------------------------------------------------------------------------- #
# Training ingestion (line-oriented, evidence-driven)
# --------------------------------------------------------------------------- #
def _pad(fields: list[str]) -> tuple[dict[str, str], list[str]]:
    d = {k: (fields[i] if i < len(fields) else "") for i, k in enumerate(INPUT_FIELDS, start=1)
         if True}
    missing = [k for i, k in enumerate(INPUT_FIELDS, start=1) if i >= len(fields)]
    return d, missing


def ingest_train(path: str | Path, labels: Iterable[str] = EXPECTED_LABELS) -> tuple[list[TrainRecord], dict]:
    """Reconstruct logical training records.

    Rules (each logged on the record):
      clean            label-start line, 5 fields, even number of quotes.
      quarantined      * 2 fields + 1 quote  -> 'truncated_open_quote' (3/5 fields not in file)
                       * wrong field count / non-empty extra fields
                       * orphan fragment not attached to a record
      recovered        label-start line, odd quote count, >=5 fields, trailing extra fields empty,
                       no orphan fragment follows (only a stray quote artefact).
      recovered_partial  terminal line cut at a cell limit:
                       (a) orphan fragment follows -> transcription = head + fragment (one char
                           at the join is unrecoverable; NOT invented), or
                       (b) no fragment follows -> transcription truncated, remainder absent.
    """
    labels = tuple(labels)
    text = read_raw_text(path)
    lines = split_physical_lines(text)
    if not lines:
        raise ValueError(f"{path} is empty")
    header = parse_single_line(lines[0])
    if tuple(h.strip() for h in header) != TRAIN_COLUMNS:
        raise ValueError(f"Unexpected train header {header!r}; expected {TRAIN_COLUMNS!r}")

    records: list[TrainRecord] = []
    consumed: set[int] = set()
    n = len(lines)
    for idx in range(1, n):
        lineno = idx + 1
        if lineno in consumed:
            continue
        line = lines[idx]
        lab = label_prefix(line, labels)
        if lab is None:
            records.append(_mk(records, lineno, lineno, None, {}, "quarantined", "orphan_fragment",
                               "physical line does not start with '<label>;' and is not the "
                               "documented overflow of a preceding cut record", ["orphan_fragment"],
                               list(INPUT_FIELDS), [line]))
            continue
        quotes = line.count('"')
        f = parse_single_line(line)
        if quotes % 2 == 0:
            if len(f) == 5:
                fields, miss = _pad(f)
                records.append(_mk(records, lineno, lineno, lab, fields, "clean", "single_line_parse",
                                   "one physical line, 5 fields, balanced quotes", [], miss, [line], keep_raw=False))
            else:
                records.append(_mk(records, lineno, lineno, lab, {}, "quarantined", "wrong_field_count",
                                   f"{len(f)} fields with balanced quotes", ["wrong_field_count"],
                                   list(INPUT_FIELDS), [line]))
            continue
        # ---- odd number of quotes: damaged ----------------------------------
        if len(f) == 2 and quotes == 1 and line[len(lab) + 1:].startswith('"') and DELIMITER not in line[len(lab) + 1:]:
            fields, miss = _pad(f)
            fields = {k: v for k, v in fields.items()}
            records.append(_mk(records, lineno, lineno, lab, fields, "quarantined", "truncated_open_quote",
                               "only label+description present; description begins with an unterminated quote "
                               "that makes a quote-aware parser swallow following lines; sample_name/"
                               "transcription/keywords are not in the file", ["truncated_open_quote"],
                               miss, [line]))
            continue
        extras = f[5:]
        nxt = lines[idx + 1] if idx + 1 < n else None
        if len(f) >= 5 and any(x.strip() for x in extras):
            records.append(_mk(records, lineno, lineno, lab, {}, "quarantined", "extra_nonempty_fields",
                               f"{len(f)} fields, extra fields are not empty", ["extra_nonempty_fields"],
                               list(INPUT_FIELDS), [line]))
            continue
        ends_cut = line.endswith('";;;') or line.endswith('";;;;')
        if (len(f) >= 4 and ends_cut and not f[4].strip()):
            frag_ok = (nxt is not None and label_prefix(nxt, labels) is None and nxt.endswith(";;;;"))
            head = f[3]
            stripped = len(head) - len(head.rstrip('"'))
            head = head.rstrip('"')
            if frag_ok:
                frag = nxt[:-4]
                consumed.add(lineno + 1)
                fields, miss = _pad([f[0], f[1], f[2], head + frag])
                fields["keywords"] = ""
                rec = _mk(records, lineno, lineno + 1, lab, fields, "recovered_partial", "cell_limit_overflow_join",
                          f"line ends with a cut transcription (head {len(head)} chars, "
                          f"{stripped} trailing artefact quote(s) removed); next physical line is an orphan fragment "
                          f"({len(frag)} chars) ending ';;;;'. Joined without inventing characters; the joint "
                          f"likely lost 1 character (cell-limit boundary). keywords not in file.",
                          ["join_may_lose_chars", "keywords_not_in_file"], ["keywords"], [line, nxt])
                rec.flags.append(f"join_offset={len(head)}")
                records.append(rec)
            else:
                fields, miss = _pad([f[0], f[1], f[2], head])
                records.append(_mk(records, lineno, lineno, lab, fields, "recovered_partial", "truncated_transcription",
                                   f"line ends with cut transcription ({stripped} artefact quote(s) removed) and no "
                                   "overflow fragment follows; remainder of transcription and keywords absent from file",
                                   ["transcription_truncated", "keywords_not_in_file"], ["keywords"], [line]))
            continue
        if len(f) >= 5:
            fields, miss = _pad(f[:5])
            records.append(_mk(records, lineno, lineno, lab, fields, "recovered", "stray_quote_complete_record",
                               f"5 fields present with {quotes} quote chars (odd); trailing empty fields "
                               f"dropped={len(extras)}; all fields kept exactly as parsed (quote artefacts retained)",
                               ["odd_quote_count"] + (["trailing_empty_fields_dropped"] if extras else []), miss, [line]))
            continue
        records.append(_mk(records, lineno, lineno, lab, {}, "quarantined", "unclassified_damage",
                           f"odd quotes ({quotes}), {len(f)} fields; pattern not recognised", ["unclassified_damage"],
                           list(INPUT_FIELDS), [line]))
    stats = {"physical_lines": n, "header_line": 1}
    return records, stats


def _mk(records, start, end, label, fields, status, method, evidence, flags, missing, raw, keep_raw=True) -> TrainRecord:
    rid = f"TR{len(records) + 1:06d}"
    full = {k: fields.get(k, "") for k in INPUT_FIELDS}
    return TrainRecord(rid, start, end, label, full, status, method, evidence, list(flags), list(missing),
                       list(raw) if keep_raw else [], sha1("\r\n".join(raw)))


# --------------------------------------------------------------------------- #
# Test ingestion
# --------------------------------------------------------------------------- #
def ingest_test(path: str | Path) -> tuple[list[TestRecord], dict]:
    """Line-oriented test ingestion. NEVER drops a physical line (blank lines are kept & flagged).

    Header detection: the first line is a header only if its fields equal INPUT_FIELDS.
    """
    text = read_raw_text(path)
    lines = split_physical_lines(text)
    start = 0
    has_header = False
    if lines and tuple(x.strip() for x in parse_single_line(lines[0])) == INPUT_FIELDS:
        has_header, start = True, 1
    recs: list[TestRecord] = []
    for idx in range(start, len(lines)):
        line = lines[idx]
        f = parse_single_line(line) if line != "" else []
        flags: list[str] = []
        if line.strip() == "":
            flags.append("blank_physical_line")
        if line.count('"') % 2:
            flags.append("odd_quote_count")
        if f and len(f) != 4:
            flags.append(f"field_count={len(f)}")
        if len(f) > 4:
            # keep information: merge surplus into the last field with the delimiter restored
            f = f[:3] + [DELIMITER.join(f[3:])]
        miss = [k for i, k in enumerate(INPUT_FIELDS) if i >= len(f) or not f[i].strip()]
        fields = {k: (f[i] if i < len(f) else "") for i, k in enumerate(INPUT_FIELDS)}
        if len(miss) == 4:
            flags.append("all_fields_empty")
        for k in miss:
            flags.append(f"empty:{k}")
        recs.append(TestRecord(f"TE{len(recs) + 1:06d}", len(recs), idx + 1, fields,
                               "flagged" if any(not x.startswith("empty:") for x in flags) else "ok", flags, line if flags else ""))
    return recs, {"physical_lines": len(lines), "has_header": has_header}


# --------------------------------------------------------------------------- #
# Text helpers (analysis + preprocessing share these definitions)
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-'/][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def whitespace_tokens(s: str) -> list[str]:
    return s.split()


def dup_normalize(s: str) -> str:
    """Duplicate-detection normalisation: NFKC + case-fold + whitespace collapse."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", s)).strip().casefold()


# Specialty strings as they appear in `keywords` (observed, data-driven; see analysis)
KEYWORD_SPECIALTY_STRINGS: dict[str, tuple[str, ...]] = {
    "Surgery": ("surgery",),
    "Cardiovascular-Pulmonary": ("cardiovascular / pulmonary", "cardiovascular-pulmonary", "cardiovascular/pulmonary"),
    "Orthopedic": ("orthopedic",),
    "Radiology": ("radiology",),
    "General Medicine": ("general medicine",),
    "Gastroenterology": ("gastroenterology",),
    "Neurology": ("neurology",),
    "Obstetrics-Gynecology": ("obstetrics / gynecology", "obstetrics-gynecology", "obstetrics/gynecology"),
    "Neurosurgery": ("neurosurgery",),
    "Ophthalmology": ("ophthalmology",),
    "Psychiatry-Psychology": ("psychiatry / psychology", "psychiatry-psychology", "psychiatry/psychology"),
    "Dermatology": ("dermatology",),
}
ALL_SPECIALTY_STRINGS = sorted({s for v in KEYWORD_SPECIALTY_STRINGS.values() for s in v}, key=len, reverse=True)
KEYWORD_PREFIX_RE = re.compile(
    r"^\s*(?P<spec>" + "|".join(re.escape(s) for s in ALL_SPECIALTY_STRINGS) + r")\s*(?:,\s*|$)", re.IGNORECASE)


def quantile_summary(values: list[float]) -> dict[str, float]:
    import numpy as np
    if not len(values):
        return {k: float("nan") for k in ("min", "mean", "median", "std", "p5", "p25", "p50", "p75", "p90", "p95", "p99", "max")}
    a = np.asarray(values, dtype=float)
    q = np.percentile(a, [5, 25, 50, 75, 90, 95, 99])
    return {"min": a.min(), "mean": a.mean(), "median": float(np.median(a)), "std": a.std(ddof=0),
            "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4], "p95": q[5], "p99": q[6], "max": a.max()}
