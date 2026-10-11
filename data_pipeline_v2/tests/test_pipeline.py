"""Synthetic tests for clinical_common / analyze_data / preprocess_data (no real clinical data needed)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import pytest

import analyze_data as ad
import clinical_common as cc
import preprocess_data as pp

HEADER = "medical_specialty;description;sample_name;transcription;keywords"


def write(tmp: Path, name: str, lines: list[str], crlf=True) -> Path:
    p = tmp / name
    p.write_bytes(("\r\n" if crlf else "\n").join(lines).encode("utf-8"))
    return p


def rec(label="Surgery", desc=" A desc ", name=" Name ", trans="Patient has no pain.", kw="surgery, pain"):
    return f"{label};{desc};{name};{trans};{kw}"


# ----------------------------------------------------------------- ingestion
def test_quoted_semicolon_inside_field(tmp_path):
    p = write(tmp_path, "t.csv", [HEADER, rec(trans='"Dose 5 mg; repeat in 4 h ""stat"""')])
    recs, _ = cc.ingest_train(p)
    assert recs[0].status == "clean" and recs[0].fields["transcription"] == 'Dose 5 mg; repeat in 4 h "stat"'


def test_quoted_multiline_is_not_silently_merged(tmp_path):
    p = write(tmp_path, "t.csv", [HEADER, rec(trans='"line one'), 'line two";kw'])
    # a quote-aware parser sees ONE record; the line-oriented ingest refuses to guess and flags both lines
    assert len(cc.default_csv_parse(cc.read_raw_text(p))) == 2
    recs, _ = cc.ingest_train(p)
    assert all(r.status == "quarantined" for r in recs), [r.status for r in recs]


def test_unbalanced_quote_swallow_and_truncated_open_quote(tmp_path):
    lines = [HEADER, 'Neurosurgery;"Dangling description', rec(label="Radiology"), rec(label="Orthopedic")]
    p = write(tmp_path, "t.csv", lines)
    default = cc.default_csv_parse(cc.read_raw_text(p))
    assert len(default) == 2  # header + ONE swallowed mega-record
    recs, _ = cc.ingest_train(p)
    assert [r.status for r in recs] == ["quarantined", "clean", "clean"]
    assert recs[0].method == "truncated_open_quote" and recs[0].label == "Neurosurgery"


def test_wrong_field_count_is_quarantined(tmp_path):
    p = write(tmp_path, "t.csv", [HEADER, "Surgery;only;three", "Surgery;a;b;c;d;e;f"])
    recs, _ = cc.ingest_train(p)
    assert [r.status for r in recs] == ["quarantined", "quarantined"]


def test_invalid_label_is_quarantined_not_mapped(tmp_path):
    p = write(tmp_path, "t.csv", [HEADER, rec(label="Cardiology"), rec()])
    recs, _ = cc.ingest_train(p)
    assert recs[0].status == "quarantined" and recs[0].label is None and recs[0].method == "orphan_fragment"
    assert recs[1].status == "clean"


def test_empty_and_whitespace_fields_kept_and_flagged(tmp_path):
    p = write(tmp_path, "t.csv", [HEADER, "Dermatology;   ;;   ;"])
    recs, _ = cc.ingest_train(p)
    assert recs[0].status == "clean" and all(not v.strip() for v in recs[0].fields.values())


def test_cell_limit_overflow_join_does_not_invent_characters(tmp_path):
    lines = [HEADER, 'Surgery; d1 ; name ;HEAD wou";;;', "nd is open and more text.;;;;", rec(label="Radiology")]
    recs, _ = cc.ingest_train(write(tmp_path, "t.csv", lines))
    assert len(recs) == 2
    r = recs[0]
    assert r.status == "recovered_partial" and r.source_line_end == r.source_line_start + 1
    assert r.fields["transcription"] == "HEAD wound is open and more text."  # plain concatenation, artefact quote removed, nothing added
    assert "join_may_lose_chars" in r.flags and r.fields["keywords"] == ""


def test_test_file_keeps_blank_and_empty_records_in_order(tmp_path):
    lines = ["d1;n1;t1;k1", ";;;", "", "d4;n4;t4;"]
    recs, info = cc.ingest_test(write(tmp_path, "te.csv", lines))
    assert len(recs) == 4 and info["has_header"] is False
    assert [r.original_order for r in recs] == [0, 1, 2, 3]
    assert "all_fields_empty" in recs[1].flags and "blank_physical_line" in recs[2].flags
    assert "empty:keywords" in recs[3].flags


# ----------------------------------------------------------------- analysis
def _ns(tmp):
    return argparse.Namespace(near_dup_threshold=0.9, near_dup_min_threshold=0.8, near_dup_field="transcription", min_chars_dup=10, snippet_chars=60)


def _frame(rows):
    df = pd.DataFrame(rows, columns=["label", "description", "sample_name", "transcription", "keywords"])
    df.insert(0, "record_id", [f"TR{i+1:06d}" for i in range(len(df))]); df["line_start"] = range(2, 2 + len(df)); return df


def test_duplicates_label_conflict_exact_and_near(tmp_path):
    base = "the patient presented with chest pain and shortness of breath for three days no fever " * 3
    df = _frame([
        ("Surgery", "d", "s", base, "surgery, a"),
        ("Orthopedic", "d", "s", base, "orthopedic, a"),                       # exact twin, different label
        ("Radiology", "x", "y", base.replace("three", "four"), "radiology, b"),  # near duplicate
        ("Neurology", "q", "w", "completely different unrelated text about the eye and vision problems seen today", "neurology"),
        ("Neurology", "q", "w", "another unrelated text regarding skin lesions on the forearm of the patient", "neurology"),
    ])
    res = ad.duplicate_analysis(df, _ns(tmp_path), tmp_path)
    g = pd.read_csv(tmp_path / "exact_duplicate_groups.csv")
    tg = g[g.level == "normalized_exact_transcription"]
    assert len(tg) == 1 and bool(tg.label_conflict.iloc[0]) and tg["size"].iloc[0] == 2
    pairs = pd.read_csv(tmp_path / "near_duplicate_pairs.csv")
    assert len(pairs) == 3 and pairs.cosine.min() > 0.8   # (0,1) exact, (0,2),(1,2) near
    assert pairs.label_conflict.all()
    assert len(df) == 5  # analysis never deletes


def test_keyword_prefix_pattern_does_not_eat_ordinary_content():
    assert pp.strip_keyword_prefix("surgery, knee, pain") == ("knee, pain", "surgery,")
    assert pp.strip_keyword_prefix("Cardiovascular / Pulmonary, chest") == ("chest", "Cardiovascular / Pulmonary,")
    assert pp.strip_keyword_prefix("neurosurgery, spine")[1] == "neurosurgery,"
    for ordinary in ("surgical planning, surgery", "surgery planning, x", "post-surgery, pain", "knee, surgery", ""):
        assert pp.strip_keyword_prefix(ordinary) == (ordinary, None)
    assert pp.strip_keyword_prefix("surgery") == ("", "surgery")


# ----------------------------------------------------------------- preprocessing units
def test_normalization_unicode_negation_numbers():
    s = "Patient denies chest pain\u2014no fever. \u201cNo\u201d dose 2.5 mg q4h; caf\u00e9 \u2019s  BP 120/80.,Next"
    raw, low, cli = (pp.normalize_text(s, m) for m in ("raw", "lowercase", "clinical"))
    for t in (raw, low, cli):
        assert "denies" in t.lower() and "2.5 mg" in t and "120/80" in t and "caf\u00e9" in t
    assert raw == pp.normalize_text(s, "raw") and "\u2014" in raw and "\u2014" not in cli and '"No"' in cli
    assert low == low.lower() and ", Next" in cli and "  " not in raw
    with pytest.raises(ValueError):
        pp.normalize_text(s, "bogus")


def test_windows_multiple_cover_everything_and_overlap():
    words = [f"w{i}" for i in range(1000)]
    wins = pp.make_windows(words, 256, 64)
    assert len(wins) > 1 and all(len(t.split()) <= 256 for _, _, t in wins)
    rebuilt: list[str] = []
    for s, e, t in wins:
        rebuilt.extend(t.split()[len(rebuilt) - s:] if rebuilt else t.split())
    assert rebuilt == words and wins[1][0] == 256 - 64
    assert len(pp.make_windows(words[:100], 256, 64)) == 1
    with pytest.raises(ValueError):
        pp.make_windows(words, 10, 10)


def test_token_deletion_protects_negations_and_numbers_and_is_seeded():
    text = "patient denies fever no cough without pain dose 5 mg bid 120/80 abscess.,no ok " * 20
    out = pp.token_deletion(text, 0.9, random.Random(1), set(pp.DEFAULT_PROTECTED))
    o, t = out.split(), text.split()
    for tok in ("denies", "no", "without", "5", "120/80", "abscess.,no"):
        assert o.count(tok) == t.count(tok), tok          # every protected token survives, every time
    assert out == pp.token_deletion(text, 0.9, random.Random(1), set(pp.DEFAULT_PROTECTED))
    assert len(out.split()) < len(text.split())


def test_augmentation_refused_for_non_train():
    for role in ("test", "validation"):
        with pytest.raises(pp.AugmentationRefused):
            pp.assert_augmentable(pp.Dataset(role, pd.DataFrame()))
    pp.assert_augmentable(pp.Dataset("train", pd.DataFrame()))
    with pytest.raises(TypeError):                       # there is deliberately no override parameter any more
        pp.assert_augmentable(pp.Dataset("test", pd.DataFrame()), allow_eval=True)


# ----------------------------------------------------------------- end-to-end on synthetic data
@pytest.fixture()
def synth(tmp_path):
    labs = list(cc.EXPECTED_LABELS)
    long_t = " ".join(f"word{i}" for i in range(60000)) + " no fever 98.6 F"   # very long transcription
    lines = [HEADER]
    for i in range(24):
        lab = labs[i % 12]
        t = long_t if i == 0 else f"Note {i} patient denies pain; BP 120/80, dose {i} mg. No fever.,Plan caf\u00e9 follow up."
        lines.append(f'{lab}; desc {i} ; name {i} ;"{t}";{lab.lower()}, kw{i}' if i else f'{lab}; desc {i} ; name {i} ;{t};{lab.lower()}, kw{i}')
    lines.insert(5, 'Surgery;"unterminated desc')  # damage
    tr = write(tmp_path, "train.csv", lines)
    te = write(tmp_path, "test.csv", ["d1;n1;test text one;k1", ";;;", "d3;n3;;", "d4;n4;test four no pain;surgery, x"])
    usable = [r.record_id for r in cc.ingest_train(tr)[0] if r.status != "quarantined"]
    pd.DataFrame({"source_record_id": usable, "partition": ["train"] * 16 + ["validation"] * (len(usable) - 16)}).to_csv(tmp_path / "split.csv", index=False)
    return tr, te, tmp_path / "out", long_t


def run_pp(synth, extra=()):
    tr, te, out, _ = synth
    code = pp.main(["--train", str(tr), "--test", str(te), "--output-dir", str(out), "--seed", "7", "--window-size", "100", "--window-overlap", "20",
                   "--split-manifest", str(tr.parent / "split.csv"), *extra])
    return code, json.loads((out / "validation_report.json").read_text())


def test_end_to_end_order_full_text_windows_and_no_test_augmentation(synth):
    variants = ["all_fields__raw__full__no_aug", "transcription_only__raw__windows__no_aug", "all_fields__lowercase__full__token_deletion",
                "description_transcription__raw__full__no_aug", "transcription_only__clinical__head_tail__no_aug"]
    code, rep = run_pp(synth, ["--variants", *variants])
    out = synth[2]
    assert code == 0 and rep["all_checks_passed"] and rep["test_records"] == 4
    assert rep["train_documents_quarantined"] == 1
    for v in rep["variants"]:
        assert v["test_manifest_order_equals_original"] and v["test_augmented_rows"] == 0 and v["test_documents_in_manifest"] == 4
    # full text fully preserved incl. the 60k-word note
    t = pd.read_csv(out / "variants/all_fields__raw__full__no_aug/train.csv", dtype=str, keep_default_na=False)
    assert synth[3] in t.text.iloc[0] and t.text.iloc[0].endswith("98.6 F" + " " + "[KEYWORDS] " + "surgery, kw0")
    # windows: > 1 window for the long doc, ids trace back, nothing omitted
    m = pd.read_csv(out / "variants/transcription_only__raw__windows__no_aug/train_document_manifest.csv")
    assert m.n_units.max() > 600 and m.omitted_words.sum() == 0
    # test rows keep the empty record
    te = pd.read_csv(out / "variants/all_fields__raw__full__no_aug/test.csv", dtype=str, keep_default_na=False)
    assert list(te.source_record_id) == ["TE000001", "TE000002", "TE000003", "TE000004"] and list(te.original_order) == ["0", "1", "2", "3"]
    # augmentation maps back to originals, originals preserved, head_tail flagged lossy
    a = pd.read_csv(out / "variants/all_fields__lowercase__full__token_deletion/train.csv", dtype=str, keep_default_na=False)
    aug, orig = a[a.aug_type == "token_deletion"], a[a.aug_type == "none"]
    train_ids = set(pd.read_csv(synth[0].parent / "split.csv", dtype=str).query("partition == 'train'").source_record_id)
    assert len(orig) == 24 and set(aug.source_record_id) == train_ids and len(aug) == 16          # ONLY train-partition docs augmented
    assert (aug.partition == "train").all() and (aug.aug_params.str.contains('"seed": 7')).all()
    ht = pd.read_csv(out / "variants/transcription_only__clinical__head_tail__no_aug/train_document_manifest.csv")
    assert ht.text_omitted.sum() == 1 and (out / "variants/description_transcription__raw__full__no_aug/ALIAS.json").exists()


def test_backtranslation_not_run_without_backend_and_requires_approval(synth):
    code, rep = run_pp(synth, ["--variants", "all_fields__raw__full__back_translation_es"])
    v = rep["variants"][0]
    assert v["executed"] is False and v["train_rows_augmented"] == 0 and (synth[2] / "variants/all_fields__raw__full__back_translation_es/BACK_TRANSLATION_NOT_RUN.json").exists()
    with pytest.raises(SystemExit):
        pp.main(["--train", str(synth[0]), "--test", str(synth[1]), "--output-dir", str(synth[2] / "x"), "--back-translation-backend", "fake_backtranslator:translate"])


def test_backtranslation_adapter_with_approved_dummy_backend(synth, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    code, rep = run_pp(synth, ["--variants", "all_fields__raw__full__back_translation_es", "--back-translation-backend", "fake_backtranslator:translate",
                                "--i-confirm-translation-backend-approved", "--review-sample-size", "5"])
    v = rep["variants"][0]
    assert v["executed"] and v["train_rows_augmented"] == 16 and v["test_augmented_rows"] == 0
    assert (synth[2] / "variants/all_fields__raw__full__back_translation_es/back_translation_review_sample.csv").exists()


def test_synonym_disabled_without_map_and_works_with_vetted_map(synth, tmp_path):
    code, rep = run_pp(synth, ["--variants", "all_fields__raw__full__synonym"])
    assert rep["variants"][0]["executed"] is False
    m = tmp_path / "syn.json"; m.write_text(json.dumps({"follow up": "follow-up visit"}))
    code, rep = run_pp(synth, ["--variants", "all_fields__raw__full__synonym", "--synonym-map", str(m)])
    assert rep["variants"][0]["executed"] and rep["variants"][0]["train_rows_augmented"] == 16
    a = pd.read_csv(synth[2] / "variants/all_fields__raw__full__synonym/train.csv", dtype=str, keep_default_na=False)
    assert a[a.aug_type == "synonym"].text.str.contains("follow-up visit").any() and not a[a.aug_type == "none"].text.str.contains("follow-up visit").any()


def test_raw_inputs_are_not_modified_and_output_dir_must_not_contain_inputs(synth):
    tr, te, out, _ = synth
    before = (tr.read_bytes(), te.read_bytes())
    run_pp(synth, ["--variants", "all_fields__raw__full__no_aug"])
    assert (tr.read_bytes(), te.read_bytes()) == before
    with pytest.raises(SystemExit):
        pp.main(["--train", str(tr), "--test", str(te), "--output-dir", str(tr.parent)])


def test_duplicate_groups_correct_when_empty_texts_precede_groups(tmp_path):
    """Regression: empty transcriptions are skipped; group members must still get their OWN labels/ids."""
    base = "the patient presented with chest pain and shortness of breath for three days no fever " * 3
    df = _frame([("Surgery", "a", "a", "", ""), ("Surgery", "a", "a", "", ""),
                 ("Radiology", "d", "s", base, "k1"), ("Neurology", "d", "s", base, "k2"),
                 ("Orthopedic", "d", "s", "second note text about knee pain and swelling after a fall yesterday", "k3"),
                 ("Orthopedic", "d", "s", "second note text about knee pain and swelling after a fall yesterday", "k3b")])
    ad.duplicate_analysis(df, _ns(tmp_path), tmp_path)
    g = pd.read_csv(tmp_path / "exact_duplicate_groups.csv")
    tg = g[g.level == "normalized_exact_transcription"].set_index("size")
    two = g[(g.level == "normalized_exact_transcription")].sort_values("labels")
    rows = {r.labels: r for r in two.itertuples()}
    assert set(rows) == {"Neurology|Radiology", "Orthopedic"}
    assert rows["Neurology|Radiology"].record_ids == "TR000003|TR000004" and bool(rows["Neurology|Radiology"].label_conflict)
    assert rows["Orthopedic"].record_ids == "TR000005|TR000006" and not bool(rows["Orthopedic"].label_conflict)


def test_augmented_variant_without_split_manifest_is_refused(synth):
    tr, te, out, _ = synth
    with pytest.raises(SystemExit):
        pp.main(["--train", str(tr), "--test", str(te), "--output-dir", str(out), "--variants", "all_fields__raw__full__token_deletion"])
