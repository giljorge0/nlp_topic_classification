"""Tests for build_model_handoff.py (synthetic data only)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import build_model_handoff as bh
import clinical_common as cc
import preprocess_data as pp

HEADER = "medical_specialty;description;sample_name;transcription;keywords"
BOILER = "The patient tolerated the procedure well and left in stable condition."


def make_inputs(tmp: Path):
    labs = list(cc.EXPECTED_LABELS)
    lines, i = [HEADER], 0
    for rep in range(8):
        for lab in labs:
            i += 1
            if i <= 12:                       # 6 conflicting-label duplicate pairs (same note, different specialty)
                pair = (i - 1) // 2
                t = f"Shared note {pair} about knee pain after a fall. No fracture seen. {BOILER}"
            elif i % 5 == 0:                  # long note: 14 sentences, one with a negation
                t = " ".join(f"Finding number {k} was noted during the examination of region {k} in note {i}." for k in range(13)) + f" No fever was recorded in note {i}. {BOILER}"
            else:
                t = f"Note uniq{i} patient denies pain. BP 120/80 dose {i} mg. {BOILER} No fever today."
            kw = "" if i % 7 == 0 else f"{cc.KEYWORD_SPECIALTY_STRINGS[lab][0]}, kw{i}"
            lines.append(f"{lab}; desc {i} ; name {i} ;{t};{kw}")
    tr = tmp / "train.csv"; tr.write_bytes("\r\n".join(lines).encode())
    te = tmp / "test.csv"
    te.write_bytes("\r\n".join(["d1;n1;test one No pain;surgery, a", ";;;", "d3;n3;;", "d4;n4;test four;"]).encode())
    return tr, te


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("h")
    tr, te = make_inputs(tmp)
    out = tmp / "model_handoff"
    argv = ["--train", str(tr), "--test", str(te), "--output-dir", str(out), "--seed", "42", "--min-freq", "1", "--boilerplate-min-df", "10"]
    assert bh.main(argv + ["--augment", "word_dropout", "sentence_dropout", "--augment-variants", "baseline_all_fields", "clinical_text",
                           "--word-dropout-prob", "0.3", "--sentence-dropout-prob", "0.3", "--sentence-min-words", "50"]) == 0
    return {"out": out, "tr": tr, "te": te, "argv": argv, "tmp": tmp}


def rd(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def test_split_created_before_augmentation_and_pipeline_order(built):
    steps = [s["step"] for s in json.loads((built["out"] / "handoff_build_log.json").read_text())]
    assert steps.index("7_split_manifest") < steps.index("8_fit_on_train_only") < steps.index("9_transform_all_partitions") < steps.index("10_augment_train_only")
    assert steps[:7] == ["1_read_raw", "2_repair_records", "3_quarantine", "4_validate_labels_columns", "5_assign_source_ids", "6_duplicate_groups", "7_split_manifest"]
    assert steps[-1] == "11_export"
    with pytest.raises(pp.AugmentationRefused):       # augmentation function cannot run without a train-only partition map
        bh.augment_train_rows(pd.DataFrame({"source_record_id": ["TR000001"], "text": ["a b"]}), {"TR000001": "validation"}, {"TR000001": "DG000001"}, "word_dropout", 0.05, 1, 1, set())


def test_same_record_same_partition_across_variants_and_matches_manifest(built):
    m = rd(built["out"] / "splits/split_manifest_primary_stratified.csv"); part = dict(zip(m.source_record_id, m.partition))
    for v in bh.VARIANTS:
        d = built["out"] / "datasets" / v
        for f, want in (("train.csv", "train"), ("validation.csv", "validation")):
            x = rd(d / f)
            assert set(x.partition) == {want} and all(part[s] == want for s in x.source_record_id), (v, f)
        assert set(rd(d / "train.csv").source_record_id) | set(rd(d / "validation.csv").source_record_id) == set(m.source_record_id)


def test_augmented_files_only_for_requested_variants_and_only_train_rows(built):
    m = rd(built["out"] / "splits/split_manifest_primary_stratified.csv"); part = dict(zip(m.source_record_id, m.partition))
    for v in bh.VARIANTS:
        d = built["out"] / "datasets" / v
        have = sorted(p.name for p in d.glob("train_augmented_*"))
        want = ["train_augmented_sentence_dropout.csv", "train_augmented_word_dropout.csv"] if v in ("baseline_all_fields", "clinical_text") else []
        assert have == want, (v, have)
    for method in ("word_dropout", "sentence_dropout"):
        a = rd(built["out"] / "datasets/baseline_all_fields" / f"train_augmented_{method}.csv")
        aug, orig = a[a.augmentation_type != "none"], a[a.augmentation_type == "none"]
        base = rd(built["out"] / "datasets/baseline_all_fields/train.csv")
        assert list(orig.example_id) == list(base.example_id) and list(orig.text) == list(base.text)       # originals retained, unchanged, same order
        assert len(aug) > 0 and set(aug.augmentation_type) == {method} and set(a.partition) == {"train"}
        assert aug.source_record_id.isin(orig.source_record_id).all() and all(part[s] == "train" for s in aug.source_record_id)   # source-ID preservation + split restriction
        assert a.example_id.is_unique and aug.example_id.str.contains("__(?:wdrop|sdrop)\\d\\d$", regex=True).all()
        assert (aug.augmentation_probability == "0.3").all() and (aug.augmentation_seed != "").all() and aug.augmentation_params.str.contains(method).all()
        assert (aug.text.str.strip() != "").all()                                                         # never empty
    chk = json.loads((built["out"] / "augmentation_check.json").read_text())
    assert set(chk) == {f"{v}/{m}" for v in ("baseline_all_fields", "clinical_text") for m in ("word_dropout", "sentence_dropout")}
    assert all(c["passed"] for c in chk.values())


def test_validation_and_test_files_unchanged_by_augmentation(built, tmp_path):
    plain = tmp_path / "plain" / "model_handoff"
    assert bh.main(built["argv"][:4] + ["--output-dir", str(plain), "--seed", "42", "--min-freq", "1", "--boilerplate-min-df", "10", "--no-zip"]) == 0
    for v in bh.VARIANTS:
        for f in ("train.csv", "validation.csv", "test.csv", "vocab_train_only.json"):
            assert (built["out"] / "datasets" / v / f).read_bytes() == (plain / "datasets" / v / f).read_bytes(), (v, f)
    assert not list(plain.rglob("train_augmented_*"))                                 # opt-in: nothing augmented by default


def test_augmentation_is_deterministic_for_fixed_seed(built, tmp_path):
    out2 = tmp_path / "again" / "model_handoff"
    assert bh.main(built["argv"][:4] + ["--output-dir", str(out2), "--seed", "42", "--min-freq", "1", "--boilerplate-min-df", "10", "--no-zip",
                   "--augment", "word_dropout", "sentence_dropout", "--augment-variants", "baseline_all_fields", "clinical_text",
                   "--word-dropout-prob", "0.3", "--sentence-dropout-prob", "0.3", "--sentence-min-words", "50"]) == 0
    for v in ("baseline_all_fields", "clinical_text"):
        for m in ("word_dropout", "sentence_dropout"):
            f = f"datasets/{v}/train_augmented_{m}.csv"
            assert (built["out"] / f).read_bytes() == (out2 / f).read_bytes()
    rows = pd.DataFrame({"example_id": ["TR1"], "source_record_id": ["TR1"], "medical_specialty": ["Surgery"], "partition": ["train"], "augmentation_type": ["none"],
                         "augmentation_seed": [""], "augmentation_params": [""], "text": ["[TRANSCRIPTION] " + " ".join(f"{chr(97 + i % 26)}{chr(97 + i // 26)}x" for i in range(60))]})
    args = (rows, {"TR1": "train"}, {"TR1": "DG1"}, "word_dropout", 0.3, 1, 7, set(pp.DEFAULT_PROTECTED))
    a1, _ = bh.augment_train_rows(*args); a2, _ = bh.augment_train_rows(*args); a3, _ = bh.augment_train_rows(rows, {"TR1": "train"}, {"TR1": "DG1"}, "word_dropout", 0.3, 1, 8, set(pp.DEFAULT_PROTECTED))
    assert list(a1.text) == list(a2.text) and list(a1.text) != list(a3.text)         # same seed -> same output, other seed -> different


def test_word_dropout_protects_negations_numbers_markers_and_never_empties():
    prot = set(pp.DEFAULT_PROTECTED)
    txt = "[DESCRIPTION] patient denies pain , no fever BP 120/80 without distress negative for abscess.,no non-contributory w/o issue " + " ".join(f"w{i}" for i in range(40))
    for seed in range(40):
        out = bh.word_dropout(txt, 0.95, __import__("random").Random(seed), prot)
        o, t = out.split(), txt.split()
        for tok in ("[DESCRIPTION]", "denies", "no", "without", "negative", "120/80", "abscess.,no", "non-contributory", "w/o"):
            assert o.count(tok) == t.count(tok), (seed, tok)
        assert len(o) >= len(t) - int(0.5 * (len(t) - 9))
    tiny = bh.word_dropout("[TRANSCRIPTION] ab cd", 0.99, __import__("random").Random(1), prot)
    assert tiny.strip() and "[TRANSCRIPTION]" in tiny                                 # a short note is never emptied
    assert bh.word_dropout("[TRANSCRIPTION] solo", 0.99, __import__("random").Random(1), prot) == "[TRANSCRIPTION] solo"
    only_prot = "[TRANSCRIPTION] no not without 12 34"
    assert bh.word_dropout(only_prot, 0.99, __import__("random").Random(1), prot) == only_prot


def test_word_dropout_is_word_level_not_character_level():
    txt = "[TRANSCRIPTION] " + " ".join(["hypertension", "myocardial", "infarction", "dyspnea"] * 20)
    out = bh.word_dropout(txt, 0.4, __import__("random").Random(2), set(pp.DEFAULT_PROTECTED))
    assert set(out.split()) <= set(txt.split()) and len(out.split()) < len(txt.split())      # every surviving token is an intact original word


def test_sentence_dropout_rules():
    prot = set(pp.DEFAULT_PROTECTED); rnd = __import__("random").Random
    long = "[DESCRIPTION] d [TRANSCRIPTION] " + " ".join(f"Finding number {k} was noted during the examination of region {k}." for k in range(12)) + " No fever was recorded. [KEYWORDS] kw"
    out = bh.sentence_dropout(long, 0.9, rnd(1), prot, 8, 50)
    assert out is not None and out.startswith("[DESCRIPTION] d [TRANSCRIPTION]") and out.endswith("[KEYWORDS] kw") and "No fever was recorded." in out
    kept = [int(x) for x in __import__("re").findall(r"Finding number (\d+)", out)]
    assert kept == sorted(kept) and 0 < len(kept) < 12 and len(kept) >= 12 - 3 - 0      # order kept; at most 25% of 13 sentences dropped
    short = "[TRANSCRIPTION] One. Two. Three."
    assert bh.sentence_dropout(short, 0.9, rnd(1), prot, 8, 50) is None                  # too short -> not eligible
    assert bh.sentence_dropout("[DESCRIPTION] only description", 0.9, rnd(1), prot) is None


def test_group_aware_augmentation_copies_stay_with_their_training_group(built, tmp_path):
    out = tmp_path / "ga" / "model_handoff_group_aware"
    assert bh.main(built["argv"][:4] + ["--output-dir", str(out), "--seed", "42", "--min-freq", "1", "--boilerplate-min-df", "10", "--no-zip", "--split", "group_aware",
                   "--augment", "word_dropout", "--word-dropout-prob", "0.3"]) == 0
    g = rd(out / "splits/split_manifest_optional_group_aware.csv"); part = dict(zip(g.source_record_id, g.partition)); grp = dict(zip(g.source_record_id, g.duplicate_group_id))
    a = rd(out / "datasets/baseline_all_fields/train_augmented_word_dropout.csv")
    aug = a[a.augmentation_type == "word_dropout"]
    val_groups = {grp[s] for s, p in part.items() if p == "validation"}
    assert len(aug) > 0 and all(part[s] == "train" for s in aug.source_record_id)
    assert all(grp[s] == dg for s, dg in zip(aug.source_record_id, aug.duplicate_group_id)) and not (set(aug.duplicate_group_id) & val_groups)
    assert set(rd(out / "datasets/baseline_all_fields/validation.csv").source_record_id) == {s for s, p in part.items() if p == "validation"}
    assert json.loads((out / "augmentation_check.json").read_text())["baseline_all_fields/word_dropout"]["augmented_groups_disjoint_from_validation_groups"] is True


def test_augmentation_refuses_validation_rows_without_override():
    rows = pd.DataFrame({"example_id": ["TR1"], "source_record_id": ["TR1"], "text": ["a b c"]})
    with pytest.raises(pp.AugmentationRefused):
        bh.augment_train_rows(rows, {"TR1": "validation"}, {"TR1": "DG1"}, "word_dropout", 0.05, 1, 1, set())
    with pytest.raises(TypeError):
        bh.augment_train_rows(rows, {"TR1": "train"}, {"TR1": "DG1"}, "word_dropout", 0.05, 1, 1, set(), allow_validation=True)
    with pytest.raises(ValueError):
        bh.augment_train_rows(rows, {"TR1": "train"}, {"TR1": "DG1"}, "char_deletion", 0.05, 1, 1, set())       # character-level methods do not exist


def test_compression_never_removes_numbers_negations_or_list_markers_and_is_logged(built):
    t = "1. First problem was a large left sided pleural effusion noted. 2. Second. 3. Third. 2. Second. Mr. Smith agreed. Mr. Smith agreed. " + \
        "The patient was seen by the cardiology team and was feeling well today. The patient was seen by the cardiology team and was feeling well today."
    out, removed = bh.compress_text(t, {}, 6, set(pp.DEFAULT_PROTECTED))
    assert [k for k, _ in removed] == ["within_note_repeat"] and out.count("2.") == 2 and out.count("Mr.") == 2 and out.count("cardiology team") == 1
    assert bh.compress_text("No fever was recorded at all today. No fever was recorded at all today.", {}, 6, set(pp.DEFAULT_PROTECTED)) == ("No fever was recorded at all today. No fever was recorded at all today.", [])
    d = built["out"] / "datasets/compressed_unique_text"
    log = rd(d / "compression_log.csv"); rem = rd(d / "compression_removed_sentences.csv")
    assert len(log) == 24 * 4 + 4 and set(log.partition) == {"train", "validation", "test"}
    assert (log.text_changed == "True").sum() > 0 and len(rem) == (log.n_removed_repeat.astype(int) + log.n_removed_rule.astype(int)).sum()
    assert set(rem.kind) <= {"train_boilerplate_rule", "within_note_repeat"} and not rem[rem.kind == "within_note_repeat"].sentence.str.contains(r"\d|\bno\b", case=False, regex=True).any()


def test_class_counts_match_manifest(built):
    m = rd(built["out"] / "splits/split_manifest_primary_stratified.csv"); c = rd(built["out"] / "weights/class_counts.csv")
    for r in c.itertuples():
        assert int(r.primary_train_count) == ((m.partition == "train") & (m.medical_specialty == r.medical_specialty)).sum()
        assert int(r.primary_validation_count) == ((m.partition == "validation") & (m.medical_specialty == r.medical_specialty)).sum()
    meta = json.loads((built["out"] / "splits/split_metadata_primary_stratified.json").read_text())
    assert sum(v["train"] for v in meta["per_class_counts"].values()) == meta["records_train"]
    assert sum(v["validation"] for v in meta["per_class_counts"].values()) == meta["records_validation"]
    assert meta["seed"] == 42 and meta["validation_fraction"] == 0.2 and meta["strategy"] == "stratified_random"


def test_vocabulary_is_fitted_on_train_only(built):
    for v in bh.VARIANTS:
        d = built["out"] / "datasets" / v
        vocab = json.loads((d / "vocab_train_only.json").read_text())
        tr_tok = {t for s in rd(d / "train.csv").text for t in bh.tokenize(s)}
        va_tok = {t for s in rd(d / "validation.csv").text for t in bh.tokenize(s)}
        te_tok = {t for s in rd(d / "test.csv").text for t in bh.tokenize(s)}
        assert set(vocab) - {"<pad>", "<unk>"} == tr_tok                       # min_freq=1 -> exactly the training tokens
        assert not ((va_tok | te_tok) - tr_tok) & set(vocab)
        assert any(t.startswith("uniq") for t in va_tok - tr_tok)              # validation-only tokens exist and are OUT of the vocab
        assert vocab["<pad>"] == 0 and vocab["<unk>"] == 1


def test_boilerplate_rules_use_training_partition_only(built):
    rules = json.loads((built["out"] / "datasets/compressed_unique_text/boilerplate_rules_train_only.json").read_text())["rules"]
    m = rd(built["out"] / "splits/split_manifest_primary_stratified.csv"); tr_ids = set(m[m.partition == "train"].source_record_id)
    src = rd(built["out"] / "audit/cleaned_train_source_fields.csv")
    n_train_with = sum(BOILER.lower() in t.lower() for t in src[src.source_record_id.isin(tr_ids)].transcription)
    assert rules[BOILER.lower()] == n_train_with                                # df counted over TRAIN docs only
    n_all = sum(BOILER.lower() in t.lower() for t in src.transcription)
    assert n_train_with < n_all
    # API-level: rules fitted from a list that excludes validation text never see it
    assert bh.fit_boilerplate([BOILER] * 3, 5, 6, set(pp.DEFAULT_PROTECTED)) == {}
    ct = rd(built["out"] / "datasets/compressed_unique_text/train.csv").text
    assert not ct.str.contains("tolerated the procedure").any()
    clin = rd(built["out"] / "datasets/clinical_text/train.csv").text
    assert clin.str.contains("tolerated the procedure").all()
    va = rd(built["out"] / "datasets/compressed_unique_text/validation.csv").text
    assert not va.str.contains("tolerated the procedure").any()                 # fixed rule applied to validation too
    assert ct.str.contains("denies").any() and ct.str.contains("120/80").any()  # negation / numbers survive compression


def test_test_order_and_count_preserved_in_every_variant(built):
    raw_n = len(cc.split_physical_lines(cc.read_raw_text(built["te"])))
    for v in bh.VARIANTS:
        t = rd(built["out"] / "datasets" / v / "test.csv")
        assert len(t) == raw_n == 4 and list(t.test_record_id) == ["TE000001", "TE000002", "TE000003", "TE000004"] and list(t.original_order) == ["0", "1", "2", "3"]


def test_keyword_prefix_policy_per_variant(built):
    base = rd(built["out"] / "datasets/baseline_all_fields/train.csv").text
    diag = rd(built["out"] / "datasets/diagnostic_original_keywords/train.csv").text
    assert not base.str.contains(r"\[KEYWORDS\] (?:surgery|orthopedic|radiology|neurology),", regex=True).any()
    assert diag.str.contains(r"\[KEYWORDS\] (?:surgery|orthopedic|radiology|neurology),", regex=True).any()
    assert "[KEYWORDS]" not in rd(built["out"] / "datasets/clinical_text/train.csv").text.iloc[0]


def test_conflicting_labels_never_resolved(built):
    src = rd(built["out"] / "audit/cleaned_train_source_fields.csv"); mp = rd(built["out"] / "mappings/record_mapping.csv")
    assert len(src) == 96 and len(mp) == 96                                      # nothing dropped
    raw_labels = {}
    for ln in cc.split_physical_lines(cc.read_raw_text(built["tr"]))[1:]:
        raw_labels[ln.split(";")[0]] = raw_labels.get(ln.split(";")[0], 0) + 1
    assert src.medical_specialty.value_counts().to_dict() == raw_labels            # labels untouched
    conf = mp[(mp.transcription_group_size.astype(int) > 1) & (mp.transcription_group_label_conflict == "True")]
    assert conf.duplicate_group_id.nunique() == 6 and len(conf) == 12
    ov = json.loads((built["out"] / "splits/split_overlap_report.json").read_text())
    assert "validation_twin_in_train_only_different_labels" in ov["primary_stratified"]["transcription"]


def test_primary_and_group_aware_manifests_are_distinct(built):
    s = built["out"] / "splits"
    p, g = rd(s / "split_manifest_primary_stratified.csv"), rd(s / "split_manifest_optional_group_aware.csv")
    assert set(p.split_strategy) == {"stratified_random"} and set(g.split_strategy) == {"duplicate_group_aware"}
    assert "PROPOSED" in json.loads((s / "split_metadata_primary_stratified.json").read_text())["status"]
    assert "OPTIONAL" in json.loads((s / "split_metadata_optional_group_aware.json").read_text())["status"]
    gm = g.merge(g, on="duplicate_group_id")
    assert gm.groupby("duplicate_group_id").partition_x.nunique().max() == 1      # groups never straddle in the group-aware split
    assert set(p.columns) >= {"source_record_id", "partition", "medical_specialty", "duplicate_group_id", "split_strategy", "split_seed"}
    assert set(p.partition) == {"train", "validation"}


def test_split_reproducible_reused_and_refuses_silent_resplit(built, tmp_path):
    out = built["out"]; before = (out / "splits/split_manifest_primary_stratified.csv").read_bytes()
    assert bh.main(built["argv"] + ["--no-zip"]) == 0
    assert (out / "splits/split_manifest_primary_stratified.csv").read_bytes() == before        # reused, unchanged
    out2 = tmp_path / "again"
    assert bh.main(built["argv"][:4] + ["--output-dir", str(out2), "--seed", "42", "--min-freq", "1", "--boilerplate-min-df", "10", "--no-zip"]) == 0
    assert (out2 / "splits/split_manifest_primary_stratified.csv").read_bytes() == before        # same data+config -> same split
    with pytest.raises(SystemExit):                                                               # different seed vs existing manifest
        bh.main(built["argv"][:4] + ["--output-dir", str(out), "--seed", "7", "--min-freq", "1", "--no-zip"])


def test_class_weights_formula_finite_positive_mean_one():
    counts = {l: c for l, c in zip(cc.EXPECTED_LABELS, [890, 308, 300, 222, 211, 195, 184, 135, 78, 73, 41, 25])}
    t = bh.class_weight_table(counts, [0, 0.25, 0.5, 0.75, 1.0])
    for name, g in t.groupby("strategy"):
        w = g.weight.values
        assert np.all(np.isfinite(w)) and np.all(w > 0) and math.isclose(w.mean(), 1.0, rel_tol=1e-9), name
    w = lambda s: t[t.strategy == s].sort_values("class_index").weight.values
    assert np.allclose(w("interpolated_alpha_0"), 1.0) and np.allclose(w("none"), 1.0)
    assert np.allclose(w("interpolated_alpha_1"), w("inverse_frequency"))
    n = np.array(list(counts.values()), float); inv = 1 / (n / n.sum()); assert np.allclose(w("inverse_frequency"), inv / inv.mean())
    isq = 1 / np.sqrt(n / n.sum()); assert np.allclose(w("inverse_sqrt_frequency"), isq / isq.mean())
    assert w("inverse_frequency")[-1] > w("inverse_frequency")[0]                 # rare class weighs more
    with pytest.raises(ValueError):
        bh.class_weight_table({**counts, "Dermatology": 0}, [0.5])
    with pytest.raises(ValueError):
        bh.class_weight_table(counts, [1.5])


def test_weight_files_and_group_weights(built):
    w = rd(built["out"] / "weights/class_weights_long.csv")
    for (b, s), g in w.groupby(["basis", "strategy"]):
        x = g.weight.astype(float).values
        assert np.isfinite(x).all() and (x > 0).all() and math.isclose(x.mean(), 1, rel_tol=1e-9)
    gw = rd(built["out"] / "weights/duplicate_group_sample_weights_primary_train.csv").group_weight.astype(float)
    assert np.isfinite(gw).all() and (gw > 0).all() and math.isclose(gw.mean(), 1, rel_tol=1e-9)
    assert json.loads((built["out"] / "weights/label_to_index.json").read_text()) == {l: i for i, l in enumerate(cc.EXPECTED_LABELS)}


def test_validation_subset_flags_exist(built):
    v = rd(built["out"] / "splits/validation_subsets_primary.csv")
    m = rd(built["out"] / "splits/split_manifest_primary_stratified.csv")
    assert set(v.source_record_id) == set(m[m.partition == "validation"].source_record_id)
    assert {"complete_input_label_conflict", "complete_input_consistent_duplicate", "no_complete_input_twin_in_train"} <= set(v.columns)


def test_manifest_records_outputs_checksums_and_detects_tampering(built, tmp_path):
    out = built["out"]
    man = json.loads((out / "handoff_manifest.json").read_text())
    for rel in ("splits/split_manifest_primary_stratified.csv", "datasets/baseline_all_fields/train.csv", "weights/class_weights_long.csv", "MODEL_HANDOFF_README.md"):
        assert rel in man["files"] and len(man["files"][rel]["sha256"]) == 64
    assert "baseline_all_fields" in man["configuration"]["variants"] and man["configuration"]["args"]["seed"] == 42
    assert bh.verify_manifest(out) == []
    victim = out / "weights/label_to_index.json"; orig = victim.read_bytes()
    victim.write_bytes(orig + b" "); assert bh.verify_manifest(out) == ["checksum mismatch weights/label_to_index.json"]; victim.write_bytes(orig)
    assert (out.parent / "model_handoff.zip").exists()


def test_documents_exist(built):
    for f in ("MODEL_HANDOFF_README.md", "CLASS_WEIGHTING_EXPERIMENTS.md", "DATA_DICTIONARY.md", "AUDIT_SUMMARY.md"):
        assert (built["out"] / f).stat().st_size > 500
