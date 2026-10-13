"""Integration tests for train.py hand-off mode (needs torch + the real model_handoff folders).

    HANDOFF_DIR=../model_handoff GROUP_AWARE_DIR=../model_handoff_group_aware python -m pytest -q tests
Optional: RAW_TRAIN_CSV=/path/train.csv and train_ORIGINAL.py next to train.py enable the raw-mode regression test.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent
HANDOFF = Path(os.environ.get("HANDOFF_DIR", MODELS.parent / "model_handoff"))
GROUP = Path(os.environ.get("GROUP_AWARE_DIR", MODELS.parent / "model_handoff_group_aware"))
pytestmark = pytest.mark.skipif(not (HANDOFF / "handoff_manifest.json").exists(), reason="model_handoff folder not found (set HANDOFF_DIR)")


def _load():
    spec = importlib.util.spec_from_file_location("train_mod", MODELS / "train.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


T = _load()
VARIANTS = ["baseline_all_fields", "clinical_text", "compressed_unique_text", "diagnostic_original_keywords"]


def ns(dataset, split="stratified", handoff=HANDOFF, group=None, extras=None):
    return types.SimpleNamespace(dataset=dataset, split=split, handoff_dir=str(handoff), group_aware_dir=str(group) if group else None, extras_dir=extras)


def parse(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return T.resolve_args(T.get_args())


def run(*argv, check=True):
    p = subprocess.run([sys.executable, str(MODELS / "train.py"), *argv], capture_output=True, text=True, cwd=MODELS)
    if check:
        assert p.returncode == 0, p.stderr[-1500:]
    return p


def read(path):
    with open(path, encoding="utf-8", newline="") as s:
        return list(csv.DictReader(s))


# ------------------------------------------------------------------ models untouched
def test_model_files_are_untouched():
    for line in (HERE / "model_files.sha256").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((MODELS / name).read_bytes()).hexdigest() == digest, name


# ------------------------------------------------------------------ dataset sizes / loading
@pytest.mark.parametrize("variant", VARIANTS)
def test_stratified_dataset_sizes_for_every_variant(variant):
    d = T.load_handoff_training_data(ns(variant))
    assert (len(d["train"]), len(d["validation"]), len(d["test"])) == (2129, 533, 398)
    assert [r["example_id"] for r in d["test"]][:2] == ["TE000001", "TE000002"]
    assert all(r["augmentation_type"] == "none" for r in d["train"] + d["validation"])


def test_augmented_datasets_append_copies_and_leave_validation_and_test_unchanged():
    base = T.load_handoff_training_data(ns("baseline_all_fields"))
    for name, copies in (("baseline_all_fields_word_dropout", 2125), ("baseline_all_fields_sentence_dropout", 1821)):
        d = T.load_handoff_training_data(ns(name))
        assert len(d["train"]) == 2129 + copies and (len(d["validation"]), len(d["test"])) == (533, 398)
        assert [r["example_id"] for r in d["validation"]] == [r["example_id"] for r in base["validation"]]
        assert [r["_text"] for r in d["test"]] == [r["_text"] for r in base["test"]]
        orig = [r for r in d["train"] if r["augmentation_type"] == "none"]
        assert [(r["example_id"], r["_text"]) for r in orig] == [(r["example_id"], r["_text"]) for r in base["train"]]
        train_ids = {r["source_record_id"] for r in base["train"]}
        assert all(r["source_record_id"] in train_ids and r["partition"] == "train" for r in d["train"])


@pytest.mark.skipif(not GROUP.exists(), reason="group-aware folder not found")
def test_group_aware_sizes_and_separation():
    meta = json.loads((GROUP / "splits/split_metadata_optional_group_aware.json").read_text())
    d = T.load_handoff_training_data(ns("baseline_all_fields", "group_aware", group=GROUP))
    assert (len(d["train"]), len(d["validation"]), len(d["test"])) == (meta["records_train"], meta["records_validation"], 398)
    assert d["root"] == GROUP
    prim = T.load_handoff_training_data(ns("baseline_all_fields"))
    assert {r["example_id"] for r in d["validation"]} != {r["example_id"] for r in prim["validation"]}
    aug = T.load_handoff_training_data(ns("baseline_all_fields_word_dropout", "group_aware", group=GROUP))
    val_groups = {r["source_record_id"] for r in aug["validation"]}
    assert not ({r["source_record_id"] for r in aug["train"]} & val_groups)


@pytest.mark.skipif(not GROUP.exists(), reason="group-aware folder not found")
def test_stratified_and_group_aware_folders_are_never_mixed():
    with pytest.raises(ValueError, match="never mixed"):
        T.load_handoff_training_data(ns("baseline_all_fields", "stratified", handoff=GROUP))
    with pytest.raises(ValueError, match="never mixed"):
        T.load_handoff_training_data(ns("baseline_all_fields", "group_aware", group=HANDOFF))


# ------------------------------------------------------------------ shuffled / tampered files
def _mini_root(tmp_path: Path) -> Path:
    root = tmp_path / "model_handoff"
    for rel in ("handoff_manifest.json", "splits/split_manifest_primary_stratified.csv", "audit/cleaned_test_source_fields.csv",
                *[f"datasets/baseline_all_fields/{f}" for f in ("train.csv", "validation.csv", "test.csv")]):
        (root / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(HANDOFF / rel, root / rel)
    return root


def _rewrite(root: Path, rel: str, rows, fields):
    with open(root / rel, "w", encoding="utf-8", newline="") as s:
        w = csv.DictWriter(s, fieldnames=fields, quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(rows)
    m = json.loads((root / "handoff_manifest.json").read_text())      # re-hash so ONLY the order check can fire
    m["files"][rel]["sha256"] = T.sha256_file(root / rel); (root / "handoff_manifest.json").write_text(json.dumps(m))


def test_shuffled_validation_rejected(tmp_path):
    root = _mini_root(tmp_path); rel = "datasets/baseline_all_fields/validation.csv"
    rows = read(root / rel); fields = list(rows[0]); rows[0], rows[1] = rows[1], rows[0]
    _rewrite(root, rel, rows, fields)
    with pytest.raises(ValueError, match="NOT in the manifest order"):
        T.load_handoff_training_data(ns("baseline_all_fields", handoff=root))


def test_shuffled_test_rejected(tmp_path):
    root = _mini_root(tmp_path); rel = "datasets/baseline_all_fields/test.csv"
    rows = read(root / rel); fields = list(rows[0]); rows[5], rows[6] = rows[6], rows[5]
    _rewrite(root, rel, rows, fields)
    with pytest.raises(ValueError, match="NOT in the manifest order|original_order"):
        T.load_handoff_training_data(ns("baseline_all_fields", handoff=root))
    with pytest.raises(ValueError, match="hand-off test validation failed"):
        T.load_handoff_test(root, "baseline_all_fields", "stratified")


def test_validation_with_missing_rows_or_wrong_partition_rejected(tmp_path):
    root = _mini_root(tmp_path); rel = "datasets/baseline_all_fields/validation.csv"
    rows = read(root / rel); fields = list(rows[0]); _rewrite(root, rel, rows[:-3], fields)
    with pytest.raises(ValueError, match="records differ"):
        T.load_handoff_training_data(ns("baseline_all_fields", handoff=root))


def test_modified_file_fails_checksum(tmp_path):
    root = _mini_root(tmp_path); p = root / "datasets/baseline_all_fields/train.csv"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        T.load_handoff_training_data(ns("baseline_all_fields", handoff=root))


def test_validate_function_rejects_validation_rows_marked_augmented_or_train():
    d = T.load_handoff_training_data(ns("baseline_all_fields"))
    bad = [dict(r) for r in d["validation"]]; bad[0]["augmentation_type"] = "word_dropout"
    with pytest.raises(ValueError, match="never augmented"):
        T.validate_handoff_split(HANDOFF, "stratified", d["train"], bad, d["test"])
    leak = [dict(r) for r in d["train"]]; leak[0]["partition"] = "validation"
    with pytest.raises(ValueError, match="partition"):
        T.validate_handoff_split(HANDOFF, "stratified", leak, d["validation"], d["test"])


# ------------------------------------------------------------------ class weights
KEYS = {"inverse_sqrt": "inverse_sqrt_frequency", "inverse_frequency": "inverse_frequency", "none": "none"}


def test_weights_are_aligned_by_class_name_not_position():
    l2i = json.loads((HANDOFF / "weights/label_to_index.json").read_text())
    table = json.loads((HANDOFF / "weights/class_weights_primary_split.json").read_text())
    assert list(l2i) != list(T.LABELS) and sorted(T.LABELS) == list(T.LABELS)            # hand-off order differs from this script's alphabetical order
    for choice, key in KEYS.items():
        w = T.load_handoff_class_weights(HANDOFF, key)
        for name in T.LABELS:
            assert w[T.LABELS.index(name)] == pytest.approx(table[key][name])
        assert w.mean() == pytest.approx(1.0) and np.all(w > 0) and np.all(np.isfinite(w))
    naive = np.array([table["inverse_frequency"][n] for n in l2i])           # what positional use of the hand-off order would give
    assert not np.allclose(naive, T.load_handoff_class_weights(HANDOFF, "inverse_frequency"))
    inv = T.load_handoff_class_weights(HANDOFF, "inverse_frequency")
    assert inv[T.LABELS.index("Dermatology")] == inv.max() and inv[T.LABELS.index("Surgery")] == inv.min()


def test_loss_receives_weights_in_model_index_order():
    args = types.SimpleNamespace(resolved_class_weights=T.load_handoff_class_weights(HANDOFF, "inverse_frequency"), class_weighting="none")
    crit = T.training_loss(args, [], T.torch.device("cpu"))
    table = json.loads((HANDOFF / "weights/class_weights_primary_split.json").read_text())["inverse_frequency"]
    for name in ("Dermatology", "Surgery", "Neurology"):
        assert float(crit.weight[T.LABELS.index(name)]) == pytest.approx(table[name], rel=1e-6)


@pytest.mark.parametrize("choice,alpha", [("none", None), ("inverse_sqrt", None), ("inverse_frequency", None), ("interpolated", 0.25), ("interpolated", 0.5), ("interpolated", 1.0)])
def test_recomputed_weights_equal_provided_weights_for_unaugmented_train(choice, alpha):
    d = T.load_handoff_training_data(ns("baseline_all_fields"))
    counts = np.bincount([T.LABELS.index(r["medical_specialty"]) for r in d["train"]], minlength=12).astype(float)
    mine = T.compute_class_weights(counts, choice, alpha)
    assert np.allclose(mine, T.load_handoff_class_weights(HANDOFF, T.weight_key(choice, alpha)), atol=1e-6)


def test_augmented_training_needs_recomputed_weights():
    args = types.SimpleNamespace(class_weight="inverse_frequency", class_weight_alpha=None, class_weight_source="train_rows")
    d = T.load_handoff_training_data(ns("baseline_all_fields_word_dropout"))
    recomputed = T.resolve_handoff_class_weights(args, HANDOFF, d["train"])
    provided = T.load_handoff_class_weights(HANDOFF, "inverse_frequency")
    assert not np.allclose(recomputed, provided) and recomputed.mean() == pytest.approx(1.0)   # copies shift class frequencies
    args.class_weight_source = "handoff_file"
    assert np.allclose(T.resolve_handoff_class_weights(args, HANDOFF, d["train"]), provided)   # (prints a warning)


def test_zero_count_class_rejected():
    counts = np.ones(12) * 5; counts[3] = 0
    with pytest.raises(ValueError, match="no training rows"):
        T.compute_class_weights(counts, "inverse_frequency", None)


# ------------------------------------------------------------------ argument rules / max length
def test_raw_mode_defaults_unchanged(monkeypatch):
    a = parse(["train", "--model", "birnn", "--train-csv", "x.csv", "--output-dir", "o"], monkeypatch)
    assert (a.mode, a.max_length, a.split, a.val_fraction, a.class_weighting) == ("raw", 512, "grouped", 0.2, "none") and a.fields == list(T.FIELDS)


def test_handoff_max_length_default_is_1024_explicit_and_2048_allowed(monkeypatch):
    base = ["train", "--model", "birnn", "--handoff-dir", str(HANDOFF), "--dataset", "baseline_all_fields", "--output-dir", "o"]
    a = parse(base, monkeypatch); assert (a.max_length, a.max_length_source, a.split, a.class_weight) == (1024, "handoff default", "stratified", "none")
    a = parse(base + ["--max-length", "2048"], monkeypatch); assert (a.max_length, a.max_length_source) == (2048, "explicit")


@pytest.mark.parametrize("extra,msg", [
    (["--split", "grouped"], "raw-mode splitter"), (["--fields", "keywords"], "cannot be combined"), (["--strip-keyword-prefix"], "cannot be combined"),
    (["--val-fraction", "0.3"], "cannot be combined"), (["--class-weighting", "sqrt_inv"], "use --class-weight"),
    (["--class-weight", "interpolated"], "alpha"), (["--class-weight", "inverse_frequency", "--class-weight-alpha", "0.5"], "only used"),
    (["--train-csv", "t.csv"], "cannot be combined")])
def test_invalid_handoff_flag_combinations_fail_loudly(monkeypatch, extra, msg):
    base = ["train", "--model", "birnn", "--handoff-dir", str(HANDOFF), "--dataset", "baseline_all_fields", "--output-dir", "o"]
    with pytest.raises(ValueError, match=msg):
        parse(base + extra, monkeypatch)


def test_handoff_requires_dataset_and_raw_rejects_handoff_flags(monkeypatch):
    with pytest.raises(ValueError, match="--dataset is required"):
        parse(["train", "--model", "birnn", "--handoff-dir", str(HANDOFF), "--output-dir", "o"], monkeypatch)
    with pytest.raises(ValueError, match="only work with --handoff-dir"):
        parse(["train", "--model", "birnn", "--train-csv", "x.csv", "--class-weight", "inverse_frequency", "--output-dir", "o"], monkeypatch)
    with pytest.raises(ValueError, match="needs --handoff-dir"):
        parse(["train", "--model", "birnn", "--train-csv", "x.csv", "--split", "group_aware", "--output-dir", "o"], monkeypatch)
    with pytest.raises(ValueError, match="unknown --dataset"):
        parse(["train", "--model", "birnn", "--handoff-dir", str(HANDOFF), "--dataset", "nope", "--output-dir", "o"], monkeypatch)


def test_diagnostic_keyword_truncation_warning(capsys):
    d = T.load_handoff_training_data(ns("diagnostic_original_keywords"))
    T.warn_if_keywords_truncated(d["train"], 256); assert "cut off" in capsys.readouterr().out
    T.warn_if_keywords_truncated(d["train"], 4096); assert capsys.readouterr().out == ""


# ------------------------------------------------------------------ end-to-end smoke (1 epoch, tiny model) + predictions
SMOKE = ["--epochs", "1", "--hidden-dim", "8", "--embedding-dim", "8", "--max-length", "64", "--device", "cpu", "--batch-size", "64"]


@pytest.mark.parametrize("dataset", VARIANTS + ["baseline_all_fields_word_dropout", "baseline_all_fields_sentence_dropout"])
def test_one_epoch_smoke_all_datasets_preserves_source_ids(tmp_path, dataset):
    out = tmp_path / "run"
    p = run("train", "--model", "birnn", "--handoff-dir", str(HANDOFF), "--dataset", dataset, "--output-dir", str(out), *SMOKE)
    n_train = {"baseline_all_fields_word_dropout": 4254, "baseline_all_fields_sentence_dropout": 3950}.get(dataset, 2129)
    assert f"Train={n_train} | Validation=533 | Split=stratified" in p.stdout
    variant = T.parse_dataset_name(dataset)[0]
    val = read(HANDOFF / "datasets" / variant / "validation.csv"); pred = read(out / "validation_predictions.csv")
    assert [r["example_id"] for r in pred] == [r["example_id"] for r in val] and [r["source_record_id"] for r in pred] == [r["source_record_id"] for r in val]
    assert [r["true_label"] for r in pred] == [r["medical_specialty"] for r in val]
    assert all(r["predicted_label"] in T.LABELS and 0.0 < float(r["prediction_probability"]) <= 1.0 for r in pred)
    cfg = json.loads((out / "run_config.json").read_text()); assert cfg["max_length"] == 64 and cfg["rows"]["validation"] == 533
    # predict -> test IDs/order preserved
    res = tmp_path / "res" / "results.txt"
    run("predict", "--checkpoint", str(out / "best.pt"), "--handoff-dir", str(HANDOFF), "--output", str(res), "--device", "cpu")
    test = read(HANDOFF / "datasets" / variant / "test.csv"); tp = read(res.with_name("test_predictions.csv"))
    assert len(tp) == 398 == len(res.read_text().splitlines()) and [r["example_id"] for r in tp] == [r["example_id"] for r in test]
    assert [r["source_record_id"] for r in tp] == [r["test_record_id"] for r in test] and all(r["true_label"] == "" for r in tp)
    assert res.read_text().splitlines() == [r["predicted_label"] for r in tp]


@pytest.mark.skipif(not GROUP.exists(), reason="group-aware folder not found")
def test_group_aware_weighted_smoke_and_checkpoint_guards(tmp_path):
    out = tmp_path / "ga"
    p = run("train", "--model", "bilstm", "--handoff-dir", str(HANDOFF), "--dataset", "baseline_all_fields_word_dropout", "--split", "group_aware",
            "--class-weight", "inverse_frequency", "--class-weight-source", "train_rows", "--output-dir", str(out), *SMOKE)
    meta = json.loads((GROUP / "splits/split_metadata_optional_group_aware.json").read_text())
    assert f"Validation={meta['records_validation']} | Split=group_aware" in p.stdout and "Class weights by name" in p.stdout
    cfg = json.loads((out / "run_config.json").read_text()); w = cfg["class_weights_by_name"]
    assert cfg["split"] == "group_aware" and w["Dermatology"] > w["Surgery"] and np.mean(list(w.values())) == pytest.approx(1.0)
    ck = str(out / "best.pt"); res = str(tmp_path / "r.txt")
    # hand-off checkpoint refuses raw test csv, wrong dataset, wrong split, and a missing folder
    assert run("predict", "--checkpoint", ck, "--test-csv", "x.csv", "--output", res, check=False).returncode != 0
    assert "--handoff-dir" in run("predict", "--checkpoint", ck, "--output", res, check=False).stderr
    assert "trained on split=group_aware" in run("predict", "--checkpoint", ck, "--handoff-dir", str(HANDOFF), "--dataset", "clinical_text", "--output", res, check=False).stderr
    run("predict", "--checkpoint", ck, "--handoff-dir", str(HANDOFF), "--group-aware-dir", str(GROUP), "--output", res, "--device", "cpu")
    assert len(Path(res).read_text().splitlines()) == 398


def test_train_final_in_handoff_mode_uses_train_plus_validation(tmp_path):
    out = tmp_path / "final"
    p = run("train-final", "--model", "birnn", "--handoff-dir", str(HANDOFF), "--dataset", "clinical_text", "--output-dir", str(out), *SMOKE)
    assert "valid=2662" in p.stdout and (out / "final.pt").exists()


# ------------------------------------------------------------------ raw mode unchanged (optional; slow)
RAW = os.environ.get("RAW_TRAIN_CSV")


@pytest.mark.skipif(not RAW or not (MODELS / "train_ORIGINAL.py").exists(), reason="set RAW_TRAIN_CSV and keep train_ORIGINAL.py to run")
def test_raw_mode_identical_to_original_script(tmp_path):
    common = ["train", "--model", "birnn", "--train-csv", RAW, "--epochs", "1", "--hidden-dim", "16", "--embedding-dim", "16", "--batch-size", "32", "--device", "cpu"]
    a = subprocess.run([sys.executable, str(MODELS / "train_ORIGINAL.py"), *common, "--output-dir", str(tmp_path / "a")], capture_output=True, text=True, cwd=MODELS)
    b = subprocess.run([sys.executable, str(MODELS / "train.py"), *common, "--output-dir", str(tmp_path / "b")], capture_output=True, text=True, cwd=MODELS)
    assert a.returncode == b.returncode == 0
    assert a.stdout.replace(str(tmp_path / "a"), "X") == b.stdout.replace(str(tmp_path / "b"), "X")
    for f in ("history.csv", "validation_metrics.json", "misclassified.csv", "split_rows.json"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == sorted(p.name for p in (tmp_path / "b").iterdir())
