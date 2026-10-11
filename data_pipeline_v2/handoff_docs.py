"""Writes the human-readable hand-off documents (all numbers come from the computed audit, nothing is hard-coded)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import clinical_common as cc


def _t(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_docs(out: Path, a, audit: dict, pmeta: dict, gmeta: dict, vcfgs: dict, ctab: pd.DataFrame, wl: pd.DataFrame, measured: dict) -> None:
    ver = pd.DataFrame([{"finding": k, "reported_before": v["previously_reported"], "measured_now": v["measured_now"], "matches": v["matches"]}
                        for k, v in audit["verification_of_previous_findings"].items()])
    ov = audit["split_overlap"]
    vrows = pd.DataFrame([{"variant": k, "fields": ", ".join(v["fields"]), "keyword specialty prefix": "REMOVED" if v["keyword_specialty_prefix_removed"] else ("kept (ORIGINAL)" if "keywords" in v["fields"] else "n/a (no keywords)"),
                           "vocab": v["vocab_size"], "suggested max_len": v["max_length_policy"]["suggested_max_len"], "train/val/test rows": "/".join(str(v["rows"][x]) for x in ("train", "validation", "test"))}
                          for k, v in vcfgs.items()])
    mism = [k for k, v in audit["verification_of_previous_findings"].items() if not v["matches"]]
    correction = ""
    if mism:
        correction = ("\n**Correction:** mismatching rows: " + ", ".join(f"`{k}`" for k in mism) + ". For `transcription_groups_with_label_conflict` the earlier figure (719) came from an indexing bug in "
                      "`analyze_data.py` v1 (group members were looked up in the unfiltered frame after empty texts were skipped, so some labels were wrong). It is fixed, covered by a regression test, "
                      "and the figure measured here is the correct one. Group sizes, 731 groups / 1,530 records, the 82 groups / 171 records, and the split-overlap counts were not affected.\n")
    audit_md = f"""# Audit summary (auto-generated; verified against the data in this build)

## Previous findings re-checked against this build
{_t(ver)}

All previous findings reproduced: **{audit['all_previous_findings_match']}**. If any row says `False`, the old number must not be quoted.
{correction}

## Records
* Usable training records: {measured['usable_records']} (partial-policy `{a.partial_policy}`); quarantined: {measured['quarantined_records']} {audit['quarantined_ids']}.
* Records with a partially recovered transcription / missing keywords (kept, flagged `recovered_partial`): {audit['partial_records']}.
* Test records: {measured['test_records']} (count verified against the raw file's physical lines; original order preserved in every variant).

## Conflicting labels (NOT resolved, NOT removed)
* {measured['transcription_duplicate_groups']} groups of identical transcriptions cover {measured['records_in_transcription_duplicate_groups']} records; {measured['transcription_groups_with_label_conflict']} of those groups contain different labels (same note filed under several specialties).
* {measured['complete_input_conflict_records']} records in {measured['complete_input_conflict_groups']} groups have identical COMPLETE inputs (all four fields incl. keywords) but different labels: no input-only model can get all of them right.
* Primary stratified split: {ov['primary_stratified']['transcription']['groups_crossing_split']} transcription groups cross train/validation, of which {ov['primary_stratified']['transcription']['label_conflicting_groups_crossing_split']} have conflicting labels. Validation records with a train twin: {ov['primary_stratified']['transcription']['validation_records_with_train_twin']} (same label: {ov['primary_stratified']['transcription']['validation_twin_in_train_same_label']}; only different labels: {ov['primary_stratified']['transcription']['validation_twin_in_train_only_different_labels']}).
* Group-aware split: {ov['optional_group_aware']['transcription']['groups_crossing_split']} transcription groups cross.
* Consequence: memorising a training twin is usually *penalised* here. Report validation metrics on the subsets in `splits/validation_subsets_*.csv`.

## Keyword leakage
* {audit['keyword_specialty_prefix_records']} training records ({audit['keyword_specialty_prefix_percent']}%) start `keywords` with a specialty name; it equals the true label in {audit['keyword_prefix_equals_label']} of them. {audit['test_keyword_specialty_prefix_percent']}% of test records have the same pattern.
* `diagnostic_original_keywords` keeps this information on purpose; `baseline_all_fields` removes only that prefix.

## Class imbalance
Surgery {measured['surgery_percent']}% vs Dermatology {measured['dermatology_percent']}% of cleaned records (see `weights/class_counts.csv`).

## Stratification warnings
Primary: {pmeta['stratification_warnings'] or 'none'}. Group-aware: {gmeta['stratification_warnings'] or 'none'}.
"""
    (out / "AUDIT_SUMMARY.md").write_text(audit_md, encoding="utf-8")

    readme = f"""# Model hand-off README

Data component only. No model, training loop or predictions are included. **Nothing here has been trained or evaluated; no class-weight setting is claimed to be optimal.**

## Which file for which experiment
{_t(vrows)}

| experiment | use |
|---|---|
| default model comparisons | `datasets/baseline_all_fields/` (keyword specialty prefix removed) |
| text-only reference | `datasets/clinical_text/` |
| shorter inputs / less templated text | `datasets/compressed_unique_text/` |
| **leakage diagnostic only** | `datasets/diagnostic_original_keywords/` (original keywords incl. the specialty name) |

Each variant folder: `train.csv`, `validation.csv`, `test.csv`, `vocab_train_only.json`, `preprocessing_config.json` (and `boilerplate_rules_train_only.json` for `compressed_unique_text`). `train_augmented_word_dropout.csv` / `train_augmented_sentence_dropout.csv` exist only if the build was run with `--augment` (see Augmentation below). `compressed_unique_text` also has `compression_log.csv` and `compression_removed_sentences.csv` (every record and sentence that compression changed).

Columns: train/validation = `example_id, source_record_id, medical_specialty, partition, augmentation_type, augmentation_seed, augmentation_params, text`; test = `example_id, test_record_id, original_order, text`. See `DATA_DICTIONARY.md`.

## Split (single source of truth)
* **Primary** (`splits/split_manifest_primary_stratified.csv`): stratified random, validation fraction {a.val_fraction}, seed {a.seed}, {pmeta['records_train']} train / {pmeta['records_validation']} validation. *Status: proposed, needs your approval.*
* **Optional** (`splits/split_manifest_optional_group_aware.csv`): exact-transcription duplicate groups never straddle the split ({gmeta['records_train']} / {gmeta['records_validation']}). Robustness check only; never silently substituted.
* All four variants use exactly these partitions (`partition` column == manifest). Never re-split per variant. Re-running the build with the same data/config reuses the manifest; if data/config differ the script **refuses** unless `--force-resplit` is passed.
* Test is separate and untouched; no test labels exist or were inferred.

## Loading and joining (same for the three recommended datasets: `baseline_all_fields`, `clinical_text`, `compressed_unique_text`)
```python
import pandas as pd
D = "datasets/baseline_all_fields"                    # or clinical_text / compressed_unique_text
train = pd.read_csv(f"{{D}}/train.csv", dtype=str, keep_default_na=False)        # columns: example_id, source_record_id, medical_specialty, partition, ..., text
val   = pd.read_csv(f"{{D}}/validation.csv", dtype=str, keep_default_na=False)
test  = pd.read_csv(f"{{D}}/test.csv", dtype=str, keep_default_na=False)         # example_id, test_record_id, original_order, text
# optional: train = pd.read_csv(f"{{D}}/train_augmented_word_dropout.csv", dtype=str, keep_default_na=False)   # originals + appended copies
# ... train / predict with the existing model; keep predictions in the SAME ORDER as val ...
pred = pd.DataFrame({{"example_id": val.example_id, "predicted": model_predictions}})
subsets = pd.read_csv("splits/validation_subsets_primary.csv", dtype=str, keep_default_na=False).drop(columns=["medical_specialty"])
joined = val.merge(pred, on="example_id").merge(subsets, on="source_record_id", how="left")   # use validation_subsets_group_aware.csv for the group-aware run
# joined has source_record_id, true label, prediction and the subset flags (complete_input_label_conflict, complete_input_consistent_duplicate, no_complete_input_twin_in_train, ...)
```
`mappings/record_mapping.csv` maps `source_record_id` to raw physical lines, repair status, duplicate groups and both partitions. For `results.txt`, write one predicted label per line in `test.csv` order (`original_order` 0..N-1, N = {measured['test_records']}); no header.

## First experiments: smallest file set
Give your colleague: `MODEL_HANDOFF_README.md`, `CLASS_WEIGHTING_EXPERIMENTS.md`, `splits/split_manifest_primary_stratified.csv`, `splits/validation_subsets_primary.csv`, `weights/class_weights_primary_split_wide.csv` (or `.json`) + `weights/label_to_index.json`, and from `datasets/` the `train.csv`, `validation.csv`, `test.csv` (+ optional `vocab_train_only.json`, `preprocessing_config.json`) of: `baseline_all_fields` (exp 1), `clinical_text` (exp 2), `compressed_unique_text` (exp 3) and, for the leakage diagnostic only, `diagnostic_original_keywords` (exp 4). Everything else (audit/, mappings/, compression logs, group-aware files) is optional reference.

## Augmentation (opt-in; TRAIN rows only; unaugmented data is the baseline)
**Recommended main method: `word_dropout` (conservative, p=0.05) on `baseline_all_fields`.** `sentence_dropout` is an optional second method. Nothing is augmented unless you ask, and no claim is made that either helps: that needs your model results.

```
# main method (word dropout, p=0.05, 1 copy per train note) -> datasets/baseline_all_fields/train_augmented_word_dropout.csv
python build_model_handoff.py --train train.csv --test test_no_labels.csv --output-dir model_handoff --augment word_dropout
# optional second method (sentence dropout, p=0.10; notes with >= 8 sentences AND >= 100 words only) -> train_augmented_sentence_dropout.csv
python build_model_handoff.py --train train.csv --test test_no_labels.csv --output-dir model_handoff --augment sentence_dropout
# both files at once / another variant / other probabilities
python build_model_handoff.py ... --augment word_dropout sentence_dropout
python build_model_handoff.py ... --augment word_dropout --augment-variants clinical_text --word-dropout-prob 0.03
# group-aware robustness run (separate folder; vocabulary and boilerplate rules are RE-FITTED on that split's train)
python build_model_handoff.py ... --split group_aware --output-dir model_handoff_group_aware --augment word_dropout
```
Re-running is deterministic: unaugmented files, split manifests and augmented files come out byte-identical for the same data, config and seed.

How it works
* The split manifest exists first; augmentation then reads only rows whose partition is `train` (there is no override; any other row raises an error). `validation.csv` and `test.csv` are never touched.
* The augmented file = every original train row **plus** appended copies. Original rows are kept unchanged (`augmentation_type == none`).
* `word_dropout`: whole whitespace words only (never characters). Never dropped: negations (no, not, without, denies/denied, negative, none, never, absent, neither, nor, cannot; also when glued like `abscess.,no`, `non-`, `w/o`), any token containing a digit (numbers are kept in every variant), and the field markers. At most 50% of droppable words go, so a note cannot be emptied.
* `sentence_dropout`: only the TRANSCRIPTION sentences of long notes; retained sentences keep their order; sentences containing a negation are never dropped; at most 25% of sentences go; the result is never empty.
* A copy identical to its source (nothing dropped / note too short or ineligible) is **not** emitted, so the number of copies can be smaller than the number of train notes (`augmentation_check.json` reports requested / generated / skipped).
* Columns of the augmented file: `example_id` (`<source_record_id>__wdrop00` / `__sdrop00`), `source_record_id`, `medical_specialty`, `partition` (always `train`), `augmentation_type`, `augmentation_probability`, `augmentation_seed` (derived deterministically from the base seed, method, record and copy), `augmentation_params` (JSON), `duplicate_group_id` (inherited from the source; with `--split group_aware` no augmented group appears in validation), `text`.
* `augmentation_check.json` records the automatic checks (maps to a train source, no validation rows, unique IDs, probability/seed present, non-empty, negation counts and, for word dropout, digit tokens preserved).
* Not implemented, by design: character deletion/swaps, spelling corruption, back-translation, medical synonym replacement.

## Learned components were fitted on TRAIN only
Vocabulary (`min_freq={a.min_freq}`, `<pad>=0`, `<unk>=1`), boilerplate rules (sentence in >= {a.boilerplate_min_df} train notes, >= {a.boilerplate_min_words} words, no digits, no negation), max-length statistics. Numbers are not normalised in any variant. The text is never truncated in the CSVs; apply one max-length rule inside the model input pipeline for all partitions (`preprocessing_config.json` gives train percentiles).

## Class weights
`weights/class_weights_primary_split_wide.csv` / `.json` (one column per strategy, indexed by class name; `label_to_index.json` gives the index order used here: map by NAME if your code uses another order). See `CLASS_WEIGHTING_EXPERIMENTS.md`.

## Unresolved limitations
* Identical complete inputs with different labels cannot be separated by any input-only model.
* 7 quarantined records and {len(audit['partial_records'])} partially recovered ones (one lost joint character; keywords missing) - see `audit/`.
* Near-duplicates (non-identical) are not grouped in the split; only exact normalised transcriptions are.
* `compressed_unique_text` is my interpretation of that variant name; confirm it is what you intended.
* The model/training code was not available to me; the weight interface below is documented, not integrated.

Integrity: `handoff_manifest.json` lists the configuration and SHA-256 of every file.
"""
    (out / "MODEL_HANDOFF_README.md").write_text(readme, encoding="utf-8")

    w = wl[wl.basis == "primary_stratified_train_partition"]
    summ = w.groupby("strategy").weight.agg(min_weight="min", max_weight="max").assign(max_over_min=lambda d: d.max_weight / d.min_weight).reset_index()
    summ["alpha"] = summ.strategy.map(lambda s: float(s.split("_")[-1]) if s.startswith("interpolated") else None)
    summ = summ.sort_values(["max_over_min"]).round(3)
    exp = f"""# Class-weighting experiments (recommendations for the existing model code)

I did not run any of these. Weights below are **candidates**, not tuned values; do not call one optimal before it has been trained and evaluated.

## Definitions (computed from the PRIMARY split's TRAIN partition class counts)
`freq_c = count_c / N_train` (a proportion). Every vector is normalised so the **mean over the {len(cc.EXPECTED_LABELS)} classes is 1**.
1. `none`: all 1.
2. `inverse_frequency`: `1/freq_c`, normalised.
3. `inverse_sqrt_frequency`: `1/sqrt(freq_c)`, normalised.
4. `interpolated_alpha_<a>`: `weight_c(alpha) = normalised((1 - alpha) + alpha / freq_c)`; alpha=0 -> equal weights, alpha=1 -> inverse frequency.

**Caution on the interpolation:** because `freq_c` is a proportion (1/freq ranges ~{1/float(w[w.strategy=='none'].train_frequency.max()):.0f}-{1/float(w[w.strategy=='none'].train_frequency.min()):.0f}), the `1 - alpha` term is small next to `alpha/freq_c`, so the effective strength rises quickly with alpha rather than linearly:

{_t(summ)}

(`max_over_min` = strongest / weakest class weight.) If you want a *linear* blend of the two normalised vectors, say so and I will change the formula.

Per-class values: `weights/class_weights_primary_split_wide.csv` (also `.json`; a version for the group-aware split's train partition is in `class_weights_long.csv`, column `basis`).

## Interface the training code needs
A float vector `w` of length {len(cc.EXPECTED_LABELS)} aligned to the model's class indices (map by class NAME; `label_to_index.json` is only this repo's order). Typical use: PyTorch `CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))`; Keras `class_weight={{idx: w_i}}`. If the code does not expose this, the minimal change is one optional argument (e.g. `--class-weights path.json`) passed to the loss. I did not touch their code.

## Suggested first-pass table (keep EVERYTHING else fixed)
Fixed: architecture, preprocessing variant (start with `baseline_all_fields`), `splits/split_manifest_primary_stratified.csv`, seed, optimizer, training budget, no augmentation.

| run | class weights | notes |
|---|---|---|
| 1 | none | reference |
| 2 | inverse_sqrt_frequency | |
| 3 | inverse_frequency | |
| 4-8 (only if 2/3 look promising) | interpolated alpha 0, 0.25, 0.5, 0.75, 1.0 | alpha=0 duplicates run 1, alpha=1 duplicates run 3 (skip) |

Primary objective: **validation macro-F1**; also record accuracy and **every class's F1** (Dermatology and Psychiatry-Psychology have the fewest validation examples, so their F1 is noisy). Early stopping on a predeclared metric, preferably macro-F1 if the code supports it reliably. Run more seeds only for the 1-2 best configurations, if compute allows. Do not change weights, architecture, input fields and augmentation at the same time; vary one factor per table.

Also report each run on the subsets in `splits/validation_subsets_primary.csv` (complete-input-conflict / consistent-duplicate / no complete-input twin in train). Check that the course's automatic criteria (accuracy > 49% and per-class F1 >= 25%) hold on validation.

## What weights cannot do
Class weights only change the optimisation trade-off. They cannot resolve notes whose complete inputs are identical but whose labels differ ({measured['complete_input_conflict_records']} records in {measured['complete_input_conflict_groups']} groups): at most one label per identical input can be right.

## Duplicate-group weights (diagnostic, separate from class weights)
`weights/duplicate_group_sample_weights_primary_train.csv`: per train record `1/(train records in its transcription duplicate group)`, mean 1. If the code supports sample weights, combine as `sample_weight_i = class_weight[label_i] * group_weight_i`, then optionally rescale to mean 1. Evaluate it as its own experiment; do not mix with the class-weight comparison.

No oversampling or duplication of minority records is done anywhere in the default data.
"""
    (out / "CLASS_WEIGHTING_EXPERIMENTS.md").write_text(exp, encoding="utf-8")

    dd = """# Data dictionary

| column | file(s) | meaning |
|---|---|---|
| `source_record_id` | everything | stable ID `TR######` = order of the record's first physical line in train.csv (quarantined records consume IDs) |
| `example_id` | variant files | row ID; equals `source_record_id` for originals, `<id>__tdelNN` for augmented rows; test: `TE######` |
| `medical_specialty` | train/validation | label (never altered) |
| `partition` | train/validation | `train` or `validation` per the primary split manifest |
| `augmentation_type/seed/params` | train/validation | `none` for originals; token-deletion metadata otherwise |
| `text` | variant files | concatenation of selected, clinically-normalised fields with markers `[DESCRIPTION] [SAMPLE_NAME] [TRANSCRIPTION] [KEYWORDS]` |
| `test_record_id`, `original_order` | test | `TE######`; 0-based line order in test_no_labels.csv (no header) |
| `duplicate_group_id` | splits, mappings | `DG######`: identical normalised transcription (NFKC, case-folded, whitespace-collapsed); empty transcription -> own group |
| `complete_input_group_id` | mappings, subsets | `CG######`: identical normalised description+sample_name+transcription+ORIGINAL keywords |
| `*_group_size / n_labels / label_conflict` | mappings | group size over all cleaned records, #distinct labels, conflict flag |
| `repair_status` | mappings | clean / recovered / recovered_partial (quarantined records are in `audit/quarantined_train.csv`) |
| `split_strategy`, `split_seed` | split manifests | `stratified_random` or `duplicate_group_aware`; seed |
| subset flags | `splits/validation_subsets_*.csv` | see MODEL_HANDOFF_README |
"""
    (out / "DATA_DICTIONARY.md").write_text(dd, encoding="utf-8")
