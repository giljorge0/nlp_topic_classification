# Model hand-off README

Data component only. No model, training loop or predictions are included. **Nothing here has been trained or evaluated; no class-weight setting is claimed to be optimal.**

## Which file for which experiment
| variant                      | fields                                            | keyword specialty prefix   |   vocab |   suggested max_len | train/val/test rows   |
|:-----------------------------|:--------------------------------------------------|:---------------------------|--------:|--------------------:|:----------------------|
| baseline_all_fields          | description, sample_name, transcription, keywords | REMOVED                    |   17222 |                1263 | 2129/533/398          |
| clinical_text                | description, transcription                        | n/a (no keywords)          |   16973 |                1249 | 2129/533/398          |
| compressed_unique_text       | description, transcription                        | n/a (no keywords)          |   16966 |                1249 | 2129/533/398          |
| diagnostic_original_keywords | description, sample_name, transcription, keywords | kept (ORIGINAL)            |   17225 |                1265 | 2129/533/398          |

| experiment | use |
|---|---|
| default model comparisons | `datasets/baseline_all_fields/` (keyword specialty prefix removed) |
| text-only reference | `datasets/clinical_text/` |
| shorter inputs / less templated text | `datasets/compressed_unique_text/` |
| **leakage diagnostic only** | `datasets/diagnostic_original_keywords/` (original keywords incl. the specialty name) |

Each variant folder: `train.csv`, `validation.csv`, `test.csv`, `vocab_train_only.json`, `preprocessing_config.json` (and `boilerplate_rules_train_only.json` for `compressed_unique_text`). `train_augmented_word_dropout.csv` / `train_augmented_sentence_dropout.csv` exist only if the build was run with `--augment` (see Augmentation below). `compressed_unique_text` also has `compression_log.csv` and `compression_removed_sentences.csv` (every record and sentence that compression changed).

Columns: train/validation = `example_id, source_record_id, medical_specialty, partition, augmentation_type, augmentation_seed, augmentation_params, text`; test = `example_id, test_record_id, original_order, text`. See `DATA_DICTIONARY.md`.

## Split (single source of truth)
* **Primary** (`splits/split_manifest_primary_stratified.csv`): stratified random, validation fraction 0.2, seed 42, 2129 train / 533 validation. *Status: proposed, needs your approval.*
* **Optional** (`splits/split_manifest_optional_group_aware.csv`): exact-transcription duplicate groups never straddle the split (2129 / 533). Robustness check only; never silently substituted.
* All four variants use exactly these partitions (`partition` column == manifest). Never re-split per variant. Re-running the build with the same data/config reuses the manifest; if data/config differ the script **refuses** unless `--force-resplit` is passed.
* Test is separate and untouched; no test labels exist or were inferred.

## Loading and joining (same for the three recommended datasets: `baseline_all_fields`, `clinical_text`, `compressed_unique_text`)
```python
import pandas as pd
D = "datasets/baseline_all_fields"                    # or clinical_text / compressed_unique_text
train = pd.read_csv(f"{D}/train.csv", dtype=str, keep_default_na=False)        # columns: example_id, source_record_id, medical_specialty, partition, ..., text
val   = pd.read_csv(f"{D}/validation.csv", dtype=str, keep_default_na=False)
test  = pd.read_csv(f"{D}/test.csv", dtype=str, keep_default_na=False)         # example_id, test_record_id, original_order, text
# optional: train = pd.read_csv(f"{D}/train_augmented_word_dropout.csv", dtype=str, keep_default_na=False)   # originals + appended copies
# ... train / predict with the existing model; keep predictions in the SAME ORDER as val ...
pred = pd.DataFrame({"example_id": val.example_id, "predicted": model_predictions})
subsets = pd.read_csv("splits/validation_subsets_primary.csv", dtype=str, keep_default_na=False).drop(columns=["medical_specialty"])
joined = val.merge(pred, on="example_id").merge(subsets, on="source_record_id", how="left")   # use validation_subsets_group_aware.csv for the group-aware run
# joined has source_record_id, true label, prediction and the subset flags (complete_input_label_conflict, complete_input_consistent_duplicate, no_complete_input_twin_in_train, ...)
```
`mappings/record_mapping.csv` maps `source_record_id` to raw physical lines, repair status, duplicate groups and both partitions. For `results.txt`, write one predicted label per line in `test.csv` order (`original_order` 0..N-1, N = 398); no header.

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
Vocabulary (`min_freq=2`, `<pad>=0`, `<unk>=1`), boilerplate rules (sentence in >= 15 train notes, >= 6 words, no digits, no negation), max-length statistics. Numbers are not normalised in any variant. The text is never truncated in the CSVs; apply one max-length rule inside the model input pipeline for all partitions (`preprocessing_config.json` gives train percentiles).

## Class weights
`weights/class_weights_primary_split_wide.csv` / `.json` (one column per strategy, indexed by class name; `label_to_index.json` gives the index order used here: map by NAME if your code uses another order). See `CLASS_WEIGHTING_EXPERIMENTS.md`.

## Unresolved limitations
* Identical complete inputs with different labels cannot be separated by any input-only model.
* 7 quarantined records and 4 partially recovered ones (one lost joint character; keywords missing) - see `audit/`.
* Near-duplicates (non-identical) are not grouped in the split; only exact normalised transcriptions are.
* `compressed_unique_text` is my interpretation of that variant name; confirm it is what you intended.
* The model/training code was not available to me; the weight interface below is documented, not integrated.

Integrity: `handoff_manifest.json` lists the configuration and SHA-256 of every file.
