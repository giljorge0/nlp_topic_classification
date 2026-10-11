# Medical-specialty data pipeline, v2 (data component only)

No model, training loop or predictions live here. Your colleagues' model code is untouched (and was not available to me).

## Files
| file | role |
|---|---|
| `clinical_common.py` | raw reading, line-oriented record repair/quarantine, test ingestion (policy-free) |
| `analyze_data.py` | EDA + integrity/duplicate/leakage audit (describes only). **v2 fixes a duplicate-grouping bug**, see below |
| `preprocess_data.py` | exploratory variant generator (field/normalisation/windows). Augmented variants now require `--split-manifest`; there is no override for augmenting validation/test |
| `build_model_handoff.py` | **v2 main entry point**: split-first, model-ready hand-off (`model_handoff/` + `model_handoff.zip`) |
| `handoff_docs.py` | writes the hand-off documents from computed numbers |
| `tests/` | 45 synthetic pytest tests |
| `config.sample.json` | sample config for `preprocess_data.py --config` |
| `example_outputs/data_analysis/` | regenerated analysis outputs (v2, bug fixed) |

## Run
```
pip install pandas numpy matplotlib scikit-learn tabulate pytest
python analyze_data.py        --train train.csv --test test_no_labels.csv --output-dir data_analysis
python build_model_handoff.py --train train.csv --test test_no_labels.csv --output-dir model_handoff --seed 42
python build_model_handoff.py ... --augment word_dropout        # recommended main augmentation (p=0.05), TRAIN only, baseline_all_fields
python build_model_handoff.py ... --augment sentence_dropout    # optional second method (long notes only)
python build_model_handoff.py ... --split group_aware --output-dir model_handoff_group_aware --augment word_dropout
python -m pytest -q tests
```
Delete any old `processed_data/` folder and old zips from iteration 1; they are superseded (they contained no split).

## Pipeline order (recorded in `handoff_build_log.json`)
1 read raw -> 2 repair defensible records -> 3 quarantine the rest -> 4 validate labels/columns (+ test count vs raw lines) -> 5 stable `TR######` IDs -> 6 exact-duplicate groups (`DG######`, and `CG######` for complete inputs) -> 7 **split manifest (created once, reused; refuses a silent resplit)** -> 8 fit vocabulary / boilerplate rules / length stats on TRAIN only -> 9 transform train/validation/test with the same fixed rules -> 10 optional augmentation (TRAIN only) -> 11 export + SHA-256 manifest + zip.

## What changed in v2
* New `build_model_handoff.py`, `handoff_docs.py`, `tests/test_handoff.py`.
* `analyze_data.py`: fixed an indexing bug (`groupby(...).indices` positions were applied to the unfiltered frame once empty texts were skipped). It corrupted labels/record IDs in `exact_duplicate_groups.csv` and the conflict counts: **719 -> 730** transcription groups with conflicting labels (of 731). Regression test added and shown to fail on the old logic. Group sizes (731 / 1,530), complete-input conflicts (82 / 171) and split-overlap numbers were not affected.
* `preprocess_data.py`: removed `--allow-eval-augmentation`; augmentation requires `--split-manifest` and only touches `partition == train`; token-deletion variant removed from its default list.
* Everything else (ingestion/repair rules, windows, normalisations) is unchanged; see the v1 notes in `example_outputs/data_analysis/analysis_report.md`.

## v2.1 changes (this iteration)
* **Augmentation revised.** Removed the old `token_deletion` option of the builder (it was already word-level, never character-level, but its default was p=0.10 and it was applied to all four variants). Now: `word_dropout` (default p=0.05, recommended main method) and optional `sentence_dropout` (p=0.10, notes with >= 8 sentences and >= 100 words). Applied to `baseline_all_fields` only unless `--augment-variants` says otherwise. No character-level, spelling, back-translation or synonym methods exist. `preprocess_data.py`'s exploratory `token_deletion` default is also 0.05 now.
* **`--split group_aware`** drives datasets, fitted components and augmentation from the group-aware manifest (use a separate `--output-dir`).
* **Bug fix in `compressed_unique_text`:** within-note repeat collapsing treated list numbers/abbreviations (`2.`, `Mr.`) as sentences and removed them (thousands of digit-bearing removals). Repeats are now collapsed only for eligible sentences (>= 6 words, no digit, no negation). That variant's files changed; the other three variants, the split manifests and the weights are byte-identical to the previous hand-off.
* New traceability files for that variant: `compression_log.csv` (every record, partition, sentences removed, words before/after) and `compression_removed_sentences.csv` (each removed sentence, kind, training document frequency).
