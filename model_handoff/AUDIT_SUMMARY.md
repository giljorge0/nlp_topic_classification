# Audit summary (auto-generated; verified against the data in this build)

## Previous findings re-checked against this build
| finding                                   |   reported_before |   measured_now | matches   |
|:------------------------------------------|------------------:|---------------:|:----------|
| usable_records                            |            2662   |         2662   | True      |
| quarantined_records                       |               7   |            7   | True      |
| transcription_duplicate_groups            |             731   |          731   | True      |
| records_in_transcription_duplicate_groups |            1530   |         1530   | True      |
| transcription_groups_with_label_conflict  |             719   |          730   | False     |
| complete_input_conflict_groups            |              82   |           82   | True      |
| complete_input_conflict_records           |             171   |          171   | True      |
| surgery_percent                           |              33.4 |           33.4 | True      |
| dermatology_percent                       |               0.9 |            0.9 | True      |
| test_records                              |             398   |          398   | True      |
| median_transcription_words                |             387   |          387   | True      |

All previous findings reproduced: **False**. If any row says `False`, the old number must not be quoted.

**Correction:** mismatching rows: `transcription_groups_with_label_conflict`. For `transcription_groups_with_label_conflict` the earlier figure (719) came from an indexing bug in `analyze_data.py` v1 (group members were looked up in the unfiltered frame after empty texts were skipped, so some labels were wrong). It is fixed, covered by a regression test, and the figure measured here is the correct one. Group sizes, 731 groups / 1,530 records, the 82 groups / 171 records, and the split-overlap counts were not affected.


## Records
* Usable training records: 2662 (partial-policy `include`); quarantined: 7 ['TR000160', 'TR000713', 'TR000946', 'TR001086', 'TR002087', 'TR002326', 'TR002390'].
* Records with a partially recovered transcription / missing keywords (kept, flagged `recovered_partial`): ['TR000960', 'TR001094', 'TR002098', 'TR002334'].
* Test records: 398 (count verified against the raw file's physical lines; original order preserved in every variant).

## Conflicting labels (NOT resolved, NOT removed)
* 731 groups of identical transcriptions cover 1530 records; 730 of those groups contain different labels (same note filed under several specialties).
* 171 records in 82 groups have identical COMPLETE inputs (all four fields incl. keywords) but different labels: no input-only model can get all of them right.
* Primary stratified split: 262 transcription groups cross train/validation, of which 261 have conflicting labels. Validation records with a train twin: 269 (same label: 1; only different labels: 268).
* Group-aware split: 0 transcription groups cross.
* Consequence: memorising a training twin is usually *penalised* here. Report validation metrics on the subsets in `splits/validation_subsets_*.csv`.

## Keyword leakage
* 2219 training records (83.36%) start `keywords` with a specialty name; it equals the true label in 2219 of them. 80.4% of test records have the same pattern.
* `diagnostic_original_keywords` keeps this information on purpose; `baseline_all_fields` removes only that prefix.

## Class imbalance
Surgery 33.4% vs Dermatology 0.9% of cleaned records (see `weights/class_counts.csv`).

## Stratification warnings
Primary: none. Group-aware: none.
