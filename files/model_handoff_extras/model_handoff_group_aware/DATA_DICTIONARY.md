# Data dictionary

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
