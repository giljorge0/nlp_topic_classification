# Help the Doctor: local Random Word Dropout + ChatGPT paraphrasing

This package is designed to be **extracted directly into your existing `Models/` folder**. It does not train a classifier or produce predictions for `test_no_labels.csv`.

- Random Word Dropout runs completely offline on your computer.
- ChatGPT paraphrases are generated **in your ChatGPT conversation**, not by a local script. The scripts only export requested clinical descriptions, validate returned paraphrases and combine datasets.
- **No API keys, external API calls, paid packages or Python installations beyond Python 3.10+** are required.
- The CSVs consumed by the classifiers always have the original columns: `medical_specialty,description,sample_name,transcription,keywords`.
- CSV files are automatically read as comma-separated or semicolon-separated (or tab-separated). Generated training and validation CSVs preserve the input delimiter. ChatGPT request and audit CSV files use commas.
- The `split` command prints a data-quality warning if it detects an unexpected number of classes or suspiciously long labels/descriptions. Such warnings **do not** repair corrupted rows. Prefer an untouched original download over CSVs resaved by Excel.
- Only `description` is augmented by default. Change `source_column` to `transcription` only if you intentionally want that experiment (full notes are larger and harder to validate).

## 1. Extract the ZIP into `Models/`

After extraction, your existing folder should include:

```text
YOUR_PROJECT/
├── train.csv                    # if this is where your original train.csv lives
├── Models/
│   ├── [all your existing model code: keep it]
│   ├── augment_data.py          # NEW
│   ├── augmentation_config.json # NEW
│   ├── README_AUGMENTATION.md   # NEW
│   ├── augmentation_pkg/        # NEW
│   │   ├── __init__.py
│   │   ├── io_utils.py
│   │   ├── splitting.py
│   │   ├── random_dropout.py
│   │   └── medical_validation.py
│   └── tests/
│       └── test_chat_augmentation.py
└── test_no_labels.csv           # NEVER used for training or augmentation
```

Unzip the contents **inside** `Models/`, not into `Models/Models/`. Existing model code and CSV files are not included in the ZIP.

If you installed the previous API-based augmentation package, this new `augment_data.py` and `augmentation_config.json` replace the earlier versions. The old `augmentation_pkg/llm_paraphrasing.py`, `.env.example` and API-related files are not used and can be removed if you wish.

## 2. Open a terminal *inside* `Models/`

For example:

```bash
cd path/to/YOUR_PROJECT/Models
python --version
python -m unittest discover -s tests -p 'test_chat_augmentation.py' -v
```

On systems where `python` is unavailable use `python3` (macOS/Linux) or `py` (Windows). The tests do not need your actual dataset or any online connection.

## 3. Choose your existing training split (important)

**Case A: You already have a train/validation split used by your models.**

Do **NOT** create another split. For every command below that accepts `--train-input`, use the path to **the same existing labeled training partition**:

```bash
python augment_data.py rwd --train-input path/to/your/train_split.csv
python augment_data.py chat-export --train-input path/to/your/train_split.csv
```

For `chat-import` and `combined`, repeat the same `--train-input` path. Preserve your validation data separately and never augment it.

**Case B: You do not yet have a fixed train/validation split.**

From `Models/`, if `train.csv` lives in its parent folder:

```bash
python augment_data.py split --input ../train.csv
```

If `train.csv` is *inside* `Models/`, use `--input train.csv`. If it lives elsewhere, use its actual relative or absolute path.

This writes:

```text
Models/data/splits/train_base.csv
Models/data/splits/validation.csv
Models/reports/augmentation/split_manifest.json
```

The separator is detected automatically. Your command works for either comma or semicolon CSVs. You **do not** need to edit the CSV in Excel.

**Important when using the October 15 uploaded `train(4).csv`:** It appears to contain corrupted merged records, including three invalid medical specialty labels and several descriptions containing other CSV records. Do not train on that damaged export; retrieve the unchanged original `train.csv` from the project materials first. The scripts can read semicolons but cannot reconstruct missing or merged source rows.

This is a best-effort group-aware stratified split, grouping exact normalized clinical notes. Review its warnings and class distributions, especially for rare labels. It will refuse to overwrite an existing split unless you explicitly pass `--force`. Avoid using `--force` after augmentation because it invalidates previous CSVs.

Without `--train-input`, subsequent commands default to `data/splits/train_base.csv`.

## 4. Random Word Dropout: fully local

```bash
python augment_data.py rwd
```

Or if you already have a training partition:

```bash
python augment_data.py rwd --train-input path/to/your/train_split.csv
```

Outputs:

```text
Models/data/augmented/train_rwd.csv
Models/reports/augmentation/rwd_audit.csv
Models/reports/augmentation/rwd_rejected.csv
Models/reports/augmentation/rwd_summary.json
```

Default `probability=0.05`, `mode=safe`, with one copy per selected example. Very conservative medical token protection may reject examples when no safe deletion is possible. Never assume every requested synthetic sample will be produced. Inspect the audit and summary.

You can edit `augmentation_config.json` to change the deletion probability to `0.10` for a second controlled experiment, or use `--tag p10` to avoid overwriting the previous output:

```bash
python augment_data.py rwd --tag p10
```

The `--tag` option appends a suffix to output filenames, e.g. `train_rwd_p10.csv`.

## 5. Create requests for ChatGPT, without APIs

```bash
python augment_data.py chat-export
```

This exports **every row with nonempty `description` in the active training partition**. There is no numerical cap or minority-class selection by default.

Or with your existing training partition:

```bash
python augment_data.py chat-export --train-input path/to/your/train_split.csv
```

Outputs:

```text
Models/data/chat_requests/paraphrase_requests.csv
Models/data/chat_requests/request_manifest.json
```

Upload **`paraphrase_requests.csv`** to this ChatGPT conversation. It contains **only** `source_row_id` and `description` (or the selected source field), never `medical_specialty`, `sample_name` or `transcription` unless you explicitly change the source field.

Ask for: "Produce one clinically faithful paraphrase of each provided description, preserving the ID exactly and returning a downloadable UTF-8 CSV with two columns: `source_row_id,paraphrase`. Keep negation, certainty, laterality, diagnoses, procedures and measurements unchanged. Never add or infer medical facts." In this project conversation you can simply upload the file and refer to these instructions.

**Data privacy:** inspect the exported descriptions and remove or mask patient-identifying data before uploading them. If you redact locally, ensure you apply redaction consistently to the training input and regenerate the request manifest. Only upload data you are allowed to share.

The local `request_manifest.json` is needed later for verification. Do not change or delete it, and do not edit/reorder the original training file after generating the requests.

## 6. Import my paraphrases back into `Models/`

Download the CSV returned by ChatGPT. Put it in your existing `Models/` folder (for example `Models/llm_paraphrases.csv`) or provide its actual location.

Then run:

```bash
python augment_data.py chat-import --input llm_paraphrases.csv
```

With existing training data:

```bash
python augment_data.py chat-import --input llm_paraphrases.csv --train-input path/to/your/train_split.csv
```

Outputs:

```text
Models/data/augmented/train_llm.csv
Models/reports/augmentation/llm_audit.csv
Models/reports/augmentation/llm_rejected.csv
Models/reports/augmentation/llm_summary.json
```

Each accepted paraphrase is joined to the original `medical_specialty`, `sample_name`, `transcription` and `keywords` via its `source_row_id`. The script rejects unknown IDs, changed training data, changed numbers, missing/additional clinical anchor words and other suspicious changes. These **heuristics cannot prove medical equivalence**. Review accepted samples as well as rejected ones.

### Import several ChatGPT batches

If the request CSV is too large to paraphrase in a single response, I can return multiple part files, each containing different IDs from the same request set.

```bash
python augment_data.py chat-import --input llm_paraphrases_part1.csv
python augment_data.py chat-import --input llm_paraphrases_part2.csv --append
python augment_data.py chat-import --input llm_paraphrases_part3.csv --append
```

This updates **one** `train_llm.csv` without repeating the originals. A repeated accepted ID is skipped and logged. Don't rerun `chat-export` with a different base file between batches.

For more source examples later, you may increase `--limit` using the same partition and same tag before generating additional batches. Previously accepted IDs must still belong to the exported request set.

## 7. Create the combined training dataset

After `rwd` and at least one successful `chat-import`, run:

```bash
python augment_data.py combined
```

This writes:

```text
Models/data/augmented/train_combined.csv
```

Your model can now train on:

| Experiment | Path relative to `Models/` |
|---|---|
| Original | `data/splits/train_base.csv` (or your existing train split) |
| RWD | `data/augmented/train_rwd.csv` |
| LLM | `data/augmented/train_llm.csv` |
| Combined | `data/augmented/train_combined.csv` |

The validation file stays unchanged across experiments. Compare accuracy, macro-F1 and F1 for all 12 medical specialties.

**Never use `test_no_labels.csv` for augmentation, training or model selection.** Only the final chosen model should generate its predictions for that file, preserving row order.

## 8. After model selection: final training data

When you have selected the augmentation settings using held-out validation, you can regenerate augmented data using the whole labeled `train.csv` for final model fitting, without treating it as validation:

```bash
python augment_data.py rwd --train-input ../train.csv --tag final
python augment_data.py chat-export --train-input ../train.csv --tag final
```

Upload `data/chat_requests/paraphrase_requests_final.csv` here, receive `llm_paraphrases_final.csv`, then:

```bash
python augment_data.py chat-import --train-input ../train.csv --tag final --input llm_paraphrases_final.csv
python augment_data.py combined --train-input ../train.csv --tag final
```

This writes `train_rwd_final.csv`, `train_llm_final.csv` and `train_combined_final.csv`. The `--tag` must match among those commands, and the exact training input must not change between export and import.

## 9. Configuration explained

`augmentation_config.json` includes:

- `seed`: 42, for deterministic splitting and RWD.
- `validation_fraction`: 0.2 when using the optional split command.
- `source_column`: `description` by default. If changed to `transcription`, exported data is much longer and semantic validation is more difficult.
- `selection.strategy`: `all` by default (or `minority` for targeted oversampling).
- `selection.target_quantile`: 0.75, a heuristic class-count target for minority oversampling.
- `selection.max_source_rows`: 0, meaning unlimited. This exports all records with nonempty source text. An explicit `--limit N` is only for users who choose to impose a cap.
- `random_dropout.probability`: 0.05, fraction of *eligible* words sampled for deletion per attempt.
- `random_dropout.mode`: `safe`. `general` is less conservative and not recommended without inspection.
- `chat_paraphrasing.min_length_ratio`: 0.5 and `max_length_ratio`: 2.0, approximate token-count change bounds.

The quota strategy is heuristic. It does not guarantee class balancing or a particular improvement in model performance.

## 10. Files retained and files removed

The revised package deliberately keeps only local utilities:

```text
augment_data.py
augmentation_config.json
augmentation_pkg/__init__.py
augmentation_pkg/io_utils.py
augmentation_pkg/splitting.py
augmentation_pkg/random_dropout.py
augmentation_pkg/medical_validation.py
tests/test_chat_augmentation.py
README_AUGMENTATION.md
```

There is no API client, no API-key configuration, no network-dependent generation library and no LLM prompt file. The actual LLM paraphrases are generated in this conversation and returned to you as an importable CSV.

### Important limitations

1. The CSV row IDs are tied to the exact training CSV contents and order. Re-export requests if you reorder or edit the training file.
2. The protection vocabulary and clinical validation are heuristics, not a medical quality certification. Semantic hallucinations can still escape automatic checks, and valid synonym-rich paraphrases can be rejected.
3. This software has not been tested against your real `Models` code or real dataset in this chat, only against synthetic test cases. Verify it with a small real sample first.
4. Do not use these augmented datasets in validation or the unlabeled test set.
