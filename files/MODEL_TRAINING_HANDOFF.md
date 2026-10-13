# Training from the data hand-off (`train.py --handoff-dir`)

`train.py` now has an opt-in **hand-off mode**. **Without `--handoff-dir` nothing changes** (verified: stdout, `history.csv`, `validation_metrics.json`, `misclassified.csv`, `split_rows.json` are byte-identical to the original script; `birnn.py`, `bilstm.py`, `bilstm_attention.py` are untouched and hash-checked by a test).
In hand-off mode the script never parses `train.csv`, never creates a split, and never builds its own text: it reads the pipeline's ready-made files exactly as provided.

## Folder layout
```
<project>/model_handoff/                    from model_handoff.zip          (stratified/primary split)
<project>/model_handoff_extras/ or merged   from model_handoff_extras.zip   (unzip next to model_handoff: it merges the augmented files into model_handoff/)
<project>/model_handoff_group_aware/        from model_handoff_extras.zip   (group-aware split, own datasets/weights/augmented files)
```
`--handoff-dir` always names the **primary** folder. `--split group_aware` automatically uses the sibling `<name>_group_aware` folder (or `--group-aware-dir PATH`). The two families are never mixed: each folder records which split it was built from and the script refuses a mismatch. If you keep the extras unzipped elsewhere, add `--extras-dir PATH`.

## Flags (hand-off mode)
| flag | meaning |
|---|---|
| `--handoff-dir model_handoff` | enables hand-off mode |
| `--dataset NAME` | **required.** `baseline_all_fields`, `clinical_text`, `compressed_unique_text`, `diagnostic_original_keywords`, or `baseline_all_fields_word_dropout` / `baseline_all_fields_sentence_dropout` (training file = originals + augmented copies; validation/test unchanged) |
| `--split stratified` (default) / `group_aware` | which split. `grouped` is the raw-mode splitter and is rejected here |
| `--class-weight none` (default) / `inverse_sqrt` / `inverse_frequency` / `interpolated` + `--class-weight-alpha A` | weights read from `<folder>/weights/`, aligned **by class name** through `label_to_index.json`; `--class-weighting` (raw flag) is rejected |
| `--class-weight-source handoff_file` (default) / `train_rows` | see "Weights with augmented data" |
| `--max-length N` | default **1024** in hand-off mode (printed as `handoff default`); raw mode keeps 512. Use 2048 for the diagnostic dataset |

Rejected with an error (not silently ignored) in hand-off mode: `--train-csv`, `--fields`, `--strip-keyword-prefix`, `--val-fraction`, `--split grouped`, `--class-weighting`. Hand-off flags are rejected in raw mode.

## What is checked before training
SHA-256 of every file used against `handoff_manifest.json`; train (originals), validation and test rows must be **exactly** the split manifest's records **in the manifest's order** (shuffled, missing or re-sorted files are rejected); validation/test contain no augmented rows; every augmented copy maps to a TRAIN record; the originals inside an augmented file equal the plain `train.csv`; test IDs/`original_order` equal the verified original test order (398 rows). Example IDs (`TR…`, `TE…`) are never changed.

## Outputs (hand-off mode)
`run_config.json` (dataset, split, max_length and its source, class-weight choice and the vector by class name, file hashes, row counts, hyper-parameters), `validation_predictions.csv`, plus the usual `best.pt`, `history.csv`, `validation_metrics.json`, `confusion_matrix.csv`, `misclassified.csv` (its `csv_row_number` column holds the example_id; `description`/`sample_name` are empty and `text_excerpt` shows the start of the model input). `predict` writes the plain label file (`--output`, one label per line, original order) **and** `test_predictions.csv` next to it.
Both CSVs have: `example_id, source_record_id, true_label (empty for test), predicted_label, prediction_probability` (softmax probability of the predicted class). Join `validation_predictions.csv` to `model_handoff/splits/validation_subsets_primary.csv` on `source_record_id` for the duplicate/conflict subset metrics.
`predict` needs the same folder family as training: `python train.py predict --checkpoint runs/X/best.pt --handoff-dir model_handoff --output runs/X/results.txt` (dataset and split are read from the checkpoint; a mismatching `--dataset/--split`, a raw `--test-csv`, or a missing `--handoff-dir` is rejected, because text formats differ between datasets).

## Quick start (MODEL = `birnn` | `bilstm` | `bilstm_attention`; keep every other setting fixed within a comparison)
```bash
H=model_handoff
# Experiment 1: baseline_all_fields, stratified, no augmentation, no weights
python train.py train --model MODEL --handoff-dir $H --dataset baseline_all_fields --output-dir runs/e1_MODEL
# Experiment 2: clinical text only
python train.py train --model MODEL --handoff-dir $H --dataset clinical_text --output-dir runs/e2_MODEL
# Experiment 3: compressed text
python train.py train --model MODEL --handoff-dir $H --dataset compressed_unique_text --output-dir runs/e3_MODEL
# Experiment 4: baseline + word dropout (needs the project template to accept this augmentation); optional: ..._sentence_dropout
python train.py train --model MODEL --handoff-dir $H --dataset baseline_all_fields_word_dropout --output-dir runs/e4_MODEL
# Experiment 5: class weighting on the same dataset (compare e1 [none] with these two)
python train.py train --model MODEL --handoff-dir $H --dataset baseline_all_fields --class-weight inverse_sqrt     --output-dir runs/e5_sqrt_MODEL
python train.py train --model MODEL --handoff-dir $H --dataset baseline_all_fields --class-weight inverse_frequency --output-dir runs/e5_inv_MODEL
# optional finer grid: --class-weight interpolated --class-weight-alpha 0.25   (0, 0.25, 0.5, 0.75, 1)
# Final: repeat the BEST configuration on the group-aware split (identical flags + --split group_aware)
python train.py train --model MODEL --handoff-dir $H --dataset <best dataset> [<best weights>] --split group_aware --output-dir runs/final_ga_MODEL
# Predictions for results.txt (same folder family as the checkpoint)
python train.py predict --checkpoint runs/<run>/best.pt --handoff-dir $H --output runs/<run>/results.txt
# Leakage investigation ONLY (see below)
python train.py train --model MODEL --handoff-dir $H --dataset diagnostic_original_keywords --max-length 2048 --output-dir runs/diag_MODEL
```
Add `--epochs 1` to any command for a quick smoke test; `--device cuda|xpu` as before.

## Rules for comparable experiments
* **`--max-length` must stay fixed** across compared runs (default 1024 prints on every run and is stored in `run_config.json`). At 1024 about 6% of training notes still lose their last field (keywords); at 512 about 39% do. The script prints a warning when keywords are cut for >= 5% of notes. The diagnostic dataset needs 2048 or its leak is partly hidden.
* Same `--seed`, model, hyper-parameters and epochs; change one factor per comparison (dataset **or** weights **or** augmentation).
* Selection and early stopping use validation macro-F1 (unchanged). Never use `test_no_labels.csv` to evaluate or tune.
* `diagnostic_original_keywords` keeps the specialty name in `keywords`: it measures **keyword leakage**, so do not use it for final reporting or as a fair benchmark.
* Class weights cannot fix inputs that are identical but carry different labels.

## Weights with augmented data
The provided weights (`weights/class_weights_primary_split.json`, one vector per strategy) were computed from the **unaugmented** train partition. An augmented training file adds copies, which changes class frequencies, so the script prints a warning if you combine them. Recompute from the rows actually used with `--class-weight-source train_rows`:
`freq_c = n_c / N` over ALL rows of the training file (originals + copies); `inverse_frequency = 1/freq`; `inverse_sqrt = 1/sqrt(freq)`; `interpolated(alpha) = (1-alpha) + alpha/freq`; every vector divided by its mean over the 12 classes (mean = 1). For non-augmented files the recomputed vector equals the provided one (tested). To compare "augmentation" with "weights", keep the weight source the same across runs and say which you used. `train-final` should also use `train_rows` (its rows = training file + validation file).
(The weight file is named `…_primary_split.json` in both folders, but it always holds the weights of the split that folder was built from.)

## `train-final`
`train-final --handoff-dir … --dataset …` refits on the training file **plus the validation file** (originals); there is no held-out score for that model. Use it only after the configuration was chosen on validation.

## Tests
`tests/test_train_handoff.py` (44 tests): sizes for every dataset (stratified 2129/533/398; augmented 4254 and 3950 train rows; group-aware from its metadata), no mixing of folders, shuffled/missing/tampered files rejected, weights aligned by name and loss weights in model-index order, recomputed == provided weights, flag rules, `max_length` defaults, 1-epoch smoke runs for all six dataset names with source-ID/order preservation for validation and test predictions, group-aware weighted run, `train-final`, checkpoint/predict guards, raw-mode byte-identity with the original script. Run: `HANDOFF_DIR=model_handoff GROUP_AWARE_DIR=model_handoff_group_aware python -m pytest -q tests` (optional: `RAW_TRAIN_CSV=train.csv` + `train_ORIGINAL.py` enable the raw-mode regression).
The smoke runs verify plumbing only; no model was trained to convergence and no performance claim is made.
