# Help the Doctor: leakage-safe augmentation CV

This is an **add-on**, not a replacement for `train.py` or `augment_data.py`.

Your existing augmentation package emits:

- `data/augmented/train_rwd_final.csv`
- `data/augmented/train_llm_final.csv`
- `data/augmented/train_combined_final.csv`
- `reports/augmentation/rwd_final_audit.csv`
- `reports/augmentation/llm_final_audit.csv`

**Keep all five files!** The audits are essential because the training CSVs contain no source IDs. The new script checks them against your original `../train.csv`.

## Install

Extract the three `.py` files, this README and `tests/test_safe_cv.py` into `Models/`, preserving the `tests/` subfolder. Do not overwrite existing `train.py`, `augment_data.py`, `augmentation_config.json` or anything in `augmentation_pkg/`.

From your **existing active `.venv`** in `Models/`:

```bat
python -c "import sklearn, numpy, torch; print('Ready')"
```

If sklearn or numpy are missing, install them with `python -m pip install scikit-learn numpy`. To use the independent BiLSTM-attention runner, PyTorch also needs to be installed in your environment. The fold-preparation and audit operations need only numpy and scikit-learn in addition to the previously installed augmentation files.

Run tests:

```bat
python -m unittest discover -s tests -p "test_safe_cv.py" -v
```

## 1. Review the data-quality quarantine

The supplied 2,616-record `train.csv` has 15 distinct specialty labels even though the project expects 12, and seven descriptions are abnormally long. We detected **10 distinct suspicious original records** (3 malformed labels + 7 long descriptions).

**Default behavior stops instead of automatically changing the dataset.** To explicitly quarantine suspicious originals and their synthetic children while making an audit CSV:

```bat
python train_augmentation_data.py audit --train-csv "../train.csv" --quarantine-suspicious --fields description keywords transcription --strip-keyword-prefix
```

Review `reports/augmentation_cv/data_quality_quarantine.csv`. This exclusion is a cautious temporary measure, not a repair of the source dataset. If you replace/fix `train.csv`, you must regenerate the original augmentation CSVs and audits, since source IDs and ordering can change.

## 2. Generate leak-free cross-validation folds

```bat
python train_augmentation_data.py prepare --train-csv "../train.csv" --quarantine-suspicious --folds 5 --fields description keywords transcription --strip-keyword-prefix
```

Output, for each fold:

```
Models/
  reports/
    augmentation_cv/
      data_quality_quarantine.csv
      folds/
        cv_manifest.json
        fold_01/
          train_baseline.csv
          train_rwd.csv
          train_llm.csv
          train_combined.csv
          validation.csv
          validation_provenance.csv
          manifest.json
        fold_02/
          ...
        ...fold_05/
```

Each `train_*` CSV contains **only source records belonging to that fold's original training split**, plus augmentations of those training records. `validation.csv` contains untouched original records only. Identical full transcriptions are grouped to prevent exact clinical-note leakage across folds.

## 3. Run the complete leakage-free experiments (no changes to train.py)

For a quick diagnostic classifier that exercises all four datasets:

```bat
python train_augmentation_data.py cv --model linear_svm --train-csv "../train.csv" --quarantine-suspicious --folds 5 --fields description keywords transcription --strip-keyword-prefix
```

For the provided **independent reference BiLSTM with attention** (PyTorch), use:

```bat
python train_augmentation_data.py cv --model birnn_attention --train-csv "../train.csv" --quarantine-suspicious --folds 5 --fields description keywords transcription --strip-keyword-prefix --epochs 5 --batch-size 32 --max-tokens 256 --device auto
```

`cv` runs **baseline, RWD, LLM and Combined** using exactly the same original fold assignments and hyperparameters. For each fold, token vocabulary/TF-IDF parameters are learned from training examples only. The held-out fold is not used for early stopping or model tuning; the epoch count is fixed.

The reference BiLSTM model is **not the model defined in your original `train.py`**. It is a self-contained implementation intended to yield a valid augmentation comparison when the original `train.py` is unavailable. Its architecture is a trainable word embedding, bidirectional single-layer LSTM, additive attention and linear classifier. Do not claim these results came from your original BiRNN implementation if its architecture or preprocessing differs.

You can limit to one of the variants, e.g. `--methods baseline rwd`, if needed. Use `--model linear_svm` for an independent check, not as a replacement for your principal neural experiments.

## 4. Results

The script writes under `reports/augmentation_cv/<model>/`:

- `comparison.csv`: accuracy, macro-F1, weighted-F1 and minimum per-specialty F1 for every variant
- `fold_metrics.csv`: per-fold accuracy and macro-F1
- `per_class_<variant>.csv`: precision, recall, F1 and support for all 12 specialties
- `confusion_<variant>.csv`: 12x12 confusion matrix
- `oof_<variant>.csv`: one out-of-fold prediction for **each original, eligible row**
- `run_config.json`: settings used

The project's score rule checks accuracy strictly greater than 49% and every specialty's F1 at least 25%. Use pooled out-of-fold metrics for comparison. These are **internal CV estimates**, not the hidden test score.

## 5. If you want to evaluate your original `train.py` BiRNN exactly

Do **not** execute `python train.py cv --train-csv data/augmented/train_combined_final.csv ...`. Its ordinary row-based CV may distribute an original note and synthetic variant into different folds.

Instead, integrate the prepared fold files into a **single train/validation run** of your own model for each method and fold:

- Read `reports/augmentation_cv/folds/fold_01/train_<variant>.csv` as training data.
- Read `reports/augmentation_cv/folds/fold_01/validation.csv` as the only validation/evaluation data.
- Fit a **fresh** model, tokenizer, vocabulary and any preprocessing on the training file only, then predict the unchanged validation records.
- Repeat folds 01-05 and aggregate the predictions. Do not internally resplit or run another `cv` on `train_<variant>.csv`.
- Keep all original model hyperparameters and random seeds as comparable as possible between variants. No augmentation of validation data.

The existing model's exact training interface cannot be safely patched without reviewing your actual `train.py` and its imports. This package therefore does not modify it or pretend that it supports `--validation-csv`.

## 6. Why this avoids augmentation leakage

Original note groups define the folds. Audited augmentation provenance attaches each synthetic row to a source record. A synthetic row can enter only its original's **training folds**, never a fold where that original is held out. Every validation record is from the original source only. The system fails closed on missing audits, unmatched source IDs, changed original records or inconsistent combined data.

Note: grouping exact normalized transcriptions cannot guarantee that every **near-duplicate** clinical note is detected. For especially stringent evaluation, review repeated near-duplicate records manually or add a near-duplicate grouping process **before** choosing folds.

### Existing dataset status

The LLM augmentation currently contains 551 accepted examples from two batches. It is partial relative to the 2,612 original requests. The comparison is valid for that explicitly partial LLM set if this is stated in the report. Do not describe it as exhaustive LLM paraphrasing.

### Reproducibility

Keep the same folds, same classifier settings and same dataset version for all four experiments. For results reported in the paper, archive the output's manifests and source/hyperparameter configuration. Fold CSVs are comma-separated, irrespective of the original input delimiter, and retain exactly the original five classifier columns.
