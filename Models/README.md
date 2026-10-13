# Medical Specialty Classification — Quick Start

Put `train.csv` and `test_no_labels.csv` in this folder, alongside the Python files. Run all commands from here.

## 1. Setup (once)

```cmd
pip install -r requirements.txt
python train.py make-folds --train-csv "train.csv"
```

## 2. Train the four models (5-fold cross-validation)

```cmd
python train.py cv --model birnn --train-csv "train.csv"
python train.py cv --model birnn_attention --train-csv "train.csv"
python train.py cv --model bilstm --train-csv "train.csv"
python train.py cv --model bilstm_attention --train-csv "train.csv"
```

By default, all four input fields are used: `description + sample_name + transcription + keywords`.

## 3. Choose different inputs

Add **one** of these options to any `cv` command:

| Input | Option |
| --- | --- |
| Keywords only | `--fields keywords` |
| Keywords without specialty prefix | `--fields keywords --strip-keyword-prefix` |
| Prefix only (all keywords if none found) | `--fields keywords --keep-keyword-prefix-only` |
| Transcription only | `--fields transcription` |
| Description only | `--fields description` |
| No keywords | `--fields description sample_name transcription` |

Example:

```cmd
python train.py cv --model bilstm_attention --train-csv "train.csv" --fields keywords --strip-keyword-prefix
```

To override training settings, append `--epochs 30 --patience 6` (early stopping uses inner-validation macro-F1). Append `--device xpu` for a working Intel XPU environment. Completed folds are skipped automatically; append `--overwrite` to rerun them.

## 4. Classical baseline (description only)

```cmd
python baseline.py --train-csv "train.csv"
```

Uses the same five outer folds. Fits TF-IDF (lowercase + Porter stemming, word unigrams) and Multinomial Naive Bayes. Saves to `runs/cv/description_only/naive_bayes/`.

## 5. View results

```cmd
python compare_results.py
```

Open **`Results/index.html`** in a browser. Find the Naive Bayes baseline alongside the neural models in `Models_Comparison/description_only.html`; other model comparisons are in `Models_Comparison/`, input comparisons in `TrainingSet_Comparison/`, and each class's F1, precision, recall, and confusion matrices in `Metrics/`. The tables show five-fold averages ± standard deviation and pooled out-of-fold results. Raw checkpoints and predictions are retained in `runs/`.

## 6. Final predictions (after choosing the best model)

Retrain your chosen configuration on all labelled training data, using an epoch count selected from cross-validation (15 below is an example):

```cmd
python train.py train-final --model bilstm_attention --train-csv "train.csv" --fields transcription --epochs 15 --output-dir runs/final
python train.py predict --checkpoint runs/final/final.pt --test-csv "test_no_labels.csv" --output results.txt
```

Submit `results.txt` (one label per test row, no header). **Never use the official unlabeled test file for training or model selection.**
