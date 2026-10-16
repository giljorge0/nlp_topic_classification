# Medical Specialty Classification from Clinical Notes

**Group 12** (Bernardo Rodrigues, Gil Jorge, Ricardo Ribeiro) - Natural Language project "Help the Doctor", 2026.

Given a clinical record (`description`, `sample_name`, `transcription`, `keywords`), predict one of **12 medical specialties**. We train bidirectional recurrent classifiers **from scratch** (no pre-trained language models) and ask:

> How much do LSTM gating and attention pooling improve a bidirectional vanilla RNN on long, imbalanced and partly duplicated clinical text?

Four models are compared (`birnn`, `birnn_attention`, `bilstm`, `bilstm_attention`) against a Multinomial Naive Bayes baseline, across different input-field combinations, class weighting and data augmentation. The paper is `report.pdf`.

---

## 1. Headline results

Five-fold cross-validation, input = **keywords + sample name + transcription** (our best input). Numbers are from `report.pdf` (Table 1); they are **internal CV estimates, not the official test score**.

| Model | Accuracy | Macro-F1 | Min. class F1 |
|---|---|---|---|
| Naive Bayes (TF-IDF) | 0.413 | 0.145 | 0.000 |
| BiRNN | 0.719 | 0.616 | 0.055 |
| BiLSTM | 0.876 | 0.810 | 0.392 |
| BiRNN + attention | 0.910 | 0.891 | 0.766 |
| BiLSTM + attention | 0.917 | 0.892 | 0.746 |

**Read these numbers with the keyword caveat.** In 82.9% of the training records the first keyword is the specialty name and equals the label. Removing that prefix drops BiLSTM + attention from 86.07% to 42.75% accuracy (keywords-only experiment, `report.pdf` section 6). High scores therefore partly reflect an explicit label cue in the data, which is also present in the unlabeled test file (80.4% of its records, from our data audit).

---

## 2. Repository layout

```
nlp_topic_classification/
|-- README.md                  <- this file
|-- report.pdf                 <- the paper (3 pages + appendix)
|-- train.csv                  <- NOT tracked: place the original course file here (semicolon-separated, header)
|-- test_no_labels.csv         <- NOT tracked: official test file (semicolon-separated, NO header, 398 records)
`-- Models/                    <- all code; run every command from here
    |-- train.py               <- training / cross-validation / prediction CLI *
    |-- birnn.py  birnn_attention.py  bilstm.py  bilstm_attention.py   <- the four models
    |-- baseline.py            <- Naive Bayes baseline (TF-IDF, lowercase + Porter stemming)
    |-- compare_results.py     <- builds the Results/ dashboard from runs/
    |-- export_misclassifications.py
    |-- augment_data.py  augmentation_config.json  augmentation_pkg/   <- word dropout + LLM-paraphrase import
    |-- augmentation_cv_core.py  augmentation_cv_models.py  train_augmentation_data.py *   <- leakage-safe augmentation CV
    |-- llm_paraphrases_batch_1.csv  llm_paraphrases_batch_2.csv       <- paraphrases returned by the LLM (accepted ones are imported)
    |-- data/
    |   |-- splits/            train_base.csv, validation.csv           <- split used for the augmentation experiments
    |   |-- augmented/         train_rwd(.csv|_final.csv), train_llm_final.csv, train_combined_final.csv
    |   `-- chat_requests/     paraphrase_requests_final.csv + request_manifest_final.json
    |-- reports/
    |   |-- augmentation/      class distributions, per-method audits / rejected rows / summaries, split_manifest.json
    |   `-- augmentation_cv/   data_quality_quarantine.csv, folds/fold_01..05, original_models/ (checkpoints + per-fold predictions)
    |-- Results/               <- generated dashboard; open Results/index.html
    |   |-- Metrics/ Models_Comparison/ TrainingSet_Comparison/ Weighting_Comparison/ Confusion_Matrices/ Augmentation/
    |-- cache/                 <- generated *.blob files (not documented elsewhere; safe to regenerate, do not submit)
    |-- model_files.sha256     <- checksums of the model definition files
    |-- requirements.txt *  tests/ *
```

`*` = referenced by the documentation but outside the part of the file tree that was supplied when this README was written; check that the path is right.

Other documents (all in `Models/` unless you moved them):

| File | What it covers |
|---|---|
| `README_1.md` (quick start) | 5-fold CV commands, input options, Naive Bayes baseline, dashboard, final prediction |
| `README_AUGMENTATION.md` | local Random Word Dropout and the ChatGPT-paraphrase import workflow (no API keys) |
| `README_LEAKAGE_SAFE_CV.md` | augmentation CV in which synthetic rows can only enter their source's training folds |
| `MODEL_TRAINING_HANDOFF.md` | alternative fixed-split protocol: `train.py --handoff-dir ...` on pre-built datasets |
| `analysis_report.md` | auto-generated data audit (corruption, duplicates, keyword leakage, length statistics) |

---

## 3. Setup

```bash
python -m pip install -r Models/requirements.txt     # torch>=2.6,<3, numpy>=1.24, scikit-learn>=1.4
cd Models
```

Put `train.csv` and `test_no_labels.csv` in the repository root and refer to them as `../train.csv` / `../test_no_labels.csv` from `Models/` (the augmentation documents use this); if you prefer the `README_1.md` style, copy both files into `Models/`. Add `--device xpu` (Intel GPU) or `--device cuda` to any training command; otherwise the device is chosen automatically.

---

## 4. Reproducing the experiments

### 4.1 Neural models, 5-fold cross-validation

```bash
python train.py make-folds --train-csv ../train.csv
python train.py cv --model bilstm_attention --train-csv ../train.csv --fields keywords sample_name transcription
```

Models: `birnn`, `birnn_attention`, `bilstm`, `bilstm_attention`. Input options (`--fields ...`, `--strip-keyword-prefix`, `--keep-keyword-prefix-only`) are listed in `README_1.md`; the default uses all four fields. Folds are identical across models (stratified, identical transcriptions grouped), and early stopping uses an inner validation split (about 64% train / 16% validation / 20% held-out per fold). Completed folds are skipped; add `--overwrite` to rerun. Use `python train.py cv --help` for class weighting (`sqrt_inv`) and the other options.

### 4.2 Naive Bayes baseline

```bash
python baseline.py --train-csv ../train.csv                      # course baseline: description only
python baseline.py --train-csv ../train.csv --fields keywords sample_name transcription
```

### 4.3 Compare everything

```bash
python compare_results.py        # then open Results/index.html
```

### 4.4 Data augmentation and leakage-safe evaluation

Follow `README_AUGMENTATION.md` (word dropout; LLM paraphrases generated in a chat and imported with `chat-import`) and then `README_LEAKAGE_SAFE_CV.md` (`train_augmentation_data.py prepare|cv`) so that no synthetic copy of a held-out note is ever trained on. The LLM set is **partial** (551 accepted paraphrases from two batches against 2,612 requested), and by default only the `description` field is augmented (`source_column` in `augmentation_config.json`).

### 4.5 Fixed-split hand-off protocol (optional)

`MODEL_TRAINING_HANDOFF.md` describes training from pre-built, hash-verified datasets with a fixed stratified or group-aware split (`train.py train --handoff-dir model_handoff --dataset ...`). Its numbers come from a **different evaluation protocol** than the 5-fold CV and must not be mixed with the Table 1 results.

### 4.6 Final predictions (`results.txt`)

Retrain the chosen configuration on all labelled data (use an epoch count selected in CV) and predict the official test file:

```bash
python train.py train-final --model bilstm_attention --train-csv ../train.csv --fields keywords sample_name transcription --epochs 15 --output-dir runs/final
python train.py predict --checkpoint runs/final/final.pt --test-csv ../test_no_labels.csv --output ../results.txt
```

`results.txt` must have **one label per line, in the original test order, no header** (398 lines). Never use `test_no_labels.csv` for training, tuning or model selection.

---

## 5. Data issues you should know about

* **The raw `train.csv` is structurally damaged.** A quote-aware parser returns 2,616 rows, three of them text fragments instead of labels; seven records open a quotation mark that is never closed and swallow 56 following records. Our data audit rebuilt 2,662 usable records and quarantined 7; the CV code in `Models/` quarantines the 10 suspicious rows it can detect (3 malformed labels + 7 oversized rows) and works with the parser's 2,613 valid rows, so **the two pipelines report different record counts**. Do not compare counts across them without saying which one is meant.
* **Imbalance:** Surgery is about 35 times larger than Dermatology (876 vs 25 records among the 2,613 valid rows).
* **Duplicates with conflicting labels:** identical transcriptions occur under several specialties (most often Surgery with Orthopedic, Cardiovascular-Pulmonary or Gastroenterology). Such inputs cannot all be classified correctly, so they bound the achievable score. Identical transcriptions are grouped when folds are built, but near-duplicates are not.
* **Label cue in keywords:** see section 1.

---

## 6. Submission checklist (deadline 16 October 2026, 23:59)

The zip must be named after the group number and be a **zip, not a rar**:

```
12.zip
|-- 12.pdf          the paper (max 3 pages + 1 page appendix)
|-- results.txt     398 lines, one specialty per line, no header
`-- code + data     the project code and the datasets we gathered (e.g. the LLM paraphrase CSVs); NOT the models
```

* Leave out `*.pt` checkpoints (many sit under `reports/augmentation_cv/original_models/`), `cache/`, `runs/` and any `Results/` output you do not need.
* Check `results.txt`: `wc -l results.txt` prints 398, every line is one of the 12 specialty names.
* The paper must run without the appendix, declare LLM usage, and contain no leftover template text.

---

## 7. Reproducibility notes

* Default seed 42 everywhere; folds are fixed and shared by all models.
* `model_files.sha256` holds checksums of the model definition files, so unintended edits to the architectures can be detected.
* Pooled out-of-fold predictions and per-fold metrics for every run are kept under `runs/` and `Results/Metrics/`.
* LLM assistance (code, debugging, English, and the paraphrases used for augmentation) is declared in `report.pdf`.
