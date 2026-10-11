# Commands to Run

## 1. Install dependencies and test the models

```bash
python -m pip install -r requirements.txt
```

## 2. Train all models (all input fields)

```bash
python train.py train --model birnn --train-csv "train.csv" --output-dir runs/birnn
python train.py train --model bilstm --train-csv "train.csv" --output-dir runs/bilstm
python train.py train --model bilstm_attention --train-csv "train.csv" --output-dir runs/bilstm_attention
```

By default, the naive inputs are concatenated as `description + sample_name + transcription + keywords`.

## 3. Try different inputs

These examples use `birnn`; replace it with `bilstm` or `bilstm_attention` to test the others.

**Keywords only:**
```bash
python train.py train --model birnn --train-csv "train.csv" --output-dir runs/birnn_keywords --fields keywords
```

**Keywords without the specialty prefix:**
```bash
python train.py train --model birnn --train-csv "train.csv" --output-dir runs/birnn_keywords_clean --fields keywords --strip-keyword-prefix
```

**Transcription only:**
```bash
python train.py train --model birnn --train-csv "train.csv" --output-dir runs/birnn_transcription --fields transcription
```

**Description only:**
```bash
python train.py train --model birnn --train-csv "train.csv" --output-dir runs/birnn_description --fields description
```

**Description + sample name + transcription:**
```bash
python train.py train --model birnn --train-csv "train.csv" --output-dir runs/birnn_no_keywords --fields description sample_name transcription
```

## 4. Get predictions from a trained model

```bash
python train.py predict --checkpoint runs/birnn/best.pt --test-csv "test_no_labels.csv" --output runs/birnn/predictions.txt
python train.py predict --checkpoint runs/bilstm/best.pt --test-csv "test_no_labels.csv" --output runs/bilstm/predictions.txt
python train.py predict --checkpoint runs/bilstm_attention/best.pt --test-csv "test_no_labels.csv" --output runs/bilstm_attention/predictions.txt
```

## Useful options

If you have Intel GPU, feel free to append `--device xpu` to a training command, or `--device cuda` if you have a supported NVIDIA GPU. 
The default training limit is 12 epochs; append `--epochs 20` or else to change it.


Results (validation metrics, confusion matrix, misclassified examples, and best checkpoint) are saved in each `runs/...` folder. 
**Do not use `test_no_labels.csv` to evaluate or tune models.**
