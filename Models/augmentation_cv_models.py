"""Independent reference classifiers for leakage-safe augmentation experiments.

IMPORTANT: the BiLSTM attention here is a reference implementation, not the
user's existing train.py model. It never uses the held-out fold for optimization.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

import numpy as np


def svm_predict(train_texts: list[str], train_labels: list[str],
                validation_texts: list[str], *, seed: int) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    # Tfidf learns vocabulary and IDF from THIS FOLD'S train texts only.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                 max_features=80000, dtype=np.float32)
    x_train = vectorizer.fit_transform(train_texts)
    x_valid = vectorizer.transform(validation_texts)
    model = LinearSVC(class_weight="balanced", random_state=seed)
    model.fit(x_train, train_labels)
    return list(model.predict(x_valid))


TOKEN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str, max_tokens: int):
    return [x.casefold() for x in TOKEN_RE.findall(text)[:max_tokens]]


def birnn_predict(train_texts: list[str], train_labels: list[str],
                  validation_texts: list[str], class_labels: list[str], *,
                  seed: int, epochs: int, batch_size: int, max_tokens: int,
                  max_vocab: int, embedding_dim: int, hidden_dim: int,
                  learning_rate: float, device_name: str) -> list[str]:
    """Train from scratch on train records and predict once on held-out originals.

    Fixed epochs, no early stopping or hyperparameter selection on held-out folds.
    Vocabulary is learned solely from the fold's training data.
    """
    try:
        import torch
        from torch import nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch missing. Activate your model .venv and run: pip install torch") from exc

    if epochs < 1 or batch_size < 1 or max_tokens < 4:
        raise ValueError("epochs and batch_size must be >= 1 and max_tokens >= 4")
    if hidden_dim < 1 or embedding_dim < 1 or max_vocab < 100:
        raise ValueError("embedding_dim and hidden_dim must be positive, max_vocab >= 100")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else
                          ("cpu" if device_name == "auto" else device_name))

    train_tokens = [tokenize(text, max_tokens) or ["[empty]"] for text in train_texts]
    val_tokens = [tokenize(text, max_tokens) or ["[empty]"] for text in validation_texts]
    counts = Counter(word for sentence in train_tokens for word in sentence)
    ordered_vocab = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_vocab-2]
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({word: i + 2 for i, (word, _) in enumerate(ordered_vocab)})

    def encode(tokens):
        return [vocab.get(token, 1) for token in tokens]

    train_arrays = [encode(tokens) for tokens in train_tokens]
    val_arrays = [encode(tokens) for tokens in val_tokens]
    label_to_id = {name: i for i, name in enumerate(class_labels)}
    train_targets = [label_to_id[label] for label in train_labels]

    class NoteDataset(Dataset):
        def __init__(self, sequences, targets=None):
            self.sequences = sequences
            self.targets = targets
        def __len__(self):
            return len(self.sequences)
        def __getitem__(self, index):
            return (self.sequences[index], None if self.targets is None else self.targets[index])

    def collate(batch):
        lengths = [len(seq) for seq, _ in batch]
        padded = torch.zeros((len(batch), max(lengths)), dtype=torch.long)
        for i, (seq, _) in enumerate(batch):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        labels = None if batch[0][1] is None else torch.tensor([item[1] for item in batch], dtype=torch.long)
        return padded, torch.tensor(lengths, dtype=torch.long), labels

    class BiLSTMAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                               bidirectional=True, num_layers=1)
            self.attention = nn.Linear(2 * hidden_dim, 1)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(2 * hidden_dim, len(class_labels))

        def forward(self, words, lengths):
            embedded = self.dropout(self.embedding(words))
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            packed_out, _ = self.rnn(packed)
            states, _ = pad_packed_sequence(packed_out, batch_first=True,
                                             total_length=words.shape[1])
            token_positions = torch.arange(words.shape[1], device=words.device)[None, :]
            mask = token_positions < lengths[:, None]
            energies = self.attention(states).squeeze(-1).masked_fill(~mask, -1e9)
            weights = torch.softmax(energies, dim=1)
            context = torch.sum(states * weights.unsqueeze(-1), dim=1)
            return self.classifier(self.dropout(context))

    model = BiLSTMAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training-set labels only. Square-root inverse frequency avoids extreme weights
    # when the smallest medical specialties have relatively few originals.
    label_counts = Counter(train_targets)
    weights = [math.sqrt(len(train_targets) / (len(class_labels) * max(1, label_counts[i])))
               for i in range(len(class_labels))]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device, dtype=torch.float32))
    train_loader = DataLoader(NoteDataset(train_arrays, train_targets), batch_size=batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(NoteDataset(val_arrays), batch_size=batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate)
    for epoch in range(epochs):
        model.train()
        loss_total, n_examples = 0.0, 0
        for tokens, lengths, targets in train_loader:
            tokens, lengths, targets = tokens.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = loss_fn(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(targets)
            n_examples += len(targets)
        print(f"    Epoch {epoch + 1}/{epochs}: training loss {loss_total / n_examples:.4f}", flush=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for tokens, lengths, _ in val_loader:
            logits = model(tokens.to(device), lengths.to(device))
            predictions.extend(class_labels[i] for i in logits.argmax(dim=1).cpu().tolist())
    return predictions
