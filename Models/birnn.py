# Bidirectional vanilla RNN

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 12,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if min(vocab_size, embedding_dim, hidden_dim, num_classes, num_layers) < 1:
            raise ValueError("Vocabulary, dimensions, classes and layers must be positive :)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be inside [0, 1)")

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or lengths.ndim != 1 or len(lengths) != len(input_ids):
            raise ValueError("Expected input_ids [batch, tokens] and lengths [batch]")
        if (lengths < 1).any().item() or (lengths > input_ids.size(1)).any().item():
            raise ValueError("Every sequence length must lie between 1 and tokens")

        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.encoder(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=input_ids.size(1)
        )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        mask = positions.unsqueeze(0) < lengths.to(input_ids.device).unsqueeze(1)
        pooled = (outputs * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / lengths.to(input_ids.device).unsqueeze(1).to(outputs.dtype)
        return self.classifier(self.dropout(pooled))
