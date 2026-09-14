"""Temporal branch: BiLSTM over the tabular lookback window."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_dim = hidden * 2
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """seq (B, L, F) -> (summary (B, 2H), steps (B, L, 2H))."""
        steps, _ = self.lstm(seq)
        summary = self.norm(steps[:, -1])          # last step = time t
        return self.drop(summary), steps
