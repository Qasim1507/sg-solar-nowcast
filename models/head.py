"""Quantile head + pinball loss.

WHY QUANTILES AND NOT GAUSSIAN NLL
----------------------------------
Given cloud now, k_t two hours later over Singapore is close to bimodal: it either
stays overcast (~0.3) or clears (~0.75). Measured: 40% of cloudy cases cleared to
k_t > 0.6 within 2h. A single Gaussian cannot represent that, and Gaussian NLL is
minimised by predicting the mean of the two modes - a value that is almost never
observed. Measured consequence in the prior project: forecast spread on cloudy
rows was 52-63% of reality's, and 90% intervals covered 81.4% on cloudy days
against 91.6% overall.

Outputs are sorted along the quantile axis so quantiles cannot cross.

EXPECT MAE TO BE SLIGHTLY WORSE than a mean-predicting model: the conditional
mean is MAE-optimal by construction, and the median is not. Judge on pinball
loss, CRPS and coverage.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    def __init__(self, in_dim: int, n_horizons: int, n_quantiles: int,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_h, self.n_q = n_horizons, n_quantiles
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, n_horizons * n_quantiles),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, H, Q), sorted ascending along Q."""
        out = self.net(z).view(-1, self.n_h, self.n_q)
        return torch.sort(out, dim=-1).values


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """pred (B,H,Q), target (B,H)."""
    qs = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    e = target.unsqueeze(-1) - pred
    return torch.mean(torch.max(qs * e, (qs - 1.0) * e))
