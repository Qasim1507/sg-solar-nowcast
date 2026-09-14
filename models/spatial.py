"""Spatial branch: a deliberately small CNN encoder over the rain-gauge raster.

PARAMETER BUDGET
----------------
The prior attempt put 14.4M image-branch parameters against ~5,300 training
samples and overfit badly. Track A here has ~13k training rows, and the rain
field is extremely sparse (a typical midday frame has 0-1 wet gauges), so the
budget is <1M parameters. `n_params()` is asserted in tests/test_model.py.

Output is a sequence of patch embeddings so the temporal summary can attend over
locations, rather than a single pooled vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(cin: int, cout: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, channels=(16, 32, 64), dropout: float = 0.1,
                 out_dim: int = 128):
        super().__init__()
        blocks, cin = [], in_channels
        for cout in channels:
            blocks.append(conv_block(cin, cout, dropout))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(cin, out_dim, 1)
        self.out_dim = out_dim
        self.pool = nn.AdaptiveAvgPool2d(4)        # -> 16 patches regardless of grid size

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """grid (B, C, H, W) -> patches (B, P, out_dim)."""
        z = self.blocks(grid)
        z = self.proj(z)
        z = self.pool(z)                            # (B, D, 4, 4)
        return z.flatten(2).transpose(1, 2)         # (B, 16, D)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
