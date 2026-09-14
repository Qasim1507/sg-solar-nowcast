"""Physics gate: how much to trust the temporal branch vs the spatial branch.

Inputs are [clearsky_ratio, cloud_cover, motion_vx, motion_vy]. alpha is a
sigmoid, and the fusion is  alpha * H_t + (1 - alpha) * H_a.

COLLAPSE IS REPORTED, NOT HIDDEN. evaluate.py logs the distribution of alpha; if
its range spans under `eval.gate_collapse_threshold` of [0, 1] the gate has
degenerated into a constant and the model is effectively single-branch. That is a
finding to state, not a bug to paper over.

Note on the motion inputs: Singapore rain fields are sparse enough that the
cross-correlation displacement is undefined for ~96% of frames (measured), so
motion_vx/vy are 0 most of the time and the gate leans on k_t and cloud cover.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsGate(nn.Module):
    def __init__(self, n_inputs: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, gate_feats: torch.Tensor) -> torch.Tensor:
        """(B, n_inputs) -> alpha (B, 1) in (0, 1)."""
        return torch.sigmoid(self.net(gate_feats))
