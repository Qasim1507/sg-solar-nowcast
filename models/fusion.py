"""Full multimodal model: BiLSTM + CNN + cross-attention + physics gate + quantiles.

    tabular_seq (L, F) --> BiLSTM ----------------------> H_t
    rain_grid (C, H, W) --> CNN --> patches --> xattn(H_t, patches) --> H_a
                                                          |
    gate_features -------> MLP --> alpha (sigmoid) -------+
                                                          v
                              fused = alpha*H_t + (1-alpha)*H_a
                    concat [fused, future_clearsky(3), nwp(3*k)]
                                                          v
                                                   quantile head

Variants (the ablation):
    temporal : BiLSTM only
    spatial  : CNN only
    concat   : both, concatenated (no gate)
    gated    : both, physics-gated fusion (full model)

The head ALWAYS receives future clear-sky GHI and, when available, NWP at the
target hours - those are the physics-known-ahead inputs and are not part of the
temporal/spatial ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.gate import PhysicsGate
from models.head import QuantileHead
from models.spatial import SpatialEncoder
from models.temporal import TemporalEncoder

VARIANTS = ("temporal", "spatial", "concat", "gated")


class MultimodalNowcaster(nn.Module):
    def __init__(self, n_features: int, n_grid_channels: int, n_horizons: int,
                 n_quantiles: int, n_nwp: int = 0, variant: str = "gated",
                 cfg: dict | None = None, target_space: str = "ghi"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
        if target_space not in ("ghi", "kt"):
            raise ValueError(f"target_space must be 'ghi' or 'kt', got {target_space!r}")
        self.variant = variant
        self.target_space = target_space
        mc = (cfg or {}).get("model", {})
        tcfg = mc.get("temporal", {})
        scfg = mc.get("spatial", {})
        fcfg = mc.get("fusion", {})
        gcfg = mc.get("gate", {})
        hcfg = mc.get("head", {})
        dim = fcfg.get("dim", 128)

        self.use_temporal = variant in ("temporal", "concat", "gated")
        self.use_spatial = variant in ("spatial", "concat", "gated")

        if self.use_temporal:
            self.temporal = TemporalEncoder(
                n_features, tcfg.get("hidden", 64), tcfg.get("layers", 1),
                tcfg.get("dropout", 0.1))
            self.t_proj = nn.Linear(self.temporal.out_dim, dim)
        if self.use_spatial:
            self.spatial = SpatialEncoder(
                n_grid_channels, tuple(scfg.get("channels", (16, 32, 64))),
                scfg.get("dropout", 0.1), out_dim=dim)

        # cross-attention only makes sense when both branches exist
        self.use_xattn = self.use_temporal and self.use_spatial
        if self.use_xattn:
            self.xattn = nn.MultiheadAttention(dim, fcfg.get("n_heads", 4), batch_first=True)
            self.xnorm = nn.LayerNorm(dim)
        elif self.use_spatial:
            self.spatial_pool = nn.Linear(dim, dim)

        self.use_gate = variant == "gated"
        if self.use_gate:
            self.gate = PhysicsGate(4, gcfg.get("hidden", 32))

        fused_dim = dim * 2 if variant == "concat" else dim
        head_in = fused_dim + n_horizons + n_nwp
        self.head = QuantileHead(head_in, n_horizons, n_quantiles,
                                 hcfg.get("hidden", 128), hcfg.get("dropout", 0.1))
        self.last_alpha: torch.Tensor | None = None

    def forward(self, batch: dict) -> torch.Tensor:
        parts = []
        h_t = h_a = None

        if self.use_temporal:
            summary, _ = self.temporal(batch["seq"])
            h_t = self.t_proj(summary)

        if self.use_spatial:
            patches = self.spatial(batch["grid"])
            if self.use_xattn:
                q = h_t.unsqueeze(1)                       # (B, 1, D)
                attn, _ = self.xattn(q, patches, patches)
                h_a = self.xnorm(attn.squeeze(1))
            else:
                h_a = self.spatial_pool(patches.mean(dim=1))

        if self.variant == "temporal":
            fused = h_t
        elif self.variant == "spatial":
            fused = h_a
        elif self.variant == "concat":
            fused = torch.cat([h_t, h_a], dim=-1)
        else:                                              # gated
            alpha = self.gate(batch["gate"])               # (B, 1)
            self.last_alpha = alpha.detach()
            fused = alpha * h_t + (1.0 - alpha) * h_a

        parts.append(fused)
        parts.append(batch["future_cs"])
        if batch["nwp"].shape[-1] > 0:
            parts.append(batch["nwp"])
        out = self.head(torch.cat(parts, dim=-1))
        if self.target_space == "kt":
            # The head emits a clear-sky INDEX; multiply by clear-sky at the target
            # hour to get W/m2. The network no longer has to re-derive the diurnal
            # and seasonal envelope from sin/cos hour+month - that envelope is
            # already exact in future_cs_raw. The loss stays in W/m2, so val/test
            # numbers remain directly comparable with the ghi-space runs.
            # cs >= 0, so the ascending sort from QuantileHead is preserved.
            out = out * batch["future_cs_raw"].unsqueeze(-1)
        return out

    def n_params(self) -> dict:
        def cnt(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "temporal": cnt(getattr(self, "temporal", None)),
            "spatial": cnt(getattr(self, "spatial", None)),
            "head": cnt(self.head),
        }
